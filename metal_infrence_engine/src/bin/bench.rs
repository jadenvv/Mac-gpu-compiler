// Benchmarks our engine on the exact prompt the MLX baseline used and records the result
// so mlx_baseline/compare.py can score it against MLX.
//
// The prompt token ids come from mlx_baseline/results/mlx_<chip>_<prompt>.json (we have
// no tokenizer), so run `uv run bench.py` in mlx_baseline once first.
//
// run with: cargo run --release --bin bench -- --engine metal --label "what I changed"
//   or just: cd mlx_baseline && uv run compare.py --run metal --label "what I changed"
//
// writes mlx_baseline/results/engine_<engine>_<chip>_<prompt>.json (latest run) and appends
// one line to mlx_baseline/results/history_<chip>_<prompt>.jsonl (every run, for the chart).

use metal_infrence_engine::cpu_inference::{Config, KVCache, Model};
use metal_infrence_engine::metal_inference::{MetalKVCache, MetalModel};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/qwen3-0.6b/model.safetensors");
const RESULTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/mlx_baseline/results");

// both engines behind one interface: fresh cache + one forward step
trait Engine
{
    fn reset(&mut self);
    fn step(&mut self, token: u32, pos: usize) -> Vec<f32>;
    fn model(&self) -> &Model;
}

struct Cpu { model: Model, cache: KVCache }
impl Engine for Cpu
{
    fn reset(&mut self) { self.cache = KVCache::new(&self.model.config); }
    fn step(&mut self, token: u32, pos: usize) -> Vec<f32> { self.model.forward(token, pos, &mut self.cache) }
    fn model(&self) -> &Model { &self.model }
}

struct Metal { model: MetalModel, cache: MetalKVCache, max_seq: usize }
impl Engine for Metal
{
    fn reset(&mut self) { self.cache = MetalKVCache::new(&self.model, self.max_seq); }
    fn step(&mut self, token: u32, pos: usize) -> Vec<f32> { self.model.forward(token, pos, &mut self.cache) }
    fn model(&self) -> &Model { &self.model.cpu }
}

struct Args { engine: String, prompt: String, runs: usize, decode_steps: usize, label: String }

fn parse_args() -> Args
{
    let mut a = Args { engine: "metal".into(), prompt: "ai".into(), runs: 3, decode_steps: 32, label: String::new() };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next()
    {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str()
        {
            "--engine" => a.engine = val(),
            "--prompt" => a.prompt = val(),
            "--runs" => a.runs = val().parse().expect("--runs must be a number"),
            "--decode-steps" => a.decode_steps = val().parse().expect("--decode-steps must be a number"),
            "--label" => a.label = val(),
            _ => panic!("unknown flag {flag} (use --engine cpu|metal, --prompt, --runs, --decode-steps, --label)"),
        }
    }
    assert!(a.engine == "cpu" || a.engine == "metal", "--engine must be cpu or metal");
    a
}

fn sh(cmd: &str, args: &[&str]) -> String
{
    Command::new(cmd).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

// same slug as bench.py: "Apple M1 Pro" -> "m1_pro"
fn chip_slug(chip: &str) -> String { chip.replace("Apple ", "").replace(' ', "_").to_lowercase() }

fn argmax(x: &[f32]) -> u32
{
    let mut best = 0;
    for i in 1..x.len() { if x[i] > x[best] { best = i; } }
    best as u32
}

// same rule as tests/mbu.rs: every tensor except the embedding table, plus one embedding row
fn weight_bytes_per_token(model: &Model) -> usize
{
    let mut bytes = 0;
    for t in &model.tensors
    {
        if t.name == "model.embed_tokens.weight" { continue; }
        bytes += (t.data_offsets[1] - t.data_offsets[0]) as usize;
    }
    bytes + model.config.hidden_size * 2
}

// f32 K and V for every cached position (0..=pos) in every layer
fn kv_bytes_at(c: &Config, pos: usize) -> usize { 2 * c.num_hidden_layers * (pos + 1) * c.num_key_value_heads * c.head_dim * 4 }

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

fn median(xs: &[f64]) -> f64
{
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

// fresh cache, token-by-token prefill, pick the first token
fn ttft(e: &mut dyn Engine, prompt: &[u32]) -> (u32, Duration)
{
    let start = Instant::now();
    e.reset();
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() { logits = e.step(tok, pos); }
    (argmax(&logits), start.elapsed())
}

fn main()
{
    let args = parse_args();
    let chip = sh("sysctl", &["-n", "machdep.cpu.brand_string"]);
    let slug = chip_slug(&chip);

    let mlx_path = PathBuf::from(RESULTS).join(format!("mlx_{slug}_{}.json", args.prompt));
    let mlx: Value = serde_json::from_str(&fs::read_to_string(&mlx_path).unwrap_or_else(|_| panic!(
        "{} not found: run `uv run bench.py --prompt {}` in mlx_baseline first", mlx_path.display(), args.prompt))).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(mlx["prompt"].clone()).unwrap();
    let peak = mlx["peak_bw_gbs"].as_f64().unwrap() * 1e9;
    let mlx_tokens: Vec<u32> = serde_json::from_value(mlx["out_of_the_box"]["tokens"].clone()).unwrap();

    println!("engine  : {}  on {chip}", args.engine);
    println!("prompt  : {} ({} tokens), {} decode steps, {} TTFT runs", args.prompt, prompt.len(), args.decode_steps, args.runs);

    let t = Instant::now();
    let max_seq = prompt.len() + args.decode_steps + 1;
    let mut engine: Box<dyn Engine> = match args.engine.as_str()
    {
        "cpu" =>
        {
            let model = Model::load_model(MODEL).expect("failed to load model");
            let cache = KVCache::new(&model.config);
            Box::new(Cpu { model, cache })
        }
        _ =>
        {
            let model = MetalModel::load_model(MODEL).expect("failed to load model");
            let cache = MetalKVCache::new(&model, max_seq);
            Box::new(Metal { model, cache, max_seq })
        }
    };
    let load = t.elapsed();

    // TTFT: first call pages in the mmap'd weights (and warms the GPU), then timed runs
    let (first, cold) = ttft(engine.as_mut(), &prompt);
    let mut ttft_ms = Vec::with_capacity(args.runs);
    for i in 0..args.runs
    {
        let (tok, dt) = ttft(engine.as_mut(), &prompt);
        assert_eq!(tok, first, "first token is not deterministic across runs");
        ttft_ms.push(ms(dt));
        println!("  ttft run {}/{}: {:.1} ms", i + 1, args.runs, ms(dt));
    }

    // decode: prefill once, then time every step on its own (like tests/mbu.rs)
    let weight_bytes = weight_bytes_per_token(engine.model());
    engine.reset();
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() { logits = engine.step(tok, pos); }
    let mut step_ms = Vec::with_capacity(args.decode_steps);
    let mut tokens = Vec::with_capacity(args.decode_steps);
    let mut total_bytes = 0usize;
    for pos in prompt.len()..prompt.len() + args.decode_steps
    {
        let next = argmax(&logits);
        tokens.push(next);
        let start = Instant::now();
        logits = engine.step(next, pos);
        step_ms.push(ms(start.elapsed()));
        total_bytes += weight_bytes + kv_bytes_at(&engine.model().config, pos);
    }
    let decode_s: f64 = step_ms.iter().sum::<f64>() / 1e3;
    let achieved = total_bytes as f64 / decode_s;

    let compared = tokens.len().min(mlx_tokens.len());
    let diverge = (0..compared).find(|&i| tokens[i] != mlx_tokens[i]);

    let status = sh("git", &["-C", env!("CARGO_MANIFEST_DIR"), "status", "--porcelain"]);
    let commit = sh("git", &["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--short", "HEAD"]);
    let result = json!({
        "engine": args.engine,
        "label": args.label,
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "commit": commit,
        "dirty": !status.is_empty(),
        "chip": chip,
        "prompt_name": args.prompt,
        "prompt_tokens": prompt.len(),
        "load_ms": ms(load),
        "first_token": first,
        "ttft": { "first_call_ms": ms(cold), "median_ms": median(&ttft_ms), "samples_ms": ttft_ms },
        "decode": {
            "steps": args.decode_steps,
            "median_ms": median(&step_ms),
            "tps": args.decode_steps as f64 / decode_s,
            "achieved_bw_gbs": achieved / 1e9,
            "mbu": achieved / peak,
            "weight_bytes_per_token": weight_bytes,
            "samples_ms": step_ms,
        },
        "parity_vs_mlx": { "compared": compared, "first_divergence": diverge },
        "tokens": tokens,
    });

    fs::create_dir_all(RESULTS).unwrap();
    let latest = PathBuf::from(RESULTS).join(format!("engine_{}_{slug}_{}.json", args.engine, args.prompt));
    fs::write(&latest, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    let history = PathBuf::from(RESULTS).join(format!("history_{slug}_{}.jsonl", args.prompt));
    let mut f = OpenOptions::new().create(true).append(true).open(&history).unwrap();
    writeln!(f, "{}", serde_json::to_string(&result).unwrap()).unwrap();

    println!("decode  : {:.1} tok/s  ({:.1} ms/token median), MBU {:.1}%",
        result["decode"]["tps"].as_f64().unwrap(), median(&step_ms), achieved / peak * 100.0);
    println!("ttft    : {:.1} ms median", median(&ttft_ms));
    match diverge
    {
        None => println!("parity  : greedy tokens match MLX for all {compared}"),
        Some(i) => println!("parity  : greedy tokens match MLX for the first {i} of {compared} (bf16 ties make some drift normal)"),
    }
    println!("wrote   : {}", latest.display());
}
