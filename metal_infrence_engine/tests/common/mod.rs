// shared helpers for model tests/benchmarks.
// files under tests/common/ are not compiled as their own test binaries.
#![allow(dead_code)]

use metal_infrence_engine::cpu_inference::Model;
use std::time::Duration;

// "The capital of France is" (ids taken from vocab.json)
pub const PROMPT: &[u32] = &[785, 6722, 315, 9625, 374];

pub fn load() -> Model
{
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/qwen3-0.6b/model.safetensors");
    Model::load_model(path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

pub fn argmax(x: &[f32]) -> u32
{
    let mut best = 0;
    for i in 1..x.len() { if x[i] > x[best] { best = i; } }
    best as u32
}

// peak DRAM bandwidth in GB/s (Apple spec). PEAK_BW_GBS overrides detection,
// which is needed for binned chips (e.g. M3 Max 300 vs 400 GB/s)
pub fn peak_bandwidth_gbs() -> f64
{
    if let Ok(v) = std::env::var("PEAK_BW_GBS") { return v.parse().expect("PEAK_BW_GBS must be a number"); }
    let out = std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().expect("sysctl failed");
    let chip = String::from_utf8_lossy(&out.stdout).trim().to_string();
    match chip.as_str()
    {
        "Apple M1" => 68.25,
        "Apple M1 Pro" => 200.0,
        "Apple M1 Max" => 400.0,
        "Apple M1 Ultra" => 800.0,
        "Apple M2" => 100.0,
        "Apple M2 Pro" => 200.0,
        "Apple M2 Max" => 400.0,
        "Apple M2 Ultra" => 800.0,
        "Apple M3" => 100.0,
        "Apple M3 Pro" => 150.0,
        "Apple M3 Max" => 400.0,
        "Apple M4" => 120.0,
        "Apple M4 Pro" => 273.0,
        "Apple M4 Max" => 546.0,
        _ => panic!("unknown chip {chip:?}, set PEAK_BW_GBS"),
    }
}

pub struct Stats { pub min: Duration, pub median: Duration, pub mean: Duration, pub max: Duration }

pub fn stats(samples: &[Duration]) -> Stats
{
    let mut s = samples.to_vec();
    s.sort();
    let mean = s.iter().sum::<Duration>() / s.len() as u32;
    Stats { min: s[0], median: s[s.len() / 2], mean, max: s[s.len() - 1] }
}
