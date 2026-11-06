// Time to first token: time from handing the prompt to the model until the
// first generated token is known (prefill over the whole prompt + sampling).
// Model load is measured separately since it's a one-time cost.
//
// run with: cargo test --release --test ttft -- --nocapture

mod common;

use common::{argmax, load, stats, PROMPT};
use metal_infrence_engine::cpu_inference::{KVCache, Model};
use std::time::{Duration, Instant};

const RUNS: usize = 5;
// "ĠParis" in vocab.json: the correct greedy answer for PROMPT
const PARIS: u32 = 12095;

// one full request: fresh cache, prefill the prompt, pick the first token
fn time_to_first_token(model: &Model, prompt: &[u32]) -> (u32, Duration)
{
    let start = Instant::now();
    let mut cache = KVCache::new(&model.config);
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate()
    {
        logits = model.forward(tok, pos, &mut cache);
    }
    let first = argmax(&logits);
    (first, start.elapsed())
}

#[test]
fn ttft()
{
    let t = Instant::now();
    let model = load();
    let load_time = t.elapsed();

    // cold: first touch of the mmap'd weights pages them in from disk
    let (first, cold) = time_to_first_token(&model, PROMPT);
    assert_eq!(first, PARIS, "expected \" Paris\" as the first token, got {first}");

    // warm: weights are now in the page cache
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS
    {
        let (tok, dt) = time_to_first_token(&model, PROMPT);
        assert_eq!(tok, first, "first token is not deterministic across runs");
        samples.push(dt);
    }
    let s = stats(&samples);

    println!();
    println!("prompt tokens : {}", PROMPT.len());
    println!("first token   : {first}");
    println!("model load    : {load_time:.2?}");
    println!("TTFT cold     : {cold:.2?}");
    println!("TTFT warm     : min {:.2?} | median {:.2?} | mean {:.2?} | max {:.2?}  ({RUNS} runs)",
        s.min, s.median, s.mean, s.max);
    println!("prefill speed : {:.2} tok/s (median)", PROMPT.len() as f64 / s.median.as_secs_f64());
}
