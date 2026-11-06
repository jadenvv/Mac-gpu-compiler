// Model Bandwidth Utilization: how close decode gets to the memory bandwidth limit.
//
//   MBU = (bytes read per decode token / time per decode token) / peak bandwidth
//
// one decode step (batch 1) must read every weight once plus the whole KV cache, so
// bandwidth is the ceiling on decode speed. MBU of 100% means decode is as fast as the
// memory system allows.
//
// run with: cargo test --release --test mbu -- --nocapture

mod common;

use common::{argmax, load, peak_bandwidth_gbs, stats, PROMPT};
use metal_infrence_engine::cpu_inference::{KVCache, Model};
use std::time::{Duration, Instant};

const DECODE_STEPS: usize = 32;

// weights touched by one forward pass: everything except the embedding table,
// of which only the one row for the current token is read
fn weight_bytes_per_token(model: &Model) -> usize
{
    let mut bytes = 0;
    for t in &model.tensors
    {
        if t.name == "model.embed_tokens.weight" { continue; }
        bytes += (t.data_offsets[1] - t.data_offsets[0]) as usize;
    }
    bytes + model.config.hidden_size * 2 // one bf16 embedding row
}

// f32 K and V for every cached position (0..=pos) in every layer
fn kv_bytes_at(model: &Model, pos: usize) -> usize
{
    let c = &model.config;
    2 * c.num_hidden_layers * (pos + 1) * c.num_key_value_heads * c.head_dim * 4
}

#[test]
fn mbu()
{
    let model = load();
    let peak = peak_bandwidth_gbs() * 1e9;
    let weight_bytes = weight_bytes_per_token(&model);

    // warm-up so the mmap'd weights are in the page cache before timing
    let mut cache = KVCache::new(&model.config);
    model.forward(PROMPT[0], 0, &mut cache);

    let mut cache = KVCache::new(&model.config);
    let mut logits = Vec::new();
    for (pos, &tok) in PROMPT.iter().enumerate() { logits = model.forward(tok, pos, &mut cache); }

    // time each decode step; ignore EOS so every run measures the same number of steps
    let mut times = Vec::with_capacity(DECODE_STEPS);
    let mut total_bytes = 0usize;
    let mut mbu_samples = Vec::with_capacity(DECODE_STEPS);
    for pos in PROMPT.len()..PROMPT.len() + DECODE_STEPS
    {
        let next = argmax(&logits);
        let start = Instant::now();
        logits = model.forward(next, pos, &mut cache);
        let dt = start.elapsed();

        let bytes = weight_bytes + kv_bytes_at(&model, pos);
        total_bytes += bytes;
        times.push(dt);
        mbu_samples.push(bytes as f64 / dt.as_secs_f64() / peak);
    }

    let total_time: Duration = times.iter().sum();
    let achieved = total_bytes as f64 / total_time.as_secs_f64();
    let mbu = achieved / peak;
    mbu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = stats(&times);

    println!();
    println!("peak bandwidth : {:.2} GB/s", peak / 1e9);
    println!("weights/token  : {:.1} MB  (+ KV cache, {:.2} MB at last step)",
        weight_bytes as f64 / 1e6, kv_bytes_at(&model, PROMPT.len() + DECODE_STEPS - 1) as f64 / 1e6);
    println!("time/token     : median {:.2?}  ({DECODE_STEPS} decode steps)", t.median);
    println!("achieved bw    : {:.2} GB/s", achieved / 1e9);
    println!("MBU            : {:.1}%  (per-step min {:.1}% | median {:.1}% | max {:.1}%)",
        mbu * 100.0,
        mbu_samples[0] * 100.0,
        mbu_samples[mbu_samples.len() / 2] * 100.0,
        mbu_samples[mbu_samples.len() - 1] * 100.0);

    assert!(mbu > 0.0 && mbu <= 1.0, "MBU {mbu} out of range: byte count or peak bandwidth is wrong");
}
