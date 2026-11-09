// GPU correctness: the Metal forward pass must match the CPU reference.
// fp32 accumulation order differs between the two, so compare with a tolerance
// and check that greedy decoding picks the same tokens.
//
// run with: cargo test --release --test metal_parity -- --nocapture

mod common;

use common::{argmax, load, PROMPT};
use metal_infrence_engine::cpu_inference::KVCache;
use metal_infrence_engine::metal_inference::{MetalKVCache, MetalModel};

const PARIS: u32 = 12095;
const MAX_ABS_DIFF: f32 = 1e-2;

#[test]
fn metal_matches_cpu()
{
    let cpu = load();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/qwen3-0.6b/model.safetensors");
    let gpu = MetalModel::load_model(path).unwrap();

    let mut cc = KVCache::new(&cpu.config);
    let mut gc = MetalKVCache::new(&gpu, 64);

    for (pos, &tok) in PROMPT.iter().enumerate()
    {
        let a = cpu.forward(tok, pos, &mut cc);
        let b = gpu.forward(tok, pos, &mut gc);
        assert_eq!(a.len(), b.len());

        let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("pos {pos}: max |cpu - gpu| = {diff:.2e}  argmax cpu {} gpu {}", argmax(&a), argmax(&b));
        assert!(b.iter().all(|v| v.is_finite()), "non-finite logits at pos {pos}");
        assert!(diff < MAX_ABS_DIFF, "pos {pos}: logits diverge by {diff}");
        assert_eq!(argmax(&a), argmax(&b), "pos {pos}: different greedy token");

        if pos == PROMPT.len() - 1 { assert_eq!(argmax(&b), PARIS); }
    }
}

// decode()'s pipelined steps and GPU argmax must pick what forward() + CPU argmax picks,
// with enough context that attention has more positions than simdgroups
#[test]
fn pipelined_decode_matches_forward()
{
    const STEPS: usize = 48;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/qwen3-0.6b/model.safetensors");
    let gpu = MetalModel::load_model(path).unwrap();
    let max_seq = PROMPT.len() + STEPS + 1;

    let mut cache = MetalKVCache::new(&gpu, max_seq);
    let mut logits = Vec::new();
    for (pos, &tok) in PROMPT.iter().enumerate() { logits = gpu.forward(tok, pos, &mut cache); }
    let first = argmax(&logits);

    let mut expected = Vec::with_capacity(STEPS);
    let mut tok = first;
    for i in 0..STEPS
    {
        tok = argmax(&gpu.forward(tok, PROMPT.len() + i, &mut cache));
        expected.push(tok);
    }

    let mut cache = MetalKVCache::new(&gpu, max_seq);
    for (pos, &tok) in PROMPT.iter().enumerate() { gpu.forward(tok, pos, &mut cache); }
    let mut got = Vec::with_capacity(STEPS);
    gpu.decode(first, PROMPT.len(), STEPS, &mut cache, &mut |t| got.push(t));

    assert_eq!(got, expected);
}
