# Mac GPU compiler: beating MLX with hand-written Metal

A from-scratch inference engine for **Qwen3-0.6B (bf16)** on Apple silicon, written in Rust
with hand-written Metal compute kernels. The goal is to decode faster than MLX's
out-of-the-box `mlx_lm.stream_generate` on the same model, prompt and machine.

- Engine: [`metal_infrence_engine/`](metal_infrence_engine) (CPU reference in
  `src/cpu_inference.rs`, GPU engine in `src/metal_inference.rs`, kernels in
  `src/shaders/inference.metal`)
- Benchmark harness: [`metal_infrence_engine/mlx_baseline/`](metal_infrence_engine/mlx_baseline)
  (see its README for how to run it)

## Where it stands

Apple M1 (8 GB, 8-core GPU), mlx 0.32.4 / mlx-lm 0.32.2, 130-token `ai` prompt, 128 greedy
decode steps, plugged in, 5 interleaved rounds (Nov 2026):

| Engine | Decode speed (median) |
|---|---|
| CPU reference | 8.7 tok/s |
| Metal, first working version | 40.4 tok/s |
| Metal, current (`pipelined-decode` branch, f32 KV cache) | 46.37 tok/s |
| Metal, bf16 KV cache (`bf16-kv-cache` branch) | 46.62 tok/s |
| **MLX `stream_generate`** | **47.15 tok/s** |
| Memory-bandwidth limit (see below) | ~49.7 tok/s |

The engine is about **1.1% behind MLX**, roughly 0.24 ms per token. Run-to-run noise is
about the same size, so any result in this range needs several interleaved rounds, on
mains power, with other apps closed.

## The main limitation: memory bandwidth, not compute

Generating one token at batch size 1 means reading every weight once. For this model that
is **1,192 MB of bf16 weights per token**, plus the KV cache. Each weight is used for one
multiply-add, so the work is about 1 FLOP per byte read.

| | Value |
|---|---|
| Measured GPU read bandwidth (`tests/read_bandwidth.rs`) | 60.3 GB/s (88% of the 68.25 GB/s spec) |
| Bytes read per token (weights + average KV) | ~1,214 MB |
| Fastest possible token from memory alone | ~20.1 ms, about **49.7 tok/s** |
| Compute used at 46 tok/s | ~55 GFLOP/s, about **2%** of the M1 GPU's ~2.6 TFLOPS |

The GPU spends nearly all its time waiting for weights to arrive, so the chip's memory
speed sets the ceiling for every engine, MLX included. MLX already runs within about 5% of
that ceiling, which is why hand-written kernels can only beat it narrowly.

Evidence that compute is not the bottleneck:

- **`matvec` runs at the bandwidth limit.** Running only the matrix-vector kernels took
  19.4–20 ms per token for 1,192 MB, about 60 GB/s: the same as the measured ceiling.
- **Faster math changed nothing.** Switching `matvec` to 16-bit (`half`) arithmetic, which
  Apple's GPU executes faster than 32-bit, gave the same fastest-run time within 0.1 ms
  (21.53–21.65 ms) over 12 interleaved rounds, and increased error against the CPU engine.
- **Skipping work that isn't memory-bound saved little.** Removing `add_inplace` and
  `silu_mul` entirely saved under 0.1 ms each.

To go clearly faster than MLX, the engine has to read fewer bytes per token: get more than
one token out of each pass over the weights (speculation), or shrink the weights
(quantization).

## What changed, and what it bought

Breakdown of the first version's 24.7 ms token:

| Part | Time |
|---|---|
| `matvec` (weights) | ~19.4–20 ms |
| Attention | ~3.2 ms |
| CPU encoding, submission and GPU idle time | ~1.7 ms |
| Other kernels and dispatch overhead | ~0.7 ms |

Changes that took decode from 40.4 to ~46 tok/s (measured together, not individually):

1. **Pipelined command buffers.** Argmax runs on the GPU and feeds the next step's `embed`,
   so the CPU encodes step N+1 while the GPU runs step N and the GPU never idles.
2. **Attention rewrite.** One threadgroup per KV head computes both of its q heads in one
   read of K and V, and each simdgroup reads one position's adjacent values instead of 32
   lanes reading 32 scattered positions.
3. **RoPE table.** sin/cos are computed once on the CPU instead of in every head of every
   layer.
4. **Concurrent encoder.** Independent dispatches (q/k/v, gate/up) overlap, with explicit
   barriers between dependent ones.

## Tried, and didn't help (or not enough)

| Idea | Result |
|---|---|
| 16-bit arithmetic in `matvec` | No change in speed; logit error rose from ~5e-5 to 1–4e-2 |
| bf16 KV cache | +0.5% (46.62 vs 46.37 tok/s), same 128 tokens; logits move up to 0.074 from the CPU engine, so it fails the 1e-2 parity test and is unmerged |
| Copying weights to a shared or private GPU buffer | Under 1%, before pipelining; worth retesting now the gap is this small |
| `wired_limit`, untracked hazards, unretained references, smaller `MAX_SEQ` | No measurable change |
| Skipping `lm_head` rows with norm or k-means bounds (exact) | Still has to read ~100% of rows |
| `fast::` math | ~0.3 ms, the same trig cost the RoPE table removed |
| Prefetching the next layer's q/k/v weights during attention | Slower: +1.2, +2.1 and +3.4 ms for 2, 4 and 8 MB per layer, matching the cost of reading those bytes twice, so the cache didn't keep them |

Still untested: other `MATVEC_ROWS_PER_SG` values on the current engine, and staging `x`
in threadgroup memory (expected to do nothing, since `matvec` is already at the
bandwidth limit).

## Next steps

- **Attention.** It's the one part that still slows down as context grows (about 1 ms over
  128 tokens), and bf16 KV barely changed that. Packing the KV cache per head or a
  Flash-Decoding-style split are the leads.
- **Multi-token forward pass.** Reads the weights once for several tokens. Needed for fast
  prompt processing (token-by-token prefill is ~10x slower than MLX's batched prefill) and
  for speculation.
- **Speculation (prompt lookup).** Same greedy output with more than one token per pass;
  replaying MLX's recorded output gave 1.22–1.32 tokens per pass, which would lift the
  ~50 tok/s ceiling. Gains depend on how repetitive the text is.
- **Quantization.** Halves the bytes per token, but it's a different model, so it has to be
  compared against quantized MLX.

## Benchmarking notes

- The machine is a fanless MacBook Air: battery power, Low Power Mode, Chrome and swap all
  shifted results by several percent. Compare engines in interleaved rounds.
- `compare.py`'s "time per token" and "MBU" rows compare the pipelined engine against MLX's
  slower synchronous mode, so they flatter the engine. Decode speed is the fair comparison.
- Greedy output matches MLX for the first 19 tokens, then drifts because bf16 logits tie.
  The CPU engine is the correctness reference (`tests/metal_parity.rs`).
