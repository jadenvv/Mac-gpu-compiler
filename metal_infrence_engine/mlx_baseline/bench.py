"""MLX baseline: the numbers the hand-written Metal engine has to beat.

Runs the same unquantized bf16 Qwen3-0.6B (../qwen3-0.6b), the same prompt token ids
and greedy decoding as the Rust side, then reports:

  1. out of the box   mlx_lm.stream_generate, exactly as a user would call it
                      (async pipelined decode). This is the headline target.
  2. instrumented     the same measurements as tests/ttft.rs and tests/mbu.rs:
                      TTFT (batched prefill, and token-by-token prefill like our engine),
                      synchronous per-token decode latency, and MBU with the same
                      bytes-per-token formula.

Results are written to results/mlx_<chip>_<prompt>.json.

run with (from metal_infrence_engine/mlx_baseline):
    uv run bench.py                    # long "AI is going to take over..." prompt
    uv run bench.py --prompt paris     # the 5-token prompt the Rust tests use
    uv run bench.py --prompt "any text you like"
"""

import argparse
import json
import platform
import statistics
import struct
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "qwen3-0.6b"

# "The capital of France is" (ids from vocab.json), same as tests/common/mod.rs
PARIS_PROMPT = [785, 6722, 315, 9625, 374]
PARIS = 12095
# longer default prompt so prefill and attention over a real context get exercised
AI_PROMPT = (
    "AI is going to take over the world, or at least that is what the headlines keep telling us. "
    "Every week there is a new model that writes code, passes exams, draws pictures, and talks like a person. "
    "Some people think this means most office jobs will disappear within a decade, while others argue that "
    "the technology is overhyped and will settle into being a useful tool, like the spreadsheet or the search engine. "
    "To understand which of these futures is more likely, we first need to look at what these systems can actually do today, "
    "where they still fail, and what it costs to run them. The most important thing to understand is that"
)
PROMPTS = {"paris": PARIS_PROMPT, "ai": AI_PROMPT}
# greedy output of the Rust CPU engine (src/main.rs) for PROMPT, 16 tokens
CPU_REFERENCE = [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]

# same table as peak_bandwidth_gbs() in tests/common/mod.rs
PEAK_BW_GBS = {
    "Apple M1": 68.25, "Apple M1 Pro": 200.0, "Apple M1 Max": 400.0, "Apple M1 Ultra": 800.0,
    "Apple M2": 100.0, "Apple M2 Pro": 200.0, "Apple M2 Max": 400.0, "Apple M2 Ultra": 800.0,
    "Apple M3": 100.0, "Apple M3 Pro": 150.0, "Apple M3 Max": 400.0,
    "Apple M4": 120.0, "Apple M4 Pro": 273.0, "Apple M4 Max": 546.0,
}


def chip_name() -> str:
    return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()


def peak_bandwidth_gbs(chip: str, override: float | None) -> float:
    if override is not None:
        return override
    if chip not in PEAK_BW_GBS:
        raise SystemExit(f"unknown chip {chip!r}, pass --peak-bw-gbs")
    return PEAK_BW_GBS[chip]


def weight_bytes_per_token() -> tuple[int, dict]:
    """Same rule as weight_bytes_per_token() in tests/mbu.rs: every tensor except the
    embedding table, plus the one embedding row for the current token. (MLX ties lm_head
    to embed_tokens, so it reads the full table as lm_head: the byte count is identical.)"""
    with open(MODEL_DIR / "model.safetensors", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    config = json.loads((MODEL_DIR / "config.json").read_text())
    total = sum(t["data_offsets"][1] - t["data_offsets"][0] for k, t in header.items() if k != "model.embed_tokens.weight")
    return total + config["hidden_size"] * 2, config


def kv_bytes_at(config: dict, pos: int, dtype_size: int) -> int:
    """K and V for every cached position (0..=pos) in every layer. MLX keeps the cache
    in the model dtype (bf16, 2 bytes); our engine uses f32 (4 bytes)."""
    return 2 * config["num_hidden_layers"] * (pos + 1) * config["num_key_value_heads"] * config["head_dim"] * dtype_size


def ms(seconds: float) -> float:
    return seconds * 1e3


def summarize(samples_s: list[float]) -> dict:
    s = sorted(samples_s)
    return {"min_ms": ms(s[0]), "median_ms": ms(statistics.median(s)), "mean_ms": ms(statistics.fmean(s)), "max_ms": ms(s[-1])}


# ---------------------------------------------------------------------------
# 1. out of the box
# ---------------------------------------------------------------------------

def out_of_the_box(model, tokenizer, prompt: list[int], max_tokens: int, runs: int) -> dict:
    def once():
        last, tokens = None, []
        for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            tokens.append(r.token)
            last = r
        return last, tokens

    once()  # warm-up: pages in weights, compiles kernels
    results = [once() for _ in range(runs)]
    last = results[-1][0]
    return {
        "max_tokens": max_tokens,
        "runs": runs,
        "prompt_tps_median": statistics.median(r.prompt_tps for r, _ in results),
        "generation_tps_median": statistics.median(r.generation_tps for r, _ in results),
        "generation_tps_all": [r.generation_tps for r, _ in results],
        "peak_memory_gb": last.peak_memory,
        "tokens": results[-1][1],
    }


# ---------------------------------------------------------------------------
# 2. instrumented, mirrors tests/ttft.rs and tests/mbu.rs
# ---------------------------------------------------------------------------

def ttft_batched(model, prompt: list[int]) -> tuple[int, float]:
    """MLX's normal prefill: the whole prompt in one forward pass."""
    start = time.perf_counter()
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    first = mx.argmax(logits[0, -1])
    mx.eval(first)
    return first.item(), time.perf_counter() - start


def ttft_sequential(model, prompt: list[int]) -> tuple[int, float]:
    """Token-by-token prefill, the way our engine's forward(token, pos) does it."""
    start = time.perf_counter()
    cache = make_prompt_cache(model)
    for tok in prompt:
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
    first = mx.argmax(logits[0, -1])
    mx.eval(first)
    return first.item(), time.perf_counter() - start


def ttft(model, prompt: list[int], runs: int) -> dict:
    out = {}
    for name, fn in [("batched", ttft_batched), ("sequential", ttft_sequential)]:
        first, cold = fn(model, prompt)
        if prompt == PARIS_PROMPT:
            assert first == PARIS, f'{name}: expected " Paris" ({PARIS}), got {first}'
        samples = []
        for _ in range(runs):
            tok, dt = fn(model, prompt)
            assert tok == first, f"{name}: first token is not deterministic"
            samples.append(dt)
        stats = summarize(samples)
        stats["first_call_ms"] = ms(cold)
        stats["prefill_tps_median"] = len(prompt) / (stats["median_ms"] / 1e3)
        out[name] = stats
    return out


def decode_mbu(model, prompt: list[int], decode_steps: int, peak_bw: float) -> dict:
    """Synchronous decode: each step is forward + argmax + wait, timed individually,
    like MetalModel::forward's commit + wait_until_completed."""
    weight_bytes, config = weight_bytes_per_token()
    kv_dtype_size = model.model.embed_tokens.weight.dtype.size

    # warm-up
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([[prompt[0]]]), cache=cache))

    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    nxt = mx.argmax(logits[0, -1])
    mx.eval(nxt)

    times, total_bytes, mbu_samples, tokens = [], 0, [], []
    for pos in range(len(prompt), len(prompt) + decode_steps):
        tokens.append(nxt.item())
        start = time.perf_counter()
        logits = model(nxt.reshape(1, 1), cache=cache)
        nxt = mx.argmax(logits[0, -1])
        mx.eval(nxt)
        dt = time.perf_counter() - start

        nbytes = weight_bytes + kv_bytes_at(config, pos, kv_dtype_size)
        total_bytes += nbytes
        times.append(dt)
        mbu_samples.append(nbytes / dt / (peak_bw * 1e9))

    achieved = total_bytes / sum(times)
    mbu_samples.sort()
    t = summarize(times)
    return {
        "decode_steps": decode_steps,
        "weight_bytes_per_token": weight_bytes,
        "kv_dtype_bytes": kv_dtype_size,
        "kv_bytes_last_step": kv_bytes_at(config, len(prompt) + decode_steps - 1, kv_dtype_size),
        "time_per_token": t,
        "decode_tps_median": 1e3 / t["median_ms"],
        "achieved_bw_gbs": achieved / 1e9,
        "mbu": achieved / (peak_bw * 1e9),
        "mbu_per_step": {"min": mbu_samples[0], "median": statistics.median(mbu_samples), "max": mbu_samples[-1]},
        "tokens": tokens,
    }


def compare_to_cpu(tokens: list[int]) -> dict:
    n = min(len(tokens), len(CPU_REFERENCE))
    diverge = next((i for i in range(n) if tokens[i] != CPU_REFERENCE[i]), None)
    return {"matches_cpu_reference": diverge is None, "first_divergence": diverge, "compared": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="ai",
                    help=f"preset ({', '.join(PROMPTS)}) or literal text; 'paris' is the Rust tests' prompt (default ai)")
    ap.add_argument("--max-tokens", type=int, default=128, help="tokens to generate in the out-of-the-box run")
    ap.add_argument("--decode-steps", type=int, default=32, help="decode steps for MBU (tests/mbu.rs uses 32)")
    ap.add_argument("--runs", type=int, default=5, help="timed repetitions (tests/ttft.rs uses 5)")
    ap.add_argument("--peak-bw-gbs", type=float, default=None, help="override peak DRAM bandwidth")
    ap.add_argument("--out", type=Path, default=None, help="results JSON path (default results/mlx_<chip>.json)")
    args = ap.parse_args()

    if platform.machine() != "arm64":
        raise SystemExit(f"running as {platform.machine()} (Rosetta?): MLX needs a native arm64 Python")

    chip = chip_name()
    peak_bw = peak_bandwidth_gbs(chip, args.peak_bw_gbs)

    t = time.perf_counter()
    model, tokenizer = load(str(MODEL_DIR))
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t
    dtype = str(model.model.embed_tokens.weight.dtype)

    prompt_name = args.prompt if args.prompt in PROMPTS else "custom"
    p = PROMPTS.get(args.prompt, args.prompt)
    prompt = p if isinstance(p, list) else tokenizer.encode(p, add_special_tokens=False)

    print(f"chip           : {chip}  (peak {peak_bw:.2f} GB/s)")
    print(f"mlx / mlx-lm   : {mx.__version__} / {__import__('mlx_lm').__version__}   device {mx.default_device()}")
    print(f"model          : {MODEL_DIR.name}  dtype {dtype}  load {ms(load_s):.0f} ms")

    print(f"prompt         : {prompt_name}, {len(prompt)} tokens: {tokenizer.decode(prompt)[:70]!r}...")
    print(f"prompt ids     : {prompt}")

    oob = out_of_the_box(model, tokenizer, prompt, args.max_tokens, args.runs)
    print("\n-- out of the box (mlx_lm.stream_generate) --")
    print(f"prompt         : {oob['prompt_tps_median']:.1f} tok/s  ({len(prompt)} tokens)")
    print(f"generation     : {oob['generation_tps_median']:.1f} tok/s  (median of {args.runs}, {args.max_tokens} tokens)  <-- target")
    print(f"peak memory    : {oob['peak_memory_gb']:.2f} GB")

    tt = ttft(model, prompt, args.runs)
    print("\n-- TTFT (tests/ttft.rs) --")
    for name, s in tt.items():
        print(f"{name:<15}: median {s['median_ms']:.2f} ms | min {s['min_ms']:.2f} | max {s['max_ms']:.2f}  "
              f"({s['prefill_tps_median']:.1f} tok/s prefill)")

    dec = decode_mbu(model, prompt, args.decode_steps, peak_bw)
    print("\n-- decode + MBU (tests/mbu.rs, synchronous per step) --")
    print(f"weights/token  : {dec['weight_bytes_per_token'] / 1e6:.1f} MB  (+ bf16 KV cache, {dec['kv_bytes_last_step'] / 1e6:.2f} MB at last step)")
    print(f"time/token     : median {dec['time_per_token']['median_ms']:.2f} ms  ({dec['decode_tps_median']:.1f} tok/s)")
    print(f"achieved bw    : {dec['achieved_bw_gbs']:.2f} GB/s")
    print(f"MBU            : {dec['mbu'] * 100:.1f}%  (per-step min {dec['mbu_per_step']['min'] * 100:.1f}% | "
          f"median {dec['mbu_per_step']['median'] * 100:.1f}% | max {dec['mbu_per_step']['max'] * 100:.1f}%)")

    print(f"\n-- output --\n{tokenizer.decode(oob['tokens'])!r}")
    parity = None
    if prompt == PARIS_PROMPT:
        parity = compare_to_cpu(oob["tokens"])
        print("\n-- parity vs Rust CPU engine --")
        if parity["matches_cpu_reference"]:
            print(f"greedy tokens  : first {parity['compared']} match")
        else:
            print(f"greedy tokens  : diverge at index {parity['first_divergence']} (bf16 vs f32 math, expected eventually)")

    out = args.out or HERE / "results" / f"mlx_{chip.replace('Apple ', '').replace(' ', '_').lower()}_{prompt_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "chip": chip, "peak_bw_gbs": peak_bw, "mlx": mx.__version__, "mlx_lm": __import__("mlx_lm").__version__,
        "dtype": dtype, "prompt_name": prompt_name, "prompt": prompt, "load_ms": ms(load_s),
        "out_of_the_box": oob, "ttft": tt, "decode": dec, "parity": parity,
    }, indent=2))
    print(f"\nwrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
