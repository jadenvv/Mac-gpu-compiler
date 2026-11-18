"""Profile kernels and the full model.

  # Per-op microbenchmarks at real shapes, with roofline numbers (no weights needed)
  uv run python -m scripts.bench --ops
  uv run python -m scripts.bench --ops --op moe_experts --kernels moe_experts=mine

  # Whole model: load real weights, time prefill + decode, roofline per token
  uv run python -m scripts.bench --model --quant int4
  uv run python -m scripts.bench --model --trace /content/drive/MyDrive/traces/run1

--trace writes a JAX profiler trace of a few decode steps. Open it at
https://ui.perfetto.dev (the *.trace.json.gz inside) or with xprof/TensorBoard.
Every op in the model has a jax.named_scope (attn, router, routed_experts,
shared_expert), so the trace groups time by model component.

Roofline: "% BW" = ideal bytes / time / peak HBM bandwidth. At batch 1 decode
is bandwidth-bound, so that is the number a kernel should push toward 100%.
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from qmoe import kernels as K

# Per-chip peaks: (HBM bytes/s, bf16 FLOP/s). Public spec-sheet numbers.
PEAKS = {
    "TPU v6 lite": (1640e9, 918e12),  # v6e (Trillium)
    "TPU v5 lite": (819e9, 197e12),  # v5e
    "TPU v5": (2765e9, 459e12),  # v5p
    "TPU v4": (1228e9, 275e12),
}


def peaks():
    kind = jax.devices()[0].device_kind
    return kind, PEAKS.get(kind, (float("nan"), float("nan")))


def time_fn(fn, *args, warmup=3, iters=20) -> float:
    """Median wall time of a jitted call, in seconds (compile excluded)."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def bench_ops(args):
    from qmoe.config import Config
    from qmoe.shapes import QWEN15_MOE, op_cases

    cfg = Config.tiny() if args.tiny else QWEN15_MOE
    kind, (bw, fl) = peaks()
    print(f"device: {kind}  peak {bw / 1e9:.0f} GB/s, {fl / 1e12:.0f} TFLOP/s")
    print(f"{'op':12s} {'impl':11s} {'case':34s} {'time':>10s} {'GB/s':>8s} {'%BW':>6s} {'TFLOP/s':>8s} {'%FLOP':>6s}")
    rows = []
    for op in [args.op] if args.op else K.OPS:
        impls = [args.impl] if args.impl else K.implementations(op)
        for case in op_cases(op, cfg):
            for impl in impls:
                fn = jax.jit(case.bind(K.impl(op, impl)))
                try:
                    t = time_fn(fn, *case.args, iters=args.iters)
                except Exception as e:
                    print(f"{op:12s} {impl:11s} {case.name:34s} ERROR {type(e).__name__}: {str(e)[:120]}")
                    continue
                gbs, tfs = case.bytes / t / 1e9, case.flops / t / 1e12
                rows.append(dict(op=op, impl=impl, case=case.name, us=t * 1e6, gbs=gbs, tflops=tfs))
                print(f"{op:12s} {impl:11s} {case.name:34s} {t * 1e6:8.1f}us {gbs:8.1f} {100 * gbs * 1e9 / bw:5.1f}% "
                      f"{tfs:8.2f} {100 * tfs * 1e12 / fl:5.1f}%")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": kind, "rows": rows}, f, indent=1)


def active_bytes_per_token(cfg, params) -> int:
    """Bytes a perfect decode step must read: everything except the unused
    experts (only top_k of n_experts per layer) and the embedding table (one row)."""
    from qmoe.quant import QArray

    total = 0
    for name, leaf in params["layers"].items():
        nb = leaf.nbytes if isinstance(leaf, QArray) else leaf.size * leaf.dtype.itemsize
        if name in ("w_gate", "w_up", "w_down"):
            nb = nb * cfg.top_k // cfg.n_experts
        total += nb
    return total + params["lm_head"].nbytes + params["final_norm"].nbytes


def bench_model(args):
    from qmoe.engine import Engine
    from qmoe.weights import QuantSpec, load, param_bytes

    spec = QuantSpec.parse(args.quant, args.group, args.targets)
    t = time.perf_counter()
    cfg, params, _ = load(args.model_path, spec)
    print(f"loaded {spec.name()} in {time.perf_counter() - t:.0f}s: {param_bytes(params) / 1e9:.2f} GB on device")

    eng = Engine(cfg, params, max_seq=args.max_seq, chunk=args.chunk)
    rng = np.random.RandomState(0)
    prompt = list(rng.randint(100, cfg.vocab_size - 100, size=args.prompt_len))

    t = time.perf_counter()
    eng.generate(prompt[: args.chunk], 3)  # compile prefill + decode programs
    print(f"compile: {time.perf_counter() - t:.0f}s  kernels: {dict(K.signature())}")

    runs = [eng.generate(prompt, args.gen_len) for _ in range(args.repeats)]
    r = min(runs, key=lambda r: r.decode_s)
    kind, (bw, _) = peaks()
    ideal = active_bytes_per_token(cfg, params)
    ms_tok = 1e3 / r.decode_tok_s
    print(f"device: {kind}")
    print(f"prefill {r.prompt_len} tokens: {r.prefill_s * 1e3:.1f} ms ({r.prompt_len / r.prefill_s:.0f} tok/s)")
    print(f"decode: {r.decode_tok_s:.1f} tok/s ({ms_tok:.2f} ms/token over {len(r.ids) - 1} tokens)")
    print(f"ideal bytes/token: {ideal / 1e9:.2f} GB -> roofline {bw / ideal:.0f} tok/s; "
          f"achieved {100 * (ideal / (ms_tok / 1e3)) / bw:.1f}% of peak BW")

    if args.trace:
        tok = prompt[: args.chunk]
        eng.generate(tok, 4)
        with jax.profiler.trace(args.trace):
            eng.generate(tok, args.trace_steps)
        print(f"trace written to {args.trace} (open the .trace.json.gz in https://ui.perfetto.dev)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=kind, quant=spec.name(), kernels=dict(K.signature()),
                           prefill_tok_s=r.prompt_len / r.prefill_s, decode_tok_s=r.decode_tok_s,
                           ideal_gb_per_token=ideal / 1e9, gb_on_device=param_bytes(params) / 1e9), f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ops", action="store_true", help="per-op microbenchmarks")
    mode.add_argument("--model", action="store_true", help="whole-model prefill/decode")
    ap.add_argument("--kernels", help='e.g. "moe_experts=mine,rmsnorm=pallas" (default: reference)')
    ap.add_argument("--json", help="also write results here")
    # --ops
    ap.add_argument("--op", choices=K.OPS)
    ap.add_argument("--impl")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    # --model
    ap.add_argument("--model-path", default="Qwen/Qwen1.5-MoE-A2.7B")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--group", type=int, default=128, help="quant group size along K (0 = per-channel)")
    ap.add_argument("--targets", default="experts,shared", help="which weights to quantize: experts,shared,attn")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--max-seq", type=int, default=2048, help="KV cache length (reference attention reads all of it)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trace", help="write a profiler trace of decode steps to this dir")
    ap.add_argument("--trace-steps", type=int, default=16)
    args = ap.parse_args()

    K.use_spec(args.kernels)
    bench_ops(args) if args.ops else bench_model(args)


if __name__ == "__main__":
    main()
