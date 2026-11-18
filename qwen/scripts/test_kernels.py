"""Check every registered kernel implementation against the reference.

For each op, builds inputs at the real model's shapes (decode T=1 and a
prefill chunk), runs reference and each alternative, reports max abs / mean
relative error, and fails if beyond tolerance. Quantized-weight variants are
included for linear and moe_experts.

    uv run python -m scripts.test_kernels                 # all ops, real shapes
    uv run python -m scripts.test_kernels --tiny          # toy shapes (fast on CPU)
    uv run python -m scripts.test_kernels --op moe_experts --impl mine
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from qmoe import kernels as K
from qmoe.config import Config
from qmoe.shapes import QWEN15_MOE, op_cases

TOL = {"rmsnorm": 2e-2, "linear": 2e-2, "attention": 2e-2, "moe_experts": 3e-2}  # mean rel err (bf16)


def compare(op, impl, case):
    ref = np.asarray(case.bind(K.impl(op, "reference"))(*case.args), np.float32)
    out = np.asarray(jax.jit(case.bind(K.impl(op, impl)))(*case.args), np.float32)
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    err = np.abs(out - ref)
    rel = err.mean() / (np.abs(ref).mean() + 1e-9)
    return float(err.max()), float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="toy shapes instead of the real model's")
    ap.add_argument("--op", choices=K.OPS)
    ap.add_argument("--impl", help="only test this implementation name")
    args = ap.parse_args()

    cfg = Config.tiny() if args.tiny else QWEN15_MOE
    print(f"backend={jax.default_backend()} shapes={'tiny' if args.tiny else 'Qwen1.5-MoE-A2.7B'}")
    failed = []
    for op in [args.op] if args.op else K.OPS:
        impls = [i for i in K.implementations(op) if i != "reference" and (not args.impl or i == args.impl)]
        if not impls:
            print(f"{op:12s} (only reference registered)")
            continue
        for case in op_cases(op, cfg, key=jax.random.PRNGKey(0)):
            for impl in impls:
                try:
                    mx, rel = compare(op, impl, case)
                    ok = rel <= TOL[op]
                    print(f"{op:12s} {impl:12s} {case.name:28s} max|err| {mx:9.4f}  rel {rel:.2e}  {'ok' if ok else 'FAIL'}")
                except Exception as e:  # report and keep going
                    ok = False
                    print(f"{op:12s} {impl:12s} {case.name:28s} ERROR {type(e).__name__}: {str(e)[:200]}")
                if not ok:
                    failed.append((op, impl, case.name))
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("all registered kernels match reference")


if __name__ == "__main__":
    main()
