"""Example Pallas kernel: RMSNorm. Copy this file as a template for your own.

Pallas mental model (TPU):
  * You write the body for ONE block. `pallas_call` runs it over a `grid`.
  * BlockSpec(block_shape, index_map) says which slice of each array a grid
    step sees: index_map(grid indices) -> block indices (not element offsets).
  * Inside the body, refs live in fast on-chip VMEM; `ref[...]` loads,
    `ref[...] = v` stores. HBM <-> VMEM copies are generated (and
    double-buffered) for you from the BlockSpecs.
  * TPU block shapes: the last two dims must be multiples of (8, 128), or
    equal the full array dims.

On a non-TPU backend (your Mac) the kernel runs in interpret mode: slow, but
the same code, so you can check correctness locally before going to Colab.

Register under a new name, then select it:
    QMOE_KERNELS=rmsnorm=pallas  (or --kernels rmsnorm=pallas)
and run scripts/test_kernels.py + scripts/bench.py --ops.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from qmoe.kernels import register


def _rmsnorm_kernel(x_ref, w_ref, o_ref, *, eps):
    x = x_ref[...].astype(jnp.float32)  # [bm, H] block
    y = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    o_ref[...] = (y * w_ref[...].astype(jnp.float32)).astype(o_ref.dtype)


@register("rmsnorm", "pallas")
def rmsnorm_pallas(x, w, eps):
    shape = x.shape
    H = shape[-1]
    x2 = x.reshape(-1, H)
    M = x2.shape[0]
    Mp = -(-M // 8) * 8  # pad rows to a multiple of 8 (TPU sublane count)
    if Mp != M:
        x2 = jnp.pad(x2, ((0, Mp - M), (0, 0)))
    bm = math.gcd(Mp, 256)  # rows per grid step

    out = pl.pallas_call(
        partial(_rmsnorm_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct((Mp, H), x.dtype),
        grid=(Mp // bm,),
        in_specs=[
            pl.BlockSpec((bm, H), lambda i: (i, 0)),  # block i of rows
            pl.BlockSpec((1, H), lambda i: (0, 0)),  # same weight every step
        ],
        out_specs=pl.BlockSpec((bm, H), lambda i: (i, 0)),
        interpret=jax.default_backend() != "tpu",
    )(x2, w.reshape(1, H))
    return out[:M].reshape(shape)
