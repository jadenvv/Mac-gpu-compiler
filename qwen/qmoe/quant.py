"""Weight-only quantization: int8 and packed int4, symmetric, group-wise.

A weight W of shape [..., K, N] (K = input/contraction dim, N = output dim) is
split into groups of `group` consecutive rows along K. Each (group, column)
pair gets one scale = max|W| / qmax, stored in bf16:

    W[..., k, n] ~= q[..., k, n] * scale[..., k // group, n]

int8: q stored as int8 in [-127, 127]                      -> q.shape = [..., K, N]
int4: q in [-8, 7], stored +8 as a nibble, two rows packed
      per byte (even k = low nibble, odd k = high nibble)  -> q.shape = [..., K/2, N]

group = None means one group spanning all of K (per-output-channel).

These are the *reference* (de)quantizers. A fast kernel should never
materialize the dequantized matrix: it should read q + scale and dequantize
in registers/VMEM right before the matmul.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.tree_util.register_dataclass, data_fields=["q", "scale"], meta_fields=["bits"])
@dataclass
class QArray:
    q: jax.Array
    scale: jax.Array
    bits: int

    @property
    def K(self) -> int:
        return self.q.shape[-2] * (2 if self.bits == 4 else 1)

    @property
    def N(self) -> int:
        return self.q.shape[-1]

    @property
    def group(self) -> int:
        return self.K // self.scale.shape[-2]

    @property
    def shape(self) -> tuple:
        return (*self.q.shape[:-2], self.K, self.N)

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def quantize(w: jax.Array, bits: int, group: int | None = 128) -> QArray:
    assert bits in (4, 8), bits
    *lead, K, N = w.shape
    g = K if not group else group
    assert K % g == 0, f"K={K} not divisible by group={g}"
    qmax = 2 ** (bits - 1) - 1  # 127 or 7

    wg = w.astype(jnp.float32).reshape(*lead, K // g, g, N)
    amax = jnp.max(jnp.abs(wg), axis=-2, keepdims=True)
    # Round the scale to its storage dtype *before* computing q so that
    # dequantize(quantize(w)) uses exactly the scale q was computed against.
    scale = (amax / qmax).astype(jnp.bfloat16).astype(jnp.float32)
    scale = jnp.where(scale == 0, 1.0, scale)
    q = jnp.clip(jnp.round(wg / scale), -qmax - 1, qmax).astype(jnp.int8)
    q = q.reshape(*lead, K, N)
    scale = scale.squeeze(-2).astype(jnp.bfloat16)  # [..., K/g, N]

    if bits == 4:
        assert K % 2 == 0
        u = (q + 8).astype(jnp.uint8)  # [-8, 7] -> [0, 15]
        q = u[..., 0::2, :] | (u[..., 1::2, :] << 4)
    return QArray(q=q, scale=scale, bits=bits)


def unpack_int4(packed: jax.Array) -> jax.Array:
    """[..., K/2, N] uint8 -> [..., K, N] int8 in [-8, 7]."""
    lo = (packed & 0xF).astype(jnp.int8) - 8
    hi = (packed >> 4).astype(jnp.int8) - 8
    *lead, half, N = packed.shape
    return jnp.stack([lo, hi], axis=-2).reshape(*lead, 2 * half, N)


def dequantize(qa: QArray, dtype=jnp.bfloat16) -> jax.Array:
    q = unpack_int4(qa.q) if qa.bits == 4 else qa.q
    *lead, K, N = q.shape
    G = qa.scale.shape[-2]
    w = q.reshape(*lead, G, K // G, N).astype(jnp.float32)
    w = w * qa.scale[..., :, None, :].astype(jnp.float32)
    return w.reshape(*lead, K, N).astype(dtype)


def materialize(w, dtype=jnp.bfloat16) -> jax.Array:
    """Dense view of a weight that may or may not be quantized."""
    return dequantize(w, dtype) if isinstance(w, QArray) else w
