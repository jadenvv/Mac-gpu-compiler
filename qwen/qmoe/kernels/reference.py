"""Reference implementations: straightforward jnp, written for clarity.

These define "correct". They are not tuned; where a reference is known to
waste memory traffic it says so, since that is where a kernel can win.
Matmuls take bf16 inputs and accumulate in f32 (preferred_element_type).
"""

import jax
import jax.numpy as jnp

from qmoe.kernels import register
from qmoe.quant import QArray, materialize

F32 = jnp.float32


@register("rmsnorm", "reference")
def rmsnorm(x, w, eps):
    xf = x.astype(F32)
    y = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * w.astype(F32)).astype(x.dtype)


@register("linear", "reference")
def linear(x, w, b=None):
    # Quantized: dequantizes the whole matrix to bf16 first, so it reads the
    # small int weights AND writes/reads a full bf16 copy -> slower than bf16.
    # A fused dequant-matmul kernel is what makes quantization pay off.
    y = jnp.matmul(x, materialize(w, x.dtype), preferred_element_type=F32)
    if b is not None:
        y = y + b.astype(F32)
    return y.astype(x.dtype)


@register("attention", "reference")
def attention(q, k_cache, v_cache, start):
    # Attends over the WHOLE cache (all S slots) and masks; at decode time
    # most of those slots are empty, so this reads S/len(context) times more
    # KV than needed. A kernel that stops at start + T wins that back.
    B, T, Hq, D = q.shape
    S, Hkv = k_cache.shape[1], k_cache.shape[2]
    if Hq != Hkv:  # grouped-query attention
        k_cache = jnp.repeat(k_cache, Hq // Hkv, axis=2)
        v_cache = jnp.repeat(v_cache, Hq // Hkv, axis=2)
    s = jnp.einsum("bthd,bshd->bhts", q, k_cache, preferred_element_type=F32) * D**-0.5
    q_pos = start + jnp.arange(T)
    mask = jnp.arange(S)[None, :] <= q_pos[:, None]  # [T, S]
    s = jnp.where(mask[None, None], s, -1e30)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum("bhts,bshd->bthd", p.astype(v_cache.dtype), v_cache, preferred_element_type=F32)
    return o.astype(q.dtype)


def _num_experts(w) -> int:
    return (w.q if isinstance(w, QArray) else w).shape[0]


@register("moe_experts", "reference")
def moe_experts(x, ids, weights, w_gate, w_up, w_down):
    """Sort token-expert pairs by expert, then one grouped matmul per weight.

    jax.lax.ragged_dot(lhs[M, K], rhs[E, K, N], group_sizes[E]) multiplies
    rows [0, g0) by rhs[0], the next g1 rows by rhs[1], and so on: the
    grouped GEMM every fast MoE engine is built around.
    """
    T, H = x.shape
    k = ids.shape[1]
    E = _num_experts(w_gate)

    flat = ids.reshape(-1)  # [T*k] expert of each (token, slot)
    order = jnp.argsort(flat)  # group rows by expert
    xs = x[order // k]  # [T*k, H] token rows, expert-sorted
    sizes = jnp.bincount(flat, length=E).astype(jnp.int32)

    # Quantized experts: dequantizes ALL E experts every call, even though
    # only the routed ones are used. At decode (T=1, k=4 of 60 experts) that
    # is ~15x the necessary weight traffic. Prime kernel target.
    wg, wu, wd = (materialize(w, x.dtype) for w in (w_gate, w_up, w_down))
    g = jax.lax.ragged_dot(xs, wg, sizes, preferred_element_type=F32)
    u = jax.lax.ragged_dot(xs, wu, sizes, preferred_element_type=F32)
    a = (jax.nn.silu(g) * u).astype(x.dtype)
    y = jax.lax.ragged_dot(a, wd, sizes, preferred_element_type=F32)  # [T*k, H]

    y = jnp.zeros_like(y).at[order].set(y)  # un-sort back to (token, slot)
    y = (y.reshape(T, k, H) * weights[..., None].astype(F32)).sum(axis=1)
    return y.astype(x.dtype)
