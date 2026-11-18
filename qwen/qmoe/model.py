"""Qwen2-MoE forward pass in plain JAX. Every hot op goes through qmoe.kernels.

Parameter tree (layers stacked on a leading L axis so the layer loop is a
lax.scan: one compiled layer body instead of 24 unrolled copies):

  embed [V, H]   final_norm [H]   lm_head [H, V]
  layers:
    attn_norm, mlp_norm          [L, H]
    wq, wk, wv [L, H, Hq*D]   bq, bk, bv [L, Hq*D]   wo [L, Hq*D, H]
    router [L, H, E]
    w_gate, w_up [L, E, H, I]   w_down [L, E, I, H]           (routed experts)
    ws_gate, ws_up [L, H, Is]   ws_down [L, Is, H]   ws_router [L, H, 1]  (shared expert)

Linear weights are stored [in, out] (HF stores [out, in]; the loader transposes).
Any matmul weight may be a quant.QArray instead of a bf16 array.

KV cache: {"k": [L, B, S, Hkv, D], "v": same}, bf16, S = max sequence length.
`step` writes the T new tokens' K/V at positions [start, start + T).
"""

from functools import partial

import jax
import jax.numpy as jnp

from qmoe import kernels as K
from qmoe.config import Config

F32 = jnp.float32


def rope(x, pos, theta):
    """Rotary embedding, rotate-half (GPT-NeoX) layout as in HF Qwen2.
    x [B, T, h, D], pos [T] absolute positions."""
    D = x.shape[-1]
    inv_freq = 1.0 / (theta ** (jnp.arange(0, D, 2, dtype=F32) / D))
    ang = pos.astype(F32)[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = jnp.concatenate([jnp.cos(ang)] * 2, axis=-1)[None, :, None, :]
    sin = jnp.concatenate([jnp.sin(ang)] * 2, axis=-1)[None, :, None, :]
    xf = x.astype(F32)
    x1, x2 = xf[..., : D // 2], xf[..., D // 2 :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return (xf * cos + rotated * sin).astype(x.dtype)


def moe_block(cfg: Config, h, lp):
    B, T, H = h.shape
    x = h.reshape(B * T, H)

    with jax.named_scope("router"):
        logits = K.linear(x, lp["router"]).astype(F32)
        probs = jax.nn.softmax(logits, axis=-1)
        weights, ids = jax.lax.top_k(probs, cfg.top_k)  # [T, k]
        if cfg.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)

    with jax.named_scope("routed_experts"):
        routed = K.moe_experts(x, ids.astype(jnp.int32), weights, lp["w_gate"], lp["w_up"], lp["w_down"])

    with jax.named_scope("shared_expert"):
        a = jax.nn.silu(K.linear(x, lp["ws_gate"]).astype(F32)) * K.linear(x, lp["ws_up"]).astype(F32)
        shared = K.linear(a.astype(x.dtype), lp["ws_down"])
        gate = jax.nn.sigmoid(K.linear(x, lp["ws_router"]).astype(F32))
        shared = (shared.astype(F32) * gate).astype(x.dtype)

    return (routed + shared).reshape(B, T, H)


def layer(cfg: Config, x, lp, k_layer, v_layer, start):
    """One decoder layer. Returns (x, k_new, v_new) where k/v_new are this
    step's T new cache entries (the caller writes them into the big cache)."""
    B, T, _ = x.shape
    Hq, Hkv, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    pos = start + jnp.arange(T)

    with jax.named_scope("attn"):
        h = K.rmsnorm(x, lp["attn_norm"], cfg.eps)
        q = K.linear(h, lp["wq"], lp["bq"]).reshape(B, T, Hq, D)
        k = K.linear(h, lp["wk"], lp["bk"]).reshape(B, T, Hkv, D)
        v = K.linear(h, lp["wv"], lp["bv"]).reshape(B, T, Hkv, D)
        q, k = rope(q, pos, cfg.rope_theta), rope(k, pos, cfg.rope_theta)
        k_layer = jax.lax.dynamic_update_slice(k_layer, k.astype(k_layer.dtype), (0, start, 0, 0))
        v_layer = jax.lax.dynamic_update_slice(v_layer, v.astype(v_layer.dtype), (0, start, 0, 0))
        o = K.attention(q, k_layer, v_layer, start)
        x = x + K.linear(o.reshape(B, T, Hq * D), lp["wo"])

    with jax.named_scope("moe"):
        x = x + moe_block(cfg, K.rmsnorm(x, lp["mlp_norm"], cfg.eps), lp)
    return x, k, v


def forward(cfg: Config, params, tokens, cache, start):
    """tokens [B, T] -> final hidden states [B, T, H], updated cache."""
    x = params["embed"][tokens]

    def body(carry, inputs):
        x, ck, cv = carry
        lp, i = inputs
        x, k, v = layer(cfg, x, lp, ck[i], cv[i], start)
        # Write only the new tokens into the stacked cache (in place in XLA).
        ck = jax.lax.dynamic_update_slice(ck, k[None].astype(ck.dtype), (i, 0, start, 0, 0))
        cv = jax.lax.dynamic_update_slice(cv, v[None].astype(cv.dtype), (i, 0, start, 0, 0))
        return (x, ck, cv), None

    (x, ck, cv), _ = jax.lax.scan(
        body, (x, cache["k"], cache["v"]), (params["layers"], jnp.arange(cfg.n_layers))
    )
    return K.rmsnorm(x, params["final_norm"], cfg.eps), {"k": ck, "v": cv}


def new_cache(cfg: Config, max_seq: int, batch: int = 1, dtype=jnp.bfloat16):
    shape = (cfg.n_layers, batch, max_seq, cfg.n_kv_heads, cfg.head_dim)
    return {"k": jnp.zeros(shape, dtype), "v": jnp.zeros(shape, dtype)}


@partial(jax.jit, static_argnums=(0, 1), donate_argnums=(4,))
def step(cfg: Config, ksig: tuple, params, tokens, cache, start, last_idx):
    """Run T tokens at positions [start, start+T); greedy-pick the token after
    position last_idx. Returns (next_id [B], its logprob [B], cache).
    ksig = kernels.signature(): static, so switching kernels recompiles."""
    h, cache = forward(cfg, params, tokens, cache, start)
    logits = K.linear(h[:, last_idx], params["lm_head"]).astype(F32)  # [B, V]
    logp = jax.nn.log_softmax(logits, axis=-1)
    next_id = jnp.argmax(logits, axis=-1)
    return next_id, jnp.take_along_axis(logp, next_id[:, None], 1)[:, 0], cache


@partial(jax.jit, static_argnums=(0, 1), donate_argnums=(4,))
def score_step(cfg: Config, ksig: tuple, params, tokens, cache, start, targets):
    """Teacher forcing: logprob of targets[t] (the token after position t) and
    the model's argmax at every position. For checking against a reference."""
    h, cache = forward(cfg, params, tokens, cache, start)
    logp = jax.nn.log_softmax(K.linear(h, params["lm_head"]).astype(F32), axis=-1)
    tgt = jnp.take_along_axis(logp, targets[..., None], -1)[..., 0]
    return tgt, jnp.argmax(logp, axis=-1), cache
