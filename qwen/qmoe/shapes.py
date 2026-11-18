"""Benchmark/test cases for each kernel op at the model's real shapes.

Each Case carries the *ideal* HBM traffic (bytes an optimal kernel must
read + write) and FLOPs, so a timing turns into roofline numbers:
achieved GB/s vs peak bandwidth, achieved TFLOP/s vs peak compute. At batch
size 1, decode is bandwidth-bound: the % of peak bandwidth is the number to
push up. Note "ideal" for MoE decode counts only the routed experts.
"""

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from qmoe.config import Config
from qmoe.quant import QArray, quantize

# Qwen/Qwen1.5-MoE-A2.7B config.json, so tests/benches don't need the weights.
QWEN15_MOE = Config(
    vocab_size=151936, hidden=2048, n_layers=24, n_heads=16, n_kv_heads=16, head_dim=128,
    n_experts=60, top_k=4, moe_inter=1408, shared_inter=5632, rope_theta=1e6, eps=1e-6,
    norm_topk=False,
)

BF16 = jnp.bfloat16
WEIGHT_FORMATS = (("bf16", 0, None), ("int8-g128", 8, 128), ("int4-g128", 4, 128))


@dataclass
class Case:
    name: str
    args: tuple
    bytes: int  # ideal HBM bytes moved
    flops: int
    static: dict = field(default_factory=dict)  # Python constants (kwargs), not traced

    def bind(self, fn):
        """fn with the static kwargs filled in; call it with *self.args."""
        return partial(fn, **self.static) if self.static else fn


def _nbytes(w) -> int:
    return w.nbytes if isinstance(w, QArray) else w.size * w.dtype.itemsize


def _fmt(w, bits, group):
    if not bits:
        return w
    return quantize(w, bits, group if w.shape[-2] % group == 0 else None)  # tiny shapes: per-channel


def _rand(key, shape, scale=1.0):
    return (jax.random.normal(key, shape, jnp.float32) * scale).astype(BF16)


def _lens(cfg: Config):
    tiny = cfg.hidden < 1024
    return {
        "chunk": 16 if tiny else 512,  # prefill chunk
        "max_seq": 64 if tiny else 8192,  # cache length
        "ctx": 24 if tiny else 2048,  # tokens already in the cache
    }


def op_cases(op: str, cfg: Config, key=jax.random.PRNGKey(0), formats=WEIGHT_FORMATS) -> list[Case]:
    L = _lens(cfg)
    H, k1, k2, k3, k4 = cfg.hidden, *jax.random.split(key, 4)
    cases = []

    if op == "rmsnorm":
        w = 1 + _rand(k2, (H,), 0.1)
        for T in (1, L["chunk"]):
            x = _rand(k1, (1, T, H))
            cases.append(Case(f"T={T}", (x, w), 2 * x.nbytes + w.nbytes, 4 * x.size, {"eps": cfg.eps}))

    elif op == "linear":
        QD = cfg.n_heads * cfg.head_dim
        mats = [("wq", H, QD), ("shared_up", H, cfg.shared_inter)]
        for wname, Kd, N in mats:
            w = _rand(k2, (Kd, N), Kd**-0.5)
            b = _rand(k3, (N,), 0.1) if wname == "wq" else None
            for fname, bits, group in formats:
                wf = _fmt(w, bits, group)
                for T in (1, L["chunk"]):
                    x = _rand(k1, (1, T, Kd))
                    io = x.nbytes + T * N * 2
                    cases.append(Case(f"{wname} {fname} T={T}", (x, wf, b), _nbytes(wf) + io, 2 * T * Kd * N))

    elif op == "attention":
        Hq, Hkv, D, S = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, L["max_seq"]
        kc, vc = _rand(k2, (1, S, Hkv, D)), _rand(k3, (1, S, Hkv, D))
        for T in (1, L["chunk"]):
            start = L["ctx"]
            q = _rand(k1, (1, T, Hq, D))
            used = start + T  # keys a causal kernel actually needs
            kv_bytes = 2 * used * Hkv * D * 2
            cases.append(Case(f"T={T} ctx={start} S={S}", (q, kc, vc, jnp.int32(start)),
                              kv_bytes + 2 * q.nbytes, 4 * T * used * Hq * D))

    elif op == "moe_experts":
        E, I, k = cfg.n_experts, cfg.moe_inter, cfg.top_k
        wg, wu = _rand(k2, (E, H, I), H**-0.5), _rand(k3, (E, H, I), H**-0.5)
        wd = _rand(k4, (E, I, H), I**-0.5)
        rng = np.random.RandomState(0)
        for fname, bits, group in formats:
            ws = tuple(_fmt(w, bits, group) for w in (wg, wu, wd))
            per_expert = sum(_nbytes(w) for w in ws) // E
            for T in (1, L["chunk"]):
                ids = np.stack([rng.choice(E, k, replace=False) for _ in range(T)]).astype(np.int32)
                wts = rng.dirichlet(np.ones(k), size=T).astype(np.float32)
                x = _rand(k1, (T, H))
                n_used = len(np.unique(ids))
                cases.append(Case(f"{fname} T={T} ({n_used}/{E} experts)",
                                  (x, jnp.asarray(ids), jnp.asarray(wts), *ws),
                                  n_used * per_expert + 2 * x.nbytes, 2 * T * k * 3 * H * I))
    else:
        raise ValueError(op)
    return cases
