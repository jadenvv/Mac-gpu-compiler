"""Load HF safetensors into the stacked parameter tree, optionally quantizing.

Reads safetensors directly (8-byte header length, JSON header, raw bytes) via
mmap + numpy, so bf16 tensors need no torch. Layers are built one at a time;
when quantizing, each layer is moved to the device, quantized there and the
bf16 copy dropped, so peak device memory is the quantized model + one bf16
layer (the full bf16 model is ~29 GB, close to a v6e chip's 32 GB).
"""

import json
import mmap
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from qmoe.config import Config
from qmoe.quant import QArray, quantize

DEFAULT_MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

_DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32}


def resolve(model: str) -> str:
    """Local dir as-is; otherwise a HF repo id, downloaded/cached."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt"])


class SafeTensors:
    def __init__(self, model_dir: str):
        self.index: dict[str, tuple[str, dict]] = {}
        self._maps: dict[str, mmap.mmap] = {}
        for name in sorted(os.listdir(model_dir)):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(model_dir, name)
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            for key, info in header.items():
                info["offset"] = 8 + n
                self.index[key] = (path, info)

    def get(self, key: str) -> np.ndarray:
        path, info = self.index[key]
        if path not in self._maps:
            with open(path, "rb") as f:
                self._maps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        lo, hi = info["data_offsets"]
        buf = memoryview(self._maps[path])[info["offset"] + lo : info["offset"] + hi]
        return np.frombuffer(buf, dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])


@dataclass(frozen=True)
class QuantSpec:
    """Which weights to quantize and how. targets: experts, shared, attn."""

    bits: int = 0  # 0 = keep bf16
    group: int | None = 128
    targets: tuple[str, ...] = ("experts", "shared")

    @classmethod
    def parse(cls, quant: str | None, group: int | None = 128, targets: str = "experts,shared"):
        if not quant or quant == "bf16":
            return cls()
        bits = {"int8": 8, "int4": 4}[quant]
        return cls(bits=bits, group=group, targets=tuple(t for t in targets.split(",") if t))

    def name(self) -> str:
        if not self.bits:
            return "bf16"
        g = f"g{self.group}" if self.group else "perchan"
        return f"int{self.bits}-{g}-{'+'.join(self.targets)}"


_QUANT_GROUPS = {
    "experts": ("w_gate", "w_up", "w_down"),
    "shared": ("ws_gate", "ws_up", "ws_down"),
    "attn": ("wq", "wk", "wv", "wo"),
}


def _t(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.T)  # HF [out, in] -> [in, out]


def _layer_np(st: SafeTensors, cfg: Config, i: int) -> dict[str, np.ndarray]:
    p = f"model.layers.{i}"
    a, m = f"{p}.self_attn", f"{p}.mlp"
    lp = {
        "attn_norm": st.get(f"{p}.input_layernorm.weight"),
        "mlp_norm": st.get(f"{p}.post_attention_layernorm.weight"),
        "router": _t(st.get(f"{m}.gate.weight")),
        "ws_gate": _t(st.get(f"{m}.shared_expert.gate_proj.weight")),
        "ws_up": _t(st.get(f"{m}.shared_expert.up_proj.weight")),
        "ws_down": _t(st.get(f"{m}.shared_expert.down_proj.weight")),
        "ws_router": _t(st.get(f"{m}.shared_expert_gate.weight")),
    }
    for n in ("q", "k", "v"):
        lp[f"w{n}"] = _t(st.get(f"{a}.{n}_proj.weight"))
        lp[f"b{n}"] = st.get(f"{a}.{n}_proj.bias")
    lp["wo"] = _t(st.get(f"{a}.o_proj.weight"))
    for ours, theirs in (("w_gate", "gate_proj"), ("w_up", "up_proj"), ("w_down", "down_proj")):
        lp[ours] = np.stack([_t(st.get(f"{m}.experts.{e}.{theirs}.weight")) for e in range(cfg.n_experts)])
    return lp


def quantize_layer(lp: dict, spec: QuantSpec) -> dict:
    """Quantize the selected matrices of one layer (works on any array type)."""
    if not spec.bits:
        return lp
    names = {n for t in spec.targets for n in _QUANT_GROUPS[t]}
    return {k: quantize(jnp.asarray(v), spec.bits, spec.group) if k in names else v for k, v in lp.items()}


def stack_layers(layers: list[dict]) -> dict:
    """List of per-layer dicts -> one dict with a leading L axis on every leaf."""
    if all(isinstance(v, np.ndarray) for v in layers[0].values()):
        # bf16 path: stack on the host, one transfer per tensor.
        return {k: jnp.asarray(np.stack([lp[k] for lp in layers])) for k in layers[0]}
    # Quantized path: leaves already on device (QArray is a pytree).
    return jax.tree.map(lambda *xs: jnp.stack(xs), *layers)


def load(model: str = DEFAULT_MODEL, spec: QuantSpec = QuantSpec(), verbose: bool = True):
    model_dir = resolve(model)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = Config.from_hf(json.load(f))
    st = SafeTensors(model_dir)

    layers = []
    for i in range(cfg.n_layers):
        lp = _layer_np(st, cfg, i)
        if spec.bits:
            lp = quantize_layer({k: jax.device_put(v) for k, v in lp.items()}, spec)
            jax.block_until_ready(lp)
        layers.append(lp)
        if verbose:
            print(f"\rloaded layer {i + 1}/{cfg.n_layers} ({spec.name()})", end="", flush=True)
    if verbose:
        print()

    params = {
        "embed": jnp.asarray(st.get("model.embed_tokens.weight")),
        "final_norm": jnp.asarray(st.get("model.norm.weight")),
        "lm_head": jnp.asarray(_t(st.get("lm_head.weight"))),
        "layers": stack_layers(layers),
    }
    return cfg, params, model_dir


def random_params(cfg: Config, key, dtype=jnp.bfloat16, spec: QuantSpec = QuantSpec()):
    """Random weights with the real tree structure (for tests on tiny configs)."""
    L, H, E, I, Is = cfg.n_layers, cfg.hidden, cfg.n_experts, cfg.moe_inter, cfg.shared_inter
    QD = cfg.n_heads * cfg.head_dim
    KVD = cfg.n_kv_heads * cfg.head_dim
    shapes = {
        "attn_norm": (H,), "mlp_norm": (H,),
        "wq": (H, QD), "bq": (QD,), "wk": (H, KVD), "bk": (KVD,), "wv": (H, KVD), "bv": (KVD,),
        "wo": (QD, H), "router": (H, E),
        "w_gate": (E, H, I), "w_up": (E, H, I), "w_down": (E, I, H),
        "ws_gate": (H, Is), "ws_up": (H, Is), "ws_down": (Is, H), "ws_router": (H, 1),
    }
    keys = iter(jax.random.split(key, 3 + L * len(shapes)))

    def rnd(shape, scale=None):
        if scale is None:
            scale = shape[-2] ** -0.5 if len(shape) >= 2 else 0.1
        return (jax.random.normal(next(keys), shape, jnp.float32) * scale).astype(dtype)

    layers = []
    for _ in range(L):
        lp = {k: rnd(s) for k, s in shapes.items()}
        lp["attn_norm"] = 1 + lp["attn_norm"]
        lp["mlp_norm"] = 1 + lp["mlp_norm"]
        layers.append(quantize_layer(lp, spec))
    return {
        "embed": rnd((cfg.vocab_size, H), 1.0),
        "final_norm": jnp.ones((H,), dtype),
        "lm_head": rnd((H, cfg.vocab_size)),
        "layers": jax.tree.map(lambda *xs: jnp.stack(xs), *layers),
    }


def param_bytes(params) -> int:
    return sum(x.nbytes for x in jax.tree.leaves(params))
