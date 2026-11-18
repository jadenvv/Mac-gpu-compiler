"""Kernel registry: every hot op in the model goes through here.

The model never calls jnp directly for the ops below; it calls `K.rmsnorm(...)`,
`K.linear(...)`, etc. Each op has a "reference" implementation (plain jnp, the
ground truth) and any number of alternatives you register:

    from qmoe.kernels import register

    @register("moe_experts", "mine")
    def moe_experts_mine(x, ids, weights, w_gate, w_up, w_down): ...

Pick implementations with `use(moe_experts="mine")`, or the QMOE_KERNELS env
var / --kernels flag: "moe_experts=mine,rmsnorm=pallas". Selection happens at
trace time, so the choice is baked into each jitted function; `signature()` is
passed to jit as a static arg so changing the selection retraces.

Op signatures (B = batch, T = new tokens, S = cache length, H = hidden):

  rmsnorm(x[..., H], w[H], eps) -> [..., H]
  linear(x[..., K], w[K, N] | QArray, b[N] | None) -> [..., N]
  attention(q[B, T, Hq, D], k_cache[B, S, Hkv, D], v_cache, start) -> [B, T, Hq, D]
      causal: query t (absolute position start + t) sees keys at positions <= start + t
  moe_experts(x[T, H], ids[T, k] int32, weights[T, k] f32,
              w_gate[E, H, I], w_up[E, H, I], w_down[E, I, H]) -> [T, H]
      expert weights may be QArray; y = sum_k weights * down(silu(gate(x)) * up(x))

scripts/test_kernels.py checks every registered implementation against the
reference at the model's real shapes; scripts/bench.py --ops times them.
"""

import os
from collections.abc import Callable

OPS = ("rmsnorm", "linear", "attention", "moe_experts")
_IMPLS: dict[str, dict[str, Callable]] = {op: {} for op in OPS}
_ACTIVE: dict[str, str] = {}


def register(op: str, name: str):
    assert op in OPS, f"unknown op {op!r}; ops are {OPS}"

    def deco(fn):
        assert name not in _IMPLS[op], f"{op}/{name} registered twice"
        _IMPLS[op][name] = fn
        return fn

    return deco


def use(**choices: str) -> None:
    for op, name in choices.items():
        assert op in OPS, f"unknown op {op!r}"
        assert name in _IMPLS[op], f"no {op} impl {name!r}; have {sorted(_IMPLS[op])}"
        _ACTIVE[op] = name


def use_spec(spec: str | None) -> None:
    """Apply "op=name,op=name" (from --kernels or QMOE_KERNELS)."""
    if spec:
        use(**dict(item.split("=", 1) for item in spec.split(",") if item))


def impl(op: str, name: str) -> Callable:
    return _IMPLS[op][name]


def implementations(op: str) -> list[str]:
    return sorted(_IMPLS[op])


def active(op: str) -> str:
    return _ACTIVE.get(op, "reference")


def signature() -> tuple:
    return tuple((op, active(op)) for op in OPS)


def rmsnorm(x, w, eps):
    return _IMPLS["rmsnorm"][active("rmsnorm")](x, w, eps)


def linear(x, w, b=None):
    return _IMPLS["linear"][active("linear")](x, w, b)


def attention(q, k_cache, v_cache, start):
    return _IMPLS["attention"][active("attention")](q, k_cache, v_cache, start)


def moe_experts(x, ids, weights, w_gate, w_up, w_down):
    return _IMPLS["moe_experts"][active("moe_experts")](x, ids, weights, w_gate, w_up, w_down)


# Import implementations so they register themselves. Add your modules here.
from qmoe.kernels import reference  # noqa: E402,F401
from qmoe.kernels import pallas_rmsnorm  # noqa: E402,F401

use_spec(os.environ.get("QMOE_KERNELS"))
