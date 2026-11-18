"""Model-logic tests on a tiny random-weight config. Runs on CPU (your Mac) in seconds.

  1. KV cache: chunked prefill + token-by-token decode must give the same
     logprobs as scoring the whole sequence in one pass.
  2. Quantization: int8 / int4 round-trip error, and how far the quantized
     model's logprobs drift from bf16 (just a sanity bound on random weights;
     real quality is measured with evals on the real model).

    uv run python -m scripts.test_model
"""

import jax
import jax.numpy as jnp
import numpy as np

from qmoe.config import Config
from qmoe.engine import Engine
from qmoe.quant import dequantize, quantize
from qmoe.weights import QuantSpec, param_bytes, random_params


def test_quant_roundtrip():
    w = jax.random.normal(jax.random.PRNGKey(1), (3, 256, 96), jnp.float32).astype(jnp.bfloat16)
    for bits, group, tol in [(8, None, 0.01), (8, 64, 0.01), (4, 64, 0.15), (4, None, 0.2)]:
        qa = quantize(w, bits, group)
        err = jnp.abs(dequantize(qa, jnp.float32) - w.astype(jnp.float32))
        rel = float(err.mean() / jnp.abs(w.astype(jnp.float32)).mean())
        ratio = w.nbytes / qa.nbytes
        print(f"  int{bits} group={group}: mean rel err {rel:.4f}, {ratio:.2f}x smaller")
        assert qa.shape == w.shape and rel < tol, (bits, group, rel)
    # int4 packing must be exactly invertible on representable values
    # (-7..7 so max|w| = 7 -> scale exactly 1; plus a 0 to make K even)
    q = jnp.array([*range(-7, 8), 0], jnp.float32).reshape(16, 1) * jnp.ones((1, 4))
    qa = quantize(q.astype(jnp.bfloat16), 4, None)
    assert jnp.array_equal(dequantize(qa, jnp.float32), q), "int4 pack/unpack mismatch"


def test_cache_consistency(cfg, params):
    ids = list(np.random.RandomState(0).randint(1, cfg.vocab_size, size=40))
    eng = Engine(cfg, params, max_seq=64, chunk=16)  # chunk < prompt: exercises multi-chunk prefill

    # Teacher-forced full pass over the sequence the model itself generates.
    gen = eng.generate(ids[:21], max_new=12)
    full = ids[:21] + gen.ids
    logp_full, pred_full = eng.score(full, start_at=21)

    # score(...)[j] is about position 21 + j, i.e. generated token j
    np.testing.assert_array_equal(pred_full, gen.ids)  # same greedy choices
    np.testing.assert_allclose(logp_full, gen.logprobs, atol=0.05)  # bf16 noise
    print(f"  cached decode == full pass over {len(gen.ids)} tokens "
          f"(max |dlogp| {np.abs(logp_full - gen.logprobs).max():.4f})")


def test_quant_drift(cfg, params_bf16):
    ids = list(np.random.RandomState(1).randint(1, cfg.vocab_size, size=48))
    ref, _ = Engine(cfg, params_bf16, 64, 16).score(ids, 1)
    for quant in ("int8", "int4"):
        spec = QuantSpec.parse(quant, group=32, targets="experts,shared,attn")
        p = random_params(cfg, jax.random.PRNGKey(0), spec=spec)
        lp, _ = Engine(cfg, p, 64, 16).score(ids, 1)
        print(f"  {spec.name()}: {param_bytes(params_bf16) / param_bytes(p):.2f}x smaller, "
              f"mean |dlogp| vs bf16 {np.abs(lp - ref).mean():.4f}")


if __name__ == "__main__":
    print(f"backend: {jax.default_backend()}")
    cfg = Config.tiny()
    params = random_params(cfg, jax.random.PRNGKey(0))
    print("quant round-trip");  test_quant_roundtrip()
    print("kv cache");          test_cache_consistency(cfg, params)
    print("quantized model");   test_quant_drift(cfg, params)
    print("OK")
