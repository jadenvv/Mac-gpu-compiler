"""Generation loop: chunked prefill + one-token decode steps, batch size 1.

Two compiled programs: `step` with T = chunk (prefill) and T = 1 (decode).
Prefill pads the last chunk; the padded positions write junk K/V past the
prompt, but decode overwrites position p before any query at >= p reads it,
and the causal mask hides everything beyond, so the junk is never seen.
"""

import time
from dataclasses import dataclass, field


import jax.numpy as jnp
import numpy as np

from qmoe import kernels as K
from qmoe.config import Config
from qmoe.model import new_cache, score_step, step


@dataclass
class GenResult:
    ids: list[int]
    logprobs: list[float]
    prompt_len: int
    prefill_s: float  # time to first token (includes the prefill of the whole prompt)
    decode_s: float  # time for the remaining tokens
    finish_reason: str = "length"
    timings: dict = field(default_factory=dict)

    @property
    def decode_tok_s(self) -> float:
        n = len(self.ids) - 1
        return n / self.decode_s if n > 0 and self.decode_s > 0 else float("nan")


class Engine:
    def __init__(self, cfg: Config, params, max_seq: int = 8192, chunk: int = 512):
        assert max_seq % chunk == 0, "max_seq must be a multiple of chunk (keeps cache writes in bounds)"
        self.cfg, self.params, self.max_seq, self.chunk = cfg, params, max_seq, chunk
        self._cache = None

    def _fresh_cache(self):
        # Reuse the buffer (step donates it); zeroing isn't needed for correctness.
        if self._cache is None:
            self._cache = new_cache(self.cfg, self.max_seq)
        cache, self._cache = self._cache, None
        return cache

    def _prefill(self, ids, cache):
        ksig = K.signature()
        C = self.chunk
        for s in range(0, len(ids), C):
            piece = ids[s : s + C]
            toks = np.zeros((1, C), np.int32)
            toks[0, : len(piece)] = piece
            nxt, lp, cache = step(self.cfg, ksig, self.params, jnp.asarray(toks), cache, jnp.int32(s), jnp.int32(len(piece) - 1))
        return nxt, lp, cache

    def generate(self, ids: list[int], max_new: int, stop_ids=(), stop_fn=None) -> GenResult:
        """Greedy decode. stop_fn(generated_ids) -> bool is checked every token."""
        n = len(ids)
        assert n >= 1 and n + max_new <= self.max_seq, f"prompt {n} + {max_new} new > max_seq {self.max_seq}"
        ksig = K.signature()
        cache = self._fresh_cache()

        t0 = time.perf_counter()
        nxt, lp, cache = self._prefill(ids, cache)
        out, lps = [int(nxt[0])], [float(lp[0])]  # int() syncs: TTFT is real
        t1 = time.perf_counter()

        reason = "length"
        pos = n
        while True:
            if out[-1] in stop_ids or (stop_fn and stop_fn(out)):
                reason = "stop"
                break
            if len(out) >= max_new:
                break
            tok = jnp.asarray([[out[-1]]], jnp.int32)
            nxt, lp, cache = step(self.cfg, ksig, self.params, tok, cache, jnp.int32(pos), jnp.int32(0))
            out.append(int(nxt[0]))
            lps.append(float(lp[0]))
            pos += 1
        t2 = time.perf_counter()

        self._cache = cache
        return GenResult(out, lps, n, t1 - t0, t2 - t1, reason)

    def score(self, ids: list[int], start_at: int) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-forced: for t in [start_at, len(ids)), logprob of ids[t] given
        ids[:t], and the model's argmax prediction for position t."""
        assert len(ids) <= self.max_seq
        ksig = K.signature()
        cache = self._fresh_cache()
        C = self.chunk
        logps, preds = [], []
        for s in range(0, len(ids) - 1, C):
            piece = ids[s : s + C]
            tgt = ids[s + 1 : s + 1 + C]
            toks = np.zeros((1, C), np.int32)
            tg = np.zeros((1, C), np.int32)
            toks[0, : len(piece)] = piece
            tg[0, : len(tgt)] = tgt
            lp, am, cache = score_step(self.cfg, ksig, self.params, jnp.asarray(toks), cache, jnp.int32(s), jnp.asarray(tg))
            logps.append(np.asarray(lp[0, : len(tgt)]))
            preds.append(np.asarray(am[0, : len(tgt)]))
        self._cache = cache
        logp, pred = np.concatenate(logps), np.concatenate(preds)
        # logp[j] scores ids[j + 1]; return entries for targets t >= start_at.
        return logp[start_at - 1 :], pred[start_at - 1 :]
