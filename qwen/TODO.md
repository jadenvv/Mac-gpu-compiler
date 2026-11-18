# TODO: quantization quality (Qwen1.5-MoE-A2.7B, bf16 vs int8/int4)

Ground rule: run the reference (bf16) and the quantized model through the **same engine**
(`qmoe`), with the same prompts and greedy decoding. That way any difference comes from
quantization alone, not from engine differences.

## 0. Unblock the engine
- [ ] Fix the import: `qmoe/engine.py` imports `new_cache`, `step`, `score_step` from
      `qmoe.model`, which is an unfinished stub (syntax error). The code lives in
      `qmoe/real_model.py`. Change the import, or rename `real_model.py` to `model.py`.
- [ ] Run `scripts/test_model.py` and `scripts/test_kernels.py --tiny` on the Mac and
      confirm both pass.

## 1. Per-layer error (for debugging)
Run the same inputs through both models and compare activations at each layer.
- [ ] Add a debug path to `real_model.forward` that returns the residual stream after
      each layer (the `scan` can output `x` per layer as its `ys`). Optionally also return
      the attention output (`o @ wo`) and the MoE output separately, to separate the two.
- [ ] For each layer, compute relative error `‖a_q − a_ref‖ / ‖a_ref‖` and cosine
      similarity. Look for spikes, for example one layer with outlier activations, or `wo`.
- [ ] **Router agreement** (MoE): have `moe_block` also return `ids` (the top-k experts).
      For each layer, measure the fraction of tokens whose top-k expert *set* matches the
      reference. This is the same flip measurement as in the determinism experiment, except
      the noise comes from quantization instead of batching.
- [ ] Compare quantization targets (`QuantSpec` targets: `experts`, `shared`, `attn`)
      one at a time to see which weights cause the error.

## 2. Next-token distributions on fixed text (main metric)
Teacher forcing: feed both models the same text and compare their predicted distributions
at every position.
- [ ] `score_step` currently returns only the target token's logprob and the argmax. Add a
      variant that computes KL on the device, so the full 151,936-way distribution never
      leaves it.
- [ ] Both models probably won't fit in memory at once (about 28 GB in bf16). Run the
      reference first and save its output. Saving every position's full distribution is
      about 600 KB per token, too large, so store each position's top-k logprobs (e.g.
      k = 32) plus the log-normalizer, and compute KL from those.
      Compare against exact KL on a short sample to check the approximation.
- [ ] Report: mean KL, **the tail of the KL distribution (p99, max)**, top-1 agreement,
      and the perplexity change (only as a sanity check, since errors in opposite directions
      cancel and hide shifted predictions).
- [ ] Text set: some general prose, some math (MATH-style worked solutions), and some long
      documents (4k–8k tokens), so the long-context positions are covered.
- [ ] Use this to sweep settings quickly: int8 vs int4, group 32/64/128, which targets.

## 3. Free-running generation
- [ ] Generate greedily with both models across many prompts (e.g. the MATH-500 prompts)
      and record the **first token where their outputs diverge** for each prompt.
- [ ] Plot the share of runs still matching against token position. This shows how drift
      compounds, which per-token metrics can't show.

## 4. Task evals (final check, not for tuning)
- [ ] Put an HTTP endpoint (`/v1/completions`) in front of the `qmoe` engine so
      `evals/run_evals.sh` can be pointed at it with `BASE_URL`.
- [ ] Run MATH-500 and RULER (4k/8k) for bf16 and for the chosen quantized setting(s).
      Note the noise: with 500 problems, a MATH-500 score has a standard error of about
      ±2 points, and RULER with 25 samples per task is noisier still, so treat small
      differences as ties.
