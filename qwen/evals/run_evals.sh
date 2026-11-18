#!/usr/bin/env bash
# Run MATH-500 and a RULER subset against the sglang-jax server on the Colab VM.
#
# Run from the Mac (colab exec runs Python, so the %%bash line makes it a shell cell):
#   (echo %%bash; cat evals/run_evals.sh) | colab exec -s qwen-moe
# With overrides, e.g. a 2-sample smoke test:
#   (echo %%bash; echo 'export LIMIT=2 RULER_LIMIT=2'; cat evals/run_evals.sh) | colab exec -s qwen-moe
#
# Env overrides: LIMIT, RULER_LIMIT, RULER_LENGTHS, RULER_TASKS, RUN (math,ruler), OUT.
set -euo pipefail

# lm-eval only talks HTTP, but the Colab VM's PyTorch auto-loads the torch_tpu
# plugin, which grabs the TPU the server already owns ("TPU is already in use")
# when lm-eval seeds torch. Keep torch (and JAX) on CPU.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export JAX_PLATFORMS=cpu

MODEL=${MODEL:-Qwen/Qwen1.5-MoE-A2.7B}
BASE_URL=${BASE_URL:-http://localhost:30000/v1/completions}
OUT=${OUT:-/content/drive/MyDrive/evals/qwen1.5-moe}
LIMIT=${LIMIT:-}               # MATH-500: blank = all 500 problems
RULER_LIMIT=${RULER_LIMIT:-25} # RULER: samples per task per length
RULER_LENGTHS=${RULER_LENGTHS:-[4096,8192]}   # model's max context is 8192
RULER_TASKS=${RULER_TASKS:-niah_single_1,niah_single_2,niah_multikey_1,niah_multivalue,ruler_vt,ruler_cwe}
RUN=${RUN:-math,ruler}

mkdir -p "$OUT"
MODEL_ARGS="model=$MODEL,base_url=$BASE_URL,tokenizer=$MODEL,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=1,max_retries=3,max_length=8192"

if [[ $RUN == *math* ]]; then
  # Base model -> few-shot. minerva_math500 = MATH-500 with the 4-shot Minerva
  # prompt, greedy, graded by math-verify (exact_match).
  lm_eval --model local-completions --model_args "$MODEL_ARGS" \
    --tasks minerva_math500 \
    ${LIMIT:+--limit $LIMIT} \
    --gen_kwargs temperature=0 \
    --output_path "$OUT/math500" --log_samples
fi

if [[ $RUN == *ruler* ]]; then
  lm_eval --model local-completions --model_args "$MODEL_ARGS" \
    --tasks "$RULER_TASKS" \
    --metadata "{\"max_seq_lengths\":$RULER_LENGTHS,\"tokenizer\":\"$MODEL\"}" \
    --limit "$RULER_LIMIT" \
    --gen_kwargs temperature=0 \
    --output_path "$OUT/ruler" --log_samples
fi
