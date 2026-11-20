# qmoe: Qwen1.5-MoE-A2.7B inference in JAX

Batch-1 inference engine for Qwen1.5-MoE-A2.7B (14.3B params, 2.4B active per token)
with swappable kernels and int8/int4 quantization. Everything real runs on a
**Colab TPU v6e** (`v6e1`). The bf16 model is 28.6 GB and the v6e has 32 GB of HBM,
so the Mac only runs the tiny-model tests.

## Layout

| Path | What it is |
|---|---|
| `qmoe/model.py` | Forward pass (layers stacked, run with `lax.scan`) |
| `qmoe/engine.py` | Chunked prefill + greedy decode loop |
| `qmoe/kernels/` | Kernel registry; `reference.py` is the plain-JAX baseline |
| `qmoe/weights.py` | safetensors loader, optional quantization |
| `scripts/bench.py` | Speed: decode tok/s, MBU, TTFT |
| `scripts/evals.py` | Quality: MATH-500 (100 problems) + RULER 4k/8k via lm-eval |
| `scripts/serve.py` | `/v1/completions` server around the engine (used by evals) |
| `scripts/results.py` | Cached baseline + side-by-side comparison |
| `results/baseline.json` | Cached bf16 baseline (bench + evals) |
| `results/runs/<name>.json` | Your runs |

## Baseline

bf16 weights, all reference kernels, Colab v6e, 512-token prompt, 128 generated tokens:

| Metric | Baseline | Best possible on v6e |
|---|---|---|
| Decode | 14.8 tok/s (67.35 ms/token) | 345 tok/s (2.90 ms/token) |
| MBU | 4.3% | 100% |
| TTFT (512-token prefill) | 69.7 ms | ~2.7 ms |

MBU = bytes a perfect decode step must read (4.76 GB: attention, router, the 4 routed
experts, the shared expert and lm_head) ÷ time per token ÷ peak bandwidth (1640 GB/s).

## Local tests (Mac, CPU)

```bash
uv sync
uv run python -m scripts.test_model          # KV cache + quantization, tiny random model
uv run python -m scripts.test_kernels --tiny
```

## Running on Colab

Uses the `colab` CLI (`uv tool install google-colab-cli`). Run these from this directory.

### 1. Start a v6e and upload the code

```bash
colab new -s qmoe --tpu v6e1
tar czf /tmp/qwen.tgz --exclude .venv --exclude __pycache__ --exclude uv.lock .
colab upload -s qmoe /tmp/qwen.tgz /content/qwen.tgz
echo '!mkdir -p /content/qwen && tar xzf /content/qwen.tgz -C /content/qwen' | colab exec -s qmoe
```

After changing a file, upload just that file:
`colab upload -s qmoe qmoe/weights.py /content/qwen/qmoe/weights.py`

### 2. Install dependencies

The Colab TPU image already has JAX and libtpu.

```bash
echo '!pip install -q "huggingface-hub>=0.26" "ml-dtypes>=0.7"' | colab exec -s qmoe
# Only needed for evals; takes 10+ minutes, so run it in the background:
echo '!nohup pip install -q "lm-eval[api,math,ruler]" > /content/pip.log 2>&1 &' | colab exec -s qmoe
```

### 3. Run long jobs in the background

Jobs here take minutes to hours, longer than a `colab exec` call stays connected, so start
them with `nohup ... &` and check the log:

```bash
echo '!cd /content/qwen && nohup python -m scripts.bench --model --name mine > /content/bench.log 2>&1 &' | colab exec -s qmoe
echo '!tail -c 2000 /content/bench.log' | colab exec -s qmoe --timeout 60
```

The first run on a fresh VM downloads the weights (~29 GB, about 2 minutes) into
`/root/.cache/huggingface`. They stay there until the VM is stopped. To make sure nothing
downloads again, prefix commands with `HF_HUB_OFFLINE=1`.

### 4. Copy results back and stop the VM

```bash
colab download -s qmoe /content/qwen/results/runs/mine.json results/runs/mine.json
colab stop -s qmoe        # an idle v6e costs ~5 compute units/hour
```

## Speed benchmark

```bash
# Your setup: saved to results/runs/<name>.json, printed next to the cached baseline
python -m scripts.bench --model --name int4 --quant int4
python -m scripts.bench --model --name my-moe --kernels moe_experts=mine

# Per-op microbenchmarks at real shapes (no weights needed)
python -m scripts.bench --ops
python -m scripts.bench --ops --op moe_experts --kernels moe_experts=mine

# Profiler trace (open the .trace.json.gz at https://ui.perfetto.dev)
python -m scripts.bench --model --name trace --trace /content/traces/run1
```

Example output:

```
bench vs cached baseline (TPU v6 lite, bf16, reference kernels)
  metric            baseline      yours  change
  decode tok/s          14.8       41.2  +178.4% better
  MBU %                  4.3       12.0  +179.1% better
  TTFT ms               69.7       52.3  +25.0% better
  ...
```

(These numbers only illustrate the format; they aren't real results.)

## Quality evals

MATH-500 (first 100 problems, 4-shot, greedy) and six RULER tasks at 4k and 8k context
(25 samples per task per length). `scripts.evals` starts `scripts.serve` on the TPU, points
`evals/run_evals.sh` (lm-eval) at it, and caches the scores.

```bash
python -m scripts.evals --name int4 --quant int4
python -m scripts.evals --name my-moe --kernels moe_experts=mine --run math
python -m scripts.evals --name smoke --limit 2 --ruler-limit 1   # quick check that everything runs
```

Server log and lm-eval's per-sample logs go to `results/evals/<name>/`.
Expect a full run (100 MATH + RULER) to take about 1.5–2 hours with the reference kernels.

## The cached baseline

Comparisons never re-run the baseline. Both `bench` and `evals` read
`results/baseline.json` and print it next to your numbers. The baseline is only re-measured
when you ask for it:

```bash
python -m scripts.bench --model --reset-baseline   # re-measure bf16 speed
python -m scripts.evals --reset-baseline           # re-run bf16 evals
python -m scripts.results                          # show what's cached
python -m scripts.results results/runs/int4.json   # baseline vs a saved run
```

`--reset-baseline` refuses to run unless the setup is bf16 with all reference kernels, so the
baseline can't be overwritten by a different configuration by mistake. Keep `--limit` the
same as the baseline's (default 100) so the MATH scores compare.

## Gotchas

- **Don't `import jax` in the Colab kernel itself** (a bare Python line in `colab exec`).
  The kernel then holds the TPU, and every script fails with "TPU is already in use". Run
  everything as `!python ...`. If it happens: `colab restart-kernel -s qmoe`.
- **`colab ssh` doesn't work** here (no SSH key set up). Use `colab exec` with `!` lines.
- **Uploading into a folder that doesn't exist on the VM fails** with a 500. Create it first
  with `!mkdir -p`.
- **Stop the VM when you're done.** It keeps billing while idle, and stopping it deletes
  the downloaded weights.
- **`--max-seq` affects speed.** The reference attention reads the whole KV cache every
  step. The bench uses 2048. The eval server uses 8704 (8k context + generation), so its
  decode is slower than the bench.
