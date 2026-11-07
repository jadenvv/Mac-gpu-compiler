# MLX baseline

The out-of-the-box MLX numbers the hand-written Metal engine has to beat: same
unquantized bf16 Qwen3-0.6B (`../qwen3-0.6b`), same prompt token ids, greedy decoding.

## Did I make it faster?

```sh
cd metal_infrence_engine/mlx_baseline
uv run bench.py                                          # once: MLX baseline (long "AI is going to take over..." prompt)
uv run compare.py --run metal --label "tiled matvec"     # every change: bench your engine, print the scoreboard
```

`compare.py` prints, in the terminal, your latest run vs MLX per metric (✓/✗ and the
speedup), how much faster decode needs to get, the change vs your previous run, and a bar
chart of decode tok/s for every run you've recorded with MLX marked. `--run cpu` benches the CPU engine; no `--run` just re-shows results.

The engine side is `cargo run --release --bin bench` (`src/bin/bench.rs`). It reads the prompt
token ids from the MLX results file, so both engines always see the identical input.

Other runs:

```sh
uv run compare.py                        # re-show the latest scoreboard without benching
uv run compare.py --run cpu              # bench the CPU engine instead
uv run bench.py --prompt paris           # once, before using --prompt paris with compare.py
```

## Run natively (arm64), not under Rosetta

MLX won't load under Rosetta, and an x86_64 shell gives misleading engine numbers. Check first:

```sh
sysctl -n sysctl.proc_translated   # must print 0 (1 = Rosetta)
```

If it prints 1:
- One-off: prefix the command, e.g. `arch -arm64 uv run compare.py --run metal --label "..."`.
- Permanent: turn off "Open using Rosetta" for iTerm (Finder → Applications → iTerm → Get Info), then
  quit iTerm fully (⌘Q, not just closing windows) and reopen it. Optionally add this to the top of
  `~/.zprofile` so a shell that starts under Rosetta restarts itself natively:

  ```sh
  if [[ -o interactive && "$(sysctl -n sysctl.proc_translated 2>/dev/null)" == 1 ]]; then
    exec arch -arm64 /bin/zsh -l
  fi
  ```

Also check that `rustc -vV | grep host` says `aarch64-apple-darwin`.

## Prompts

| `--prompt` | tokens | use |
|---|---|---|
| `ai` (default) | 130 | "AI is going to take over the world..." paragraph; exercises prefill and attention over real context |
| `paris` | 5 | the Rust tests' prompt; checks greedy parity against the CPU engine |
| any text | — | `uv run bench.py --prompt "..."`, then `compare.py --prompt custom` |

## What's measured

| metric | MLX side | your side |
|---|---|---|
| Decode speed | `mlx_lm.stream_generate` tok/s, 128 tokens (async pipelined) | 32 synchronous decode steps |
| Time per token / MBU | synchronous per-step, same bytes/token formula as `tests/mbu.rs` | same |
| TTFT token-by-token | prompt fed one token at a time, like `forward(tok, pos)` | same |
| TTFT vs batched | MLX's normal whole-prompt prefill | yours (token-by-token until you add batched prefill) |

Notes:
- Needs a native arm64 Python (pinned in `.python-version`); under Rosetta, MLX won't load.
- MLX keeps the KV cache in bf16 (our engine: f32), so its KV bytes are counted at 2 bytes.
- Greedy output drifts between engines after a few tokens because bf16 logits tie
  (e.g. " France" vs " Italy" both 16.625 on the paris prompt); the CPU engine is the
  correctness reference, not MLX.
