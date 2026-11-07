"""Did I beat MLX? Scores your engine's runs against the MLX baseline.

Reads results/mlx_<chip>_<prompt>.json (from bench.py) and every engine run in
results/history_<chip>_<prompt>.jsonl (from `cargo run --release --bin bench`), and prints
a scoreboard plus a chart of decode speed across all your runs.

run with (from metal_infrence_engine/mlx_baseline):
    uv run compare.py --run metal --label "tiled matvec"   # bench your engine, then compare
    uv run compare.py                                       # just re-show the latest results
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CRATE = HERE.parent

GREEN, RED, DIM, BOLD, RESET = ("\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m") if sys.stdout.isatty() else ("",) * 5


def chip_slug() -> str:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return chip.replace("Apple ", "").replace(" ", "_").lower()


def load(prompt: str):
    slug = chip_slug()
    mlx_path = RESULTS / f"mlx_{slug}_{prompt}.json"
    if not mlx_path.exists():
        sys.exit(f"{mlx_path.name} missing: run `uv run bench.py --prompt {prompt}` first")
    hist_path = RESULTS / f"history_{slug}_{prompt}.jsonl"
    runs = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()] if hist_path.exists() else []
    if not runs:
        sys.exit(f"no engine runs yet: `uv run compare.py --run metal` (or --run cpu)")
    return json.loads(mlx_path.read_text()), runs


def metrics(mlx: dict, run: dict) -> list[dict]:
    """Each row: what it is, MLX value, yours, and whether higher is better."""
    return [
        dict(key="decode", name="Decode speed", unit="tok/s", higher=True, headline=True,
             mlx=mlx["out_of_the_box"]["generation_tps_median"], you=run["decode"]["tps"],
             note="vs mlx_lm.stream_generate out of the box"),
        dict(key="latency", name="Time per token", unit="ms", higher=False,
             mlx=mlx["decode"]["time_per_token"]["median_ms"], you=run["decode"]["median_ms"],
             note="median decode step, both synchronous"),
        dict(key="mbu", name="Bandwidth used (MBU)", unit="%", higher=True,
             mlx=mlx["decode"]["mbu"] * 100, you=run["decode"]["mbu"] * 100,
             note="share of peak memory bandwidth"),
        dict(key="ttft_seq", name="TTFT, token-by-token", unit="ms", higher=False,
             mlx=mlx["ttft"]["sequential"]["median_ms"], you=run["ttft"]["median_ms"],
             note="same prefill method as your forward(tok, pos)"),
        dict(key="ttft", name="TTFT vs MLX batched", unit="ms", higher=False,
             mlx=mlx["ttft"]["batched"]["median_ms"], you=run["ttft"]["median_ms"],
             note="what a user waits for; needs batched prefill to win"),
    ]


def speedup(m: dict) -> float:
    """> 1 means you are faster/better."""
    return m["you"] / m["mlx"] if m["higher"] else m["mlx"] / m["you"]


def fmt(v: float, unit: str) -> str:
    if unit == "ms" and v >= 1000:
        return f"{v / 1000:.2f} s"
    return f"{v:.1f}{'%' if unit == '%' else ' ' + unit}"


def describe(run: dict) -> str:
    when = datetime.fromtimestamp(run["timestamp"]).strftime("%b %d %H:%M")
    commit = run["commit"] + ("*" if run["dirty"] else "")
    return f"{run['engine']} · {run['label'] or 'no label'} · {commit} · {when}"


def ceiling_tps(mlx: dict) -> float:
    return mlx["peak_bw_gbs"] * 1e9 / mlx["decode"]["weight_bytes_per_token"]


# ---------------------------------------------------------------------------
# terminal
# ---------------------------------------------------------------------------

def print_scoreboard(mlx: dict, runs: list[dict]):
    run = runs[-1]
    rows = metrics(mlx, run)
    print(f"\n{BOLD}Your engine vs MLX{RESET}  {DIM}{mlx['chip']}, prompt '{mlx['prompt_name']}' "
          f"({len(mlx['prompt'])} tokens), mlx {mlx['mlx']}{RESET}")
    print(f"{DIM}latest run: {describe(run)}{RESET}\n")
    print(f"  {'':<24}{'MLX':>12}{'yours':>12}{'':>4}{'':<22}")
    for m in rows:
        s = speedup(m)
        win = s >= 1
        verdict = (f"{GREEN}✓ {s:.2f}× faster{RESET}" if win else f"{RED}✗ {1 / s:.2f}× slower{RESET}") \
            if m["unit"] != "%" else (f"{GREEN}✓ +{m['you'] - m['mlx']:.1f} pts{RESET}" if win
                                      else f"{RED}✗ {m['you'] - m['mlx']:.1f} pts{RESET}")
        name = f"{BOLD}{m['name']}{RESET}" if m.get("headline") else m["name"]
        pad = 24 + (len(BOLD) + len(RESET) if m.get("headline") else 0)
        print(f"  {name:<{pad}}{fmt(m['mlx'], m['unit']):>12}{fmt(m['you'], m['unit']):>12}    {verdict}")

    dec = rows[0]
    print()
    if dec["you"] >= dec["mlx"]:
        print(f"  {GREEN}{BOLD}You beat MLX at decode by {speedup(dec):.2f}×.{RESET}")
    else:
        print(f"  Need {BOLD}{dec['mlx'] / dec['you']:.2f}×{RESET} faster decode to beat MLX "
              f"({dec['you']:.1f} → >{dec['mlx']:.1f} tok/s).")
    print(f"  {DIM}Bandwidth ceiling on this chip: {ceiling_tps(mlx):.1f} tok/s (100% MBU).{RESET}")

    same = [r for r in runs[:-1] if r["engine"] == run["engine"]]
    if same:
        prev = same[-1]
        d = run["decode"]["tps"] / prev["decode"]["tps"] - 1
        color = GREEN if d >= 0 else RED
        print(f"  vs your previous {run['engine']} run ({prev['label'] or prev['commit']}): "
              f"{color}{d * 100:+.1f}%{RESET} decode")
    p = run["parity_vs_mlx"]
    if p["first_divergence"] is None:
        print(f"  {DIM}Output: greedy tokens match MLX for all {p['compared']}.{RESET}")
    else:
        warn = "" if p["first_divergence"] >= 3 else f"  {RED}← check correctness{RESET}"
        print(f"  {DIM}Output: matches MLX for the first {p['first_divergence']} of {p['compared']} tokens "
              f"(bf16 ties cause some drift){RESET}{warn}")


def print_history(mlx: dict, runs: list[dict], last: int = 12):
    """Decode tok/s for each run as a bar, with MLX marked, so you can see progress."""
    ceiling, target = ceiling_tps(mlx), mlx["out_of_the_box"]["generation_tps_median"]
    top = max([ceiling] + [r["decode"]["tps"] for r in runs])
    width = 40
    mark = round(target / top * width)
    shown = list(enumerate(runs))[-last:]
    print(f"\n{BOLD}Decode speed by run{RESET}  {DIM}(│ = MLX {target:.1f} tok/s, bar end = {top:.0f} tok/s"
          f"{', the bandwidth ceiling' if top == ceiling else ''}){RESET}")
    for i, r in shown:
        tps = r["decode"]["tps"]
        n = round(tps / top * width)
        bar = "".join("█" if k < n else ("│" if k == mark else "·") for k in range(width))
        if mark < width and mark < n:
            bar = bar[:mark] + "│" + bar[mark + 1:]
        color = GREEN if tps >= target else ""
        label = (r["label"] or r["commit"])[:22]
        print(f"  #{i + 1:<3}{r['engine']:<6}{label:<23}{color}{bar}{RESET} {tps:6.1f}")
    if len(runs) > last:
        print(f"  {DIM}({len(runs) - last} older runs not shown){RESET}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=["cpu", "metal"], help="benchmark this engine first (cargo run --release --bin bench)")
    ap.add_argument("--label", default="", help="what you changed, shown in the run history")
    ap.add_argument("--prompt", default="ai", help="prompt preset the MLX baseline was run with (default ai)")
    ap.add_argument("--runs", type=int, default=3, help="TTFT repetitions for --run")
    args = ap.parse_args()

    if args.run:
        cmd = ["cargo", "run", "--release", "--bin", "bench", "--", "--engine", args.run,
               "--prompt", args.prompt, "--runs", str(args.runs), "--label", args.label]
        if subprocess.run(cmd, cwd=CRATE).returncode != 0:
            sys.exit("engine benchmark failed")

    mlx, runs = load(args.prompt)
    print_scoreboard(mlx, runs)
    print_history(mlx, runs)
    print()


if __name__ == "__main__":
    main()
