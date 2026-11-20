"""Cached baseline (bf16, reference kernels) and side-by-side comparison.

results/baseline.json holds {"bench": {...}, "evals": {task: {metric: value}}}.
It is only written when a command is run with --reset-baseline; every other
run compares against it as cached.

  python -m scripts.results                      # show the cached baseline
  python -m scripts.results results/runs/int4.json   # baseline vs a saved run
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BASELINE = os.path.join(ROOT, "baseline.json")
RUNS = os.path.join(ROOT, "runs")

# bench keys to show: (key, label, format, higher_is_better)
BENCH_ROWS = [
    ("decode_tok_s", "decode tok/s", "{:.1f}", True),
    ("ms_per_token", "ms/token", "{:.2f}", False),
    ("mbu_pct", "MBU %", "{:.1f}", True),
    ("ttft_ms", "TTFT ms", "{:.1f}", False),
    ("prefill_tok_s", "prefill tok/s", "{:.0f}", True),
    ("gb_on_device", "GB on device", "{:.2f}", False),
]


def _read(path) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _write(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def load_baseline() -> dict:
    return _read(BASELINE)


def save_baseline(section: str, data: dict):
    """Overwrite one section (bench / evals). Callers only do this on --reset-baseline."""
    b = load_baseline()
    if section == "evals":  # merge per task so math and ruler can be reset separately
        b.setdefault("evals", {}).update(data)
    else:
        b[section] = data
    _write(BASELINE, b)
    print(f"baseline {section} saved to {BASELINE}")


def save_run(name: str, section: str, data: dict) -> str:
    path = os.path.join(RUNS, f"{name}.json")
    r = _read(path)
    if section == "evals":
        r.setdefault("evals", {}).update(data)
    else:
        r[section] = data
    _write(path, r)
    return path


def _delta(b, y, higher_better):
    if not isinstance(b, (int, float)) or not isinstance(y, (int, float)) or b == 0:
        return ""
    pct = 100 * (y - b) / abs(b)
    better = (pct > 0) == higher_better
    return f"{pct:+.1f}% {'better' if better else 'worse'}" if abs(pct) >= 0.05 else "same"


def show_bench(mine: dict | None = None):
    base = load_baseline().get("bench")
    if not base:
        print("no cached bench baseline (run scripts.bench --model --reset-baseline)")
        return
    print(f"\nbench vs cached baseline ({base.get('device')}, {base.get('quant')}, reference kernels)")
    print(f"  {'metric':15s} {'baseline':>10s}" + (f" {'yours':>10s}  change" if mine else ""))
    for key, label, fmt, hib in BENCH_ROWS:
        b = base.get(key)
        line = f"  {label:15s} {fmt.format(b) if b is not None else '-':>10s}"
        if mine:
            y = mine.get(key)
            line += f" {fmt.format(y) if y is not None else '-':>10s}  {_delta(b, y, hib)}"
        print(line)


def show_evals(mine: dict | None = None):
    base = load_baseline().get("evals", {})
    tasks = sorted(set(base) | set(mine or {}))
    if not tasks:
        print("no cached eval baseline (run scripts.evals --reset-baseline)")
        return
    print("\nevals vs cached baseline (bf16, reference kernels)")
    print(f"  {'task':22s} {'metric':22s} {'baseline':>9s}" + (f" {'yours':>9s}  change" if mine else ""))
    for t in tasks:
        metrics = sorted(set(base.get(t, {})) | set((mine or {}).get(t, {})))
        for m in metrics:
            b = base.get(t, {}).get(m)
            line = f"  {t:22s} {m:22s} {b if b is None else f'{b:.4f}':>9}"
            if mine:
                y = mine.get(t, {}).get(m)
                line += f" {y if y is None else f'{y:.4f}':>9}  {_delta(b, y, True)}"
            print(line)


def main():
    mine = _read(sys.argv[1]) if len(sys.argv) > 1 else {}
    show_bench(mine.get("bench"))
    show_evals(mine.get("evals"))


if __name__ == "__main__":
    main()
