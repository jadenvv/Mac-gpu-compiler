"""Run MATH-500 and RULER (4k/8k) through the qmoe engine and compare to the cached baseline.

Starts scripts.serve on the TPU, points evals/run_evals.sh (lm-eval) at it,
then saves the scores and prints them next to the cached bf16 baseline.

  # your setup: scores saved to results/runs/<name>.json, baseline read from cache
  python -m scripts.evals --name int4-g128 --quant int4
  python -m scripts.evals --name my-moe --kernels moe_experts=mine --run math

  # hard reset: re-run the bf16 reference model and overwrite the cached baseline
  python -m scripts.evals --reset-baseline

Raw lm-eval output (per-sample logs) goes to results/evals/<name|baseline>/.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

from qmoe import kernels as K
from scripts import results as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wait_ready(url: str, proc, timeout_s: int):
    t = time.time()
    while time.time() - t < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode} (see its log)")
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except OSError:
            time.sleep(5)
    raise SystemExit("server did not come up in time")


def parse_lm_eval(out_dir: str) -> dict:
    """Newest results_*.json under out_dir -> {task: {metric: value}}, stderr dropped."""
    files = sorted(glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no lm-eval results under {out_dir}")
    with open(files[-1]) as f:
        res = json.load(f)["results"]
    return {task: {k.removesuffix(",none"): v for k, v in m.items()
                   if isinstance(v, (int, float)) and "stderr" not in k}
            for task, m in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="save as results/runs/<name>.json and compare to the cached baseline")
    ap.add_argument("--reset-baseline", action="store_true",
                    help="re-run the bf16 reference model and overwrite the cached eval baseline")
    ap.add_argument("--run", default="math,ruler", help="math, ruler, or math,ruler")
    ap.add_argument("--limit", default="100",
                    help="MATH-500 problems (default 100, the first 100; keep it equal to the baseline's so scores compare)")
    ap.add_argument("--ruler-limit", default="25", help="RULER samples per task per length")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--group", default="128")
    ap.add_argument("--targets", default="experts,shared")
    ap.add_argument("--kernels", default="")
    ap.add_argument("--port", default="30000")
    args = ap.parse_args()

    K.use_spec(args.kernels or None)
    if args.reset_baseline:
        if args.quant != "bf16" or any(v != "reference" for v in dict(K.signature()).values()):
            raise SystemExit("--reset-baseline re-measures the baseline: use bf16 with reference kernels")
        name = "baseline"
    elif args.name:
        name = args.name
    else:
        raise SystemExit("pass --name NAME for your run, or --reset-baseline to re-measure the baseline")

    out = os.path.join(R.ROOT, "evals", name)
    os.makedirs(out, exist_ok=True)
    serve_cmd = [sys.executable, "-m", "scripts.serve", "--quant", args.quant, "--group", args.group,
                 "--targets", args.targets, "--port", args.port]
    if args.kernels:
        serve_cmd += ["--kernels", args.kernels]
    log = open(os.path.join(out, "server.log"), "w")
    # HF_HUB_OFFLINE: weights come from the local HF cache, never re-downloaded.
    proc = subprocess.Popen(serve_cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "HF_HUB_OFFLINE": "1"})
    try:
        print(f"starting server: {' '.join(serve_cmd[2:])}", flush=True)
        wait_ready(f"http://127.0.0.1:{args.port}/health", proc, timeout_s=1200)
        print("server ready; running lm-eval", flush=True)
        env = {**os.environ, "OUT": out, "RUN": args.run, "LIMIT": args.limit, "RULER_LIMIT": args.ruler_limit,
               "BASE_URL": f"http://127.0.0.1:{args.port}/v1/completions"}
        subprocess.run(["bash", os.path.join(HERE, "evals", "run_evals.sh")], env=env, check=True)
    finally:
        proc.terminate()
        proc.wait(timeout=60)

    scores = {}
    if "math" in args.run:
        scores.update(parse_lm_eval(os.path.join(out, "math500")))
    if "ruler" in args.run:
        scores.update(parse_lm_eval(os.path.join(out, "ruler")))

    if args.reset_baseline:
        R.save_baseline("evals", scores)
        R.show_evals()
    else:
        print(f"saved run to {R.save_run(name, 'evals', scores)}")
        R.show_evals(scores)


if __name__ == "__main__":
    main()
