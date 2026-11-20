"""Minimal OpenAI-style /v1/completions server in front of the qmoe Engine,
so lm-eval (`--model local-completions`) can run MATH-500 / RULER against it.

  python -m scripts.serve --quant bf16 --port 30000

Greedy only (temperature is ignored), one request at a time. Handles the fields
lm-eval sends: prompt (str or list of str), max_tokens, stop.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from tokenizers import Tokenizer

from qmoe import kernels as K
from qmoe.engine import Engine
from qmoe.weights import QuantSpec, load


def _first_stop(text: str, stops: list[str]) -> int:
    """Index of the earliest stop string in text, or -1."""
    hits = [i for i in (text.find(s) for s in stops) if i >= 0]
    return min(hits) if hits else -1


class Server:
    def __init__(self, args):
        spec = QuantSpec.parse(args.quant, args.group, args.targets)
        self.cfg, params, model_dir = load(args.model_path, spec)
        self.eng = Engine(self.cfg, params, max_seq=args.max_seq, chunk=args.chunk)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        with open(os.path.join(model_dir, "generation_config.json")) as f:
            eos = json.load(f)["eos_token_id"]
        self.eos = tuple(eos if isinstance(eos, list) else [eos])
        self.max_seq = args.max_seq
        self.eng.generate([self.eos[0]], 2)  # compile before taking requests
        print(f"ready: {spec.name()} kernels={dict(K.signature())} max_seq={args.max_seq}", flush=True)

    def complete(self, prompt: str, max_tokens: int, stops: list[str]) -> tuple[str, str]:
        ids = self.tok.encode(prompt).ids
        ids = ids[-(self.max_seq - 1):]  # keep the end if it can't fit at all
        max_new = max(1, min(max_tokens, self.max_seq - len(ids)))

        def stop_fn(out):
            return bool(stops) and _first_stop(self.tok.decode(out), stops) >= 0

        r = self.eng.generate(ids, max_new, stop_ids=self.eos, stop_fn=stop_fn)
        out = [t for t in r.ids if t not in self.eos]
        text = self.tok.decode(out)
        cut = _first_stop(text, stops) if stops else -1
        if cut >= 0:
            text = text[:cut]
        return text, "stop" if r.finish_reason == "stop" else "length"


def make_handler(srv: Server):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/completions"):
                return self._send(404, {"error": "only /v1/completions"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompts = req["prompt"] if isinstance(req["prompt"], list) else [req["prompt"]]
            stops = req.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            t = time.perf_counter()
            choices = []
            for i, p in enumerate(prompts):
                text, reason = srv.complete(p, int(req.get("max_tokens", 256)), stops)
                choices.append({"index": i, "text": text, "finish_reason": reason, "logprobs": None})
            print(f"{len(prompts)} prompt(s) in {time.perf_counter() - t:.1f}s", flush=True)
            self._send(200, {"object": "text_completion", "model": req.get("model", ""), "choices": choices})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="Qwen/Qwen1.5-MoE-A2.7B")
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--targets", default="experts,shared")
    ap.add_argument("--kernels", help='e.g. "moe_experts=mine" (default: reference)')
    ap.add_argument("--max-seq", type=int, default=8704, help="prompt + generation limit (model context is 8192)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()
    K.use_spec(args.kernels)
    srv = Server(args)
    HTTPServer(("127.0.0.1", args.port), make_handler(srv)).serve_forever()


if __name__ == "__main__":
    main()
