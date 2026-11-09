#!/usr/bin/env python3
"""Promptimize decision server: keeps LAYA warm so the hook stops paying
the per-prompt model load.

The Claude Code hook is a fresh process per prompt, so the in-process
path pays LAYA's model load (~4-7 s cold, ~4 s warm) on EVERY prompt.
This tiny stdlib-only server loads the model and the gate head once and
answers decision requests in ~0.3 s. The hook tries it first and falls
back to the in-process path on any error, so the gate still works (just
slower) when the server is not running. It is NOT required.

Run (macOS dogfooding):

    PROMPTIMIZE_LAYA_ONNX=/path/to/model.onnx \
      nohup python3 /path/to/Promptimize/plugin/laya_serve.py \
      > ~/.promptimize/serve.log 2>&1 &

Binds 127.0.0.1 only - and loopback alone is NOT treated as
sufficient: every request must carry a localhost Host header (blocks
DNS-rebinding drive-bys from web pages) and the shared token from
~/.promptimize/serve_token (0600, auto-created on first run; blocks
other local processes). The hook sends the token automatically.
Concurrency is bounded (4 inferences, 503 beyond) and bodies are
capped at 1 MB. Prompt text is never logged; stderr carries request
lines only.

Endpoints:
  GET  /healthz -> {"ok": true, "deciders": [...]}
  POST /score   {"prompt": str, "decider": "head"|"noul"}
                -> {"p_can_execute": float, "decider": str,
                    "p_ambiguous": float|null, "latency_ms": int}

Config (env):
  PROMPTIMIZE_LAYA_ONNX    path to model.onnx (default: plugin/model.onnx)
  PROMPTIMIZE_SERVE_PORT   bind port (default 8735)
  PROMPTIMIZE_GATE_HEAD    path to gate_head.json (default: plugin/)
  PROMPTIMIZE_SERVE_TOKEN_FILE  token path (default ~/.promptimize/serve_token)
"""

import hmac
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import promptimize_gate as gate

DEFAULT_PORT = 8735
MAX_CONCURRENT = 4  # inferences at once; the hook sends one per prompt
TOKEN_PATH_DEFAULT = os.path.expanduser("~/.promptimize/serve_token")
_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


def _token_path():
    return os.environ.get("PROMPTIMIZE_SERVE_TOKEN_FILE", TOKEN_PATH_DEFAULT)


def _load_token():
    try:
        with open(_token_path()) as f:
            return f.read().strip() or None
    except OSError:
        return None


def ensure_token():
    """The shared localhost token, created 0600 on first use."""
    tok = _load_token()
    if tok:
        return tok
    os.makedirs(os.path.dirname(_token_path()), exist_ok=True)
    tok = secrets.token_hex(16)
    fd = os.open(_token_path(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    return tok


def score(prompt, decider):
    """The exact scoring the hook would do in-process, reusing its
    loaders so train/inference parity (gate_normalize) is unchanged."""
    t0 = time.time()
    if decider == "head":
        p_amb = gate.score_head(gate.embed_prompt(prompt), gate.load_head())
        return {"p_can_execute": 1.0 - p_amb, "decider": "head",
                "p_ambiguous": p_amb,
                "latency_ms": int((time.time() - t0) * 1000)}
    if decider == "noul":
        return {"p_can_execute": gate.decide_noul(prompt), "decider": "noul",
                "p_ambiguous": None,
                "latency_ms": int((time.time() - t0) * 1000)}
    raise ValueError("unknown decider: %r" % decider)


class Handler(BaseHTTPRequestHandler):
    server_version = "promptimize-serve/1"

    def log_message(self, fmt, *args):  # keep access log off stdout; counts only
        sys.stderr.write("serve: " + fmt % args + "\n")

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _prelude(self):
        """Host-header check (blocks DNS-rebinding drive-bys from
        browsers) + shared-token check (blocks other local processes).
        Loopback alone is NOT treated as sufficient. Returns True when
        the request may proceed."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in _ALLOWED_HOSTS:
            self._send(403, {"error": "forbidden host"})
            return False
        expected = getattr(self.server, "token", None)
        if expected is None:  # embedded without a token: refuse, never open
            self._send(500, {"error": "server token not configured"})
            return False
        got = self.headers.get("X-Promptimize-Token", "")
        if not hmac.compare_digest(got, expected):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._prelude():
            return
        if self.path == "/healthz":
            self._send(200, {"ok": True, "deciders": ["head", "noul"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._prelude():
            return
        if self.path != "/score":
            self._send(404, {"error": "not found"})
            return
        if not _inference_slots.acquire(blocking=False):
            self._send(503, {"error": "busy"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 1024 * 1024:
                self._send(400, {"error": "bad content length"})
                return
            data = json.loads(self.rfile.read(length))
            prompt = data.get("prompt")
            decider = data.get("decider", "head")
            if not isinstance(prompt, str) or not prompt:
                self._send(400, {"error": "missing prompt"})
                return
            self._send(200, score(prompt, decider))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # fail closed toward the hook's local path
            self._send(500, {"error": type(e).__name__})
        finally:
            _inference_slots.release()


def main():
    port = int(os.environ.get("PROMPTIMIZE_SERVE_PORT", str(DEFAULT_PORT)))
    # Warm the model + head at startup so the first prompt is fast too.
    gate._load_agent()
    try:
        gate.load_head()
    except Exception:
        pass  # noul-only deployments have no artifact; head requests will 500
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.token = ensure_token()
    sys.stderr.write("promptimize-serve listening on 127.0.0.1:%d (token auth on)\n" % port)
    server.serve_forever()


if __name__ == "__main__":
    main()
