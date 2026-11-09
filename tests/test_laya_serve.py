import http.client
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import laya_serve
import promptimize_gate as gate


def hook_input(prompt):
    return json.dumps({
        "session_id": "s1",
        "hook_event_name": "UserPromptSubmit",
        "cwd": "/tmp",
        "prompt": prompt,
    })


class ServeScoreTests(unittest.TestCase):
    """laya_serve.score dispatches to the same loaders the hook uses."""

    def test_head_score_matches_gate_math(self):
        with mock.patch.object(gate, "embed_prompt", return_value=[2.0]):
            with mock.patch.object(gate, "load_head",
                                   return_value={"w": [1.0], "b": 0.0, "mu": [0.0], "sd": [1.0]}):
                out = laya_serve.score("fix the thing", "head")
        self.assertEqual(out["decider"], "head")
        self.assertAlmostEqual(out["p_ambiguous"], 1.0 / (1.0 + pow(2.718281828459045, -2.0)), places=6)
        self.assertAlmostEqual(out["p_can_execute"], 1.0 - out["p_ambiguous"], places=9)

    def test_noul_score_passthrough(self):
        with mock.patch.object(gate, "decide_noul", return_value=0.42):
            out = laya_serve.score("fix the thing", "noul")
        self.assertEqual(out["p_can_execute"], 0.42)
        self.assertIsNone(out["p_ambiguous"])

    def test_unknown_decider_rejected(self):
        with self.assertRaises(ValueError):
            laya_serve.score("fix the thing", "wat")


class ServeHttpTests(unittest.TestCase):
    """HTTP surface: health, scoring, and clean errors."""

    TOKEN = "test-token-0123456789"

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), laya_serve.Handler)
        cls.server.token = cls.TOKEN
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, obj, token=None):
        body = json.dumps(obj).encode()
        headers = {"Content-Type": "application/json"}
        headers["X-Promptimize-Token"] = self.TOKEN if token is None else token
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path), data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_healthz(self):
        req = urllib.request.Request(
            "http://127.0.0.1:%d/healthz" % self.port,
            headers={"X-Promptimize-Token": self.TOKEN})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        self.assertTrue(data["ok"])

    def test_score_roundtrip(self):
        with mock.patch.object(laya_serve, "score",
                               return_value={"p_can_execute": 0.9, "decider": "head",
                                             "p_ambiguous": 0.1, "latency_ms": 3}):
            status, data = self.post("/score", {"prompt": "add an index", "decider": "head"})
        self.assertEqual(status, 200)
        self.assertEqual(data["p_can_execute"], 0.9)

    def test_missing_prompt_is_400(self):
        status, _ = self.post("/score", {"decider": "head"})
        self.assertEqual(status, 400)

    def test_unknown_path_is_404(self):
        status, _ = self.post("/nope", {"prompt": "x"})
        self.assertEqual(status, 404)

    def test_wrong_token_is_401(self):
        status, _ = self.post("/score", {"prompt": "x"}, token="wrong")
        self.assertEqual(status, 401)

    def test_foreign_host_header_is_403(self):
        # DNS-rebinding drive-by: a browser page pointed at 127.0.0.1
        # carries its own Host. http.client lets us set it explicitly.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/score", body=json.dumps({"prompt": "x"}),
                     headers={"Content-Type": "application/json",
                              "X-Promptimize-Token": self.TOKEN,
                              "Host": "evil.example.com"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 403)

    def test_concurrency_cap_returns_503(self):
        slots = [laya_serve._inference_slots.acquire(blocking=False)
                 for _ in range(laya_serve.MAX_CONCURRENT)]
        try:
            status, data = self.post("/score", {"prompt": "x"})
            self.assertEqual(status, 503)
        finally:
            for _ in slots:
                laya_serve._inference_slots.release()

    def test_prompt_text_never_logged(self):
        secret_prompt = "xyzzy-sentence-that-must-not-appear-in-logs"
        buf = io.StringIO()
        with redirect_stderr(buf):
            with mock.patch.object(laya_serve, "score",
                                   return_value={"p_can_execute": 0.5, "decider": "head",
                                                 "p_ambiguous": 0.5, "latency_ms": 1}):
                status, _ = self.post("/score", {"prompt": secret_prompt})
        self.assertEqual(status, 200)
        self.assertNotIn(secret_prompt, buf.getvalue())

    def test_missing_server_token_config_refuses(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), laya_serve.Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/healthz",
                         headers={"X-Promptimize-Token": "anything"})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(resp.status, 500)
        finally:
            server.shutdown()
            server.server_close()


class TokenFileTests(unittest.TestCase):
    def test_ensure_token_creates_0600_and_reuses(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "serve_token")
        with mock.patch.dict(os.environ, {"PROMPTIMIZE_SERVE_TOKEN_FILE": path}):
            tok = laya_serve.ensure_token()
            self.assertEqual(len(tok), 32)
            self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
            self.assertEqual(laya_serve.ensure_token(), tok)  # reused

    def test_load_token_missing_returns_none(self):
        with mock.patch.dict(os.environ,
                             {"PROMPTIMIZE_SERVE_TOKEN_FILE": "/nonexistent/tok"}):
            self.assertIsNone(laya_serve._load_token())


class HookServePathTests(unittest.TestCase):
    """The hook tries the server first and falls back to in-process."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self._old_log = gate.USAGE_LOG
        gate.USAGE_LOG = os.path.join(self.dir.name, "usage.jsonl")
        for var in ("PROMPTIMIZE_SERVE", "PROMPTIMIZE_SERVE_URL",
                    "PROMPTIMIZE_DECIDER", "PROMPTIMIZE_DISABLED"):
            os.environ.pop(var, None)

    def tearDown(self):
        gate.USAGE_LOG = self._old_log
        for var in ("PROMPTIMIZE_SERVE", "PROMPTIMIZE_SERVE_URL"):
            os.environ.pop(var, None)
        self.dir.cleanup()

    def test_serve_disabled_env_skips_http(self):
        os.environ["PROMPTIMIZE_SERVE"] = "0"
        with mock.patch("urllib.request.urlopen") as urlopen:
            self.assertIsNone(gate._serve_score("prompt text here", "head"))
        urlopen.assert_not_called()

    def test_serve_unreachable_returns_none(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertIsNone(gate._serve_score("prompt text here", "head"))

    def test_serve_response_parsed(self):
        class FakeResp:
            def read(self):
                return json.dumps({"p_can_execute": 0.77}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            self.assertEqual(gate._serve_score("prompt text here", "head"), 0.77)

    def test_decide_uses_server_when_available(self):
        with mock.patch.object(gate, "_serve_score", return_value=0.88):
            with mock.patch.object(gate, "decide_noul") as noul:
                with mock.patch.object(gate, "embed_prompt") as emb:
                    self.assertEqual(gate.decide("make the endpoint faster somehow"), 0.88)
        noul.assert_not_called()
        emb.assert_not_called()
        self.assertEqual(gate._last_via, "server")

    def test_decide_falls_back_to_local_noul(self):
        os.environ["PROMPTIMIZE_DECIDER"] = "noul"
        with mock.patch.object(gate, "_serve_score", return_value=None):
            with mock.patch.object(gate, "decide_noul", return_value=0.5):
                self.assertEqual(gate.decide("make the endpoint faster somehow"), 0.5)
        self.assertEqual(gate._last_via, "local")

    def test_decision_log_records_via(self):
        os.environ["PROMPTIMIZE_DECIDER"] = "noul"
        with mock.patch.object(gate, "_serve_score", return_value=0.9):
            gate.run(hook_input("Add an index on created_at to the orders table"))
        with open(gate.USAGE_LOG) as f:
            entries = [json.loads(l) for l in f if '"decision"' in l]
        self.assertEqual(entries[0]["via"], "server")

    def test_hook_sends_token_header_when_file_present(self):
        tok_path = os.path.join(self.dir.name, "serve_token")
        with open(tok_path, "w") as f:
            f.write("abc123")
        os.environ["PROMPTIMIZE_SERVE_TOKEN_FILE"] = tok_path
        try:
            captured = {}

            class FakeResp:
                def read(self):
                    return json.dumps({"p_can_execute": 0.5}).encode()
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    return False

            def fake_urlopen(req, timeout=None):
                captured["headers"] = dict(req.header_items())
                return FakeResp()

            with mock.patch("urllib.request.urlopen", fake_urlopen):
                gate._serve_score("prompt text here", "head")
            self.assertEqual(captured["headers"].get("X-promptimize-token"), "abc123")
        finally:
            os.environ.pop("PROMPTIMIZE_SERVE_TOKEN_FILE", None)


if __name__ == "__main__":
    unittest.main()
