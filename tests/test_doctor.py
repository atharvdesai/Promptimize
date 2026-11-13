import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import promptimize_doctor as doctor

PASS, WARN, FAIL, SKIP, INFO = doctor.PASS, doctor.WARN, doctor.FAIL, doctor.SKIP, doctor.INFO


class DoctorTestBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = self.dir.name
        self.paths = {
            "model": os.path.join(d, "model.onnx"),
            "head": os.path.join(d, "gate_head.json"),
            "budget": os.path.join(d, "budget.json"),
            "usage_log": os.path.join(d, "sub", "usage.jsonl"),
            "token": os.path.join(d, "serve_token"),
            "serve_url": "http://127.0.0.1:9",  # nothing listens here
            "claude_settings": os.path.join(d, "settings.json"),
            "probes": os.path.join(d, "probes.jsonl"),
            "probe_result": os.path.join(d, "probe_result.json"),
        }

    def write(self, key, content, mode=None):
        with open(self.paths[key], "w") as f:
            f.write(content)
        if mode is not None:
            os.chmod(self.paths[key], mode)

    def good_head(self):
        return json.dumps({"format": "promptimize-gate-head/v1",
                           "w": [1.0, 2.0], "mu": [0.0, 0.0], "sd": [1.0, 1.0],
                           "b": 0.0, "threshold": 0.05})


class BasicChecks(DoctorTestBase):
    def test_python_passes(self):
        status, name, detail, fix = doctor.check_python({}, self.paths)
        self.assertEqual(status, PASS)

    def test_deps_missing_fails(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            status, name, detail, fix = doctor.check_deps({}, self.paths)
        self.assertEqual(status, FAIL)
        self.assertIn("pip install", fix)

    def test_model_missing_fails(self):
        status, _, detail, fix = doctor.check_model_file({}, self.paths)
        self.assertEqual(status, FAIL)
        self.assertIn("curl", fix)

    def test_model_truncated_fails(self):
        with open(self.paths["model"], "wb") as f:
            f.truncate(10_000_000)
        status, _, detail, _ = doctor.check_model_file({}, self.paths)
        self.assertEqual(status, FAIL)
        self.assertIn("truncated", detail)

    def test_model_present_passes(self):
        with open(self.paths["model"], "wb") as f:
            f.truncate(101_000_000)
        status, _, _, _ = doctor.check_model_file({}, self.paths)
        self.assertEqual(status, PASS)

    def test_head_missing_warns(self):
        status, _, _, _ = doctor.check_head_artifact({}, self.paths)
        self.assertEqual(status, WARN)

    def test_head_bad_json_fails(self):
        self.write("head", "{nope")
        status, _, _, _ = doctor.check_head_artifact({}, self.paths)
        self.assertEqual(status, FAIL)

    def test_head_wrong_format_fails(self):
        self.write("head", json.dumps({"format": "other"}))
        status, _, detail, _ = doctor.check_head_artifact({}, self.paths)
        self.assertEqual(status, FAIL)
        self.assertIn("format", detail)

    def test_head_dim_mismatch_fails(self):
        self.write("head", json.dumps({"format": "promptimize-gate-head/v1",
                                       "w": [1.0], "mu": [0.0, 0.0], "sd": [1.0]}))
        status, _, detail, _ = doctor.check_head_artifact({}, self.paths)
        self.assertEqual(status, FAIL)
        self.assertIn("length", detail)

    def test_head_good_passes(self):
        self.write("head", self.good_head())
        status, _, detail, _ = doctor.check_head_artifact({}, self.paths)
        self.assertEqual(status, PASS)
        self.assertIn("2-dim", detail)

    def test_normalization_contract(self):
        status, _, _, _ = doctor.check_normalization({}, self.paths)
        self.assertEqual(status, PASS)

    def test_decider_head_passes(self):
        with mock.patch("promptimize_gate.decider_name", return_value="head"):
            status, _, _, _ = doctor.check_decider({}, self.paths)
        self.assertEqual(status, PASS)

    def test_decider_fallback_warns(self):
        with mock.patch("promptimize_gate.decider_name", return_value="noul"):
            status, _, _, _ = doctor.check_decider({}, self.paths)
        self.assertEqual(status, WARN)


class ScoringCheck(DoctorTestBase):
    def setUp(self):
        super().setUp()
        with open(self.paths["model"], "wb") as f:
            f.truncate(101_000_000)
        self.write("head", self.good_head())
        self.write("probes", json.dumps({"id": "A10", "prompt": "amb prompt"}) + "\n"
                   + json.dumps({"id": "E1", "prompt": "exec prompt"}) + "\n")
        self.write("probe_result", json.dumps({"per_probe": [
            {"id": "A10", "p_ambiguous": 0.9262}, {"id": "E1", "p_ambiguous": 0.0}]}))

    def test_skip_without_probe_expectations(self):
        os.remove(self.paths["probes"])
        status, _, _, _ = doctor.check_scoring({}, self.paths)
        self.assertEqual(status, SKIP)

    def test_pass_within_bands(self):
        # head w=[1,2], b=0 on emb [1,1] -> z=3 -> p_amb ~0.953 (ask side);
        # the same fake emb is used for both probes, so force per-id scores.
        scores = {"amb prompt": 0.9, "exec prompt": 0.01}

        def fake_score(emb, head):
            return scores[emb[0]]

        with mock.patch.object(doctor, "_deps_available", return_value=True), \
             mock.patch("promptimize_gate.embed_prompt",
                        side_effect=lambda p: [p]), \
             mock.patch("promptimize_gate.load_head", return_value={}), \
             mock.patch("promptimize_gate.score_head", side_effect=fake_score):
            status, _, detail, _ = doctor.check_scoring({}, self.paths)
        self.assertEqual(status, PASS)
        self.assertIn("A10", detail)

    def test_fail_out_of_band(self):
        with mock.patch.object(doctor, "_deps_available", return_value=True), \
             mock.patch("promptimize_gate.embed_prompt",
                        side_effect=lambda p: [p]), \
             mock.patch("promptimize_gate.load_head", return_value={}), \
             mock.patch("promptimize_gate.score_head", return_value=0.5):
            status, _, detail, fix = doctor.check_scoring({}, self.paths)
        # E1 at 0.5 violates the pass-side band
        self.assertEqual(status, FAIL)
        self.assertIn("E1", detail)


class EnvChecks(DoctorTestBase):
    def test_key_cmd_passes(self):
        status, _, _, _ = doctor.check_key(
            {"PROMPTIMIZE_ANTHROPIC_KEY_CMD": "security find-generic-password -w"}, self.paths)
        self.assertEqual(status, PASS)

    def test_key_file_missing_fails(self):
        status, _, _, _ = doctor.check_key(
            {"PROMPTIMIZE_ANTHROPIC_KEY_FILE": self.paths["token"]}, self.paths)
        self.assertEqual(status, FAIL)

    def test_key_file_world_readable_fails(self):
        self.write("token", "sk-ant-x", mode=0o644)
        status, _, detail, _ = doctor.check_key(
            {"PROMPTIMIZE_ANTHROPIC_KEY_FILE": self.paths["token"]}, self.paths)
        self.assertEqual(status, FAIL)
        self.assertIn("refuses", detail)

    def test_key_file_0600_passes(self):
        self.write("token", "sk-ant-x", mode=0o600)
        status, _, _, _ = doctor.check_key(
            {"PROMPTIMIZE_ANTHROPIC_KEY_FILE": self.paths["token"]}, self.paths)
        self.assertEqual(status, PASS)

    def test_key_unset_warns(self):
        status, _, _, _ = doctor.check_key({}, self.paths)
        self.assertEqual(status, WARN)

    def test_budget_missing_warns(self):
        status, _, _, fix = doctor.check_budget({}, self.paths)
        self.assertEqual(status, WARN)
        self.assertIn("init", fix)

    def test_budget_corrupt_fails(self):
        self.write("budget", "{bad")
        status, _, _, _ = doctor.check_budget({}, self.paths)
        self.assertEqual(status, FAIL)

    def test_budget_good_passes(self):
        self.write("budget", json.dumps({"cap_usd": 10.0}))
        status, _, _, _ = doctor.check_budget({}, self.paths)
        self.assertEqual(status, PASS)

    def test_usage_log_writable(self):
        status, _, _, _ = doctor.check_usage_log({}, self.paths)
        self.assertEqual(status, PASS)
        line = open(self.paths["usage_log"]).read().strip()
        self.assertEqual(json.loads(line)["kind"], "doctor")
        self.assertNotIn("prompt", line)

    def test_server_disabled_info(self):
        status, _, _, _ = doctor.check_server({"PROMPTIMIZE_SERVE": "0"}, self.paths)
        self.assertEqual(status, INFO)

    def test_server_unreachable_info(self):
        status, _, detail, fix = doctor.check_server({}, self.paths)
        self.assertEqual(status, INFO)
        self.assertIn("falls back", detail)

    def test_server_reachable_passes(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        self.write("token", "tok123", mode=0o600)
        self.paths["serve_url"] = "http://127.0.0.1:%d" % srv.server_address[1]
        status, _, detail, _ = doctor.check_server({}, self.paths)
        self.assertEqual(status, PASS)

    def test_server_token_perms_warn(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        self.write("token", "tok123", mode=0o644)
        self.paths["serve_url"] = "http://127.0.0.1:%d" % srv.server_address[1]
        status, _, detail, _ = doctor.check_server({}, self.paths)
        self.assertEqual(status, WARN)

    def test_claude_wiring_missing_warns(self):
        status, _, _, _ = doctor.check_claude_wiring({}, self.paths)
        self.assertEqual(status, WARN)

    def test_claude_wiring_without_hook_warns(self):
        self.write("claude_settings", json.dumps({"hooks": {}}))
        status, _, _, _ = doctor.check_claude_wiring({}, self.paths)
        self.assertEqual(status, WARN)

    def test_claude_wiring_with_hook_passes(self):
        self.write("claude_settings", json.dumps(
            {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command",
             "command": "python3 /x/plugin/promptimize_gate.py"}]}]}}))
        status, _, _, _ = doctor.check_claude_wiring({}, self.paths)
        self.assertEqual(status, PASS)


class RunChecksIntegration(DoctorTestBase):
    def test_run_checks_covers_all_and_never_crashes(self):
        results = doctor.run_checks(env={})
        self.assertEqual(len(results), len(doctor.ALL_CHECKS))
        for r in results:
            self.assertIn(r["status"], (PASS, WARN, FAIL, SKIP, INFO))
            self.assertTrue(r["check"])
            self.assertTrue(r["detail"])

    def test_exit_code_zero_without_fail(self):
        with mock.patch.object(doctor, "run_checks", return_value=[
                {"status": PASS, "check": "x", "detail": "ok", "fix": None},
                {"status": WARN, "check": "y", "detail": "meh", "fix": "do y"}]):
            with self.assertRaises(SystemExit) as cm:
                with mock.patch("sys.stdout"):
                        doctor.main()
        self.assertEqual(cm.exception.code, 0)

    def test_exit_code_one_on_fail(self):
        with mock.patch.object(doctor, "run_checks", return_value=[
                {"status": FAIL, "check": "x", "detail": "bad", "fix": "fix it"}]):
            with self.assertRaises(SystemExit) as cm:
                with mock.patch("sys.stdout"):
                        doctor.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
