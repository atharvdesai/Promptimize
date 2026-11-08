import json
import os
import tempfile
import unittest
from unittest import mock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import promptimize_gate as gate
from budget_ledger import BudgetExhausted


def hook_input(prompt):
    return json.dumps({
        "session_id": "s1",
        "hook_event_name": "UserPromptSubmit",
        "cwd": "/tmp",
        "prompt": prompt,
    })


class GateHookTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.dir.name, "usage.jsonl")
        self._old_log = gate.USAGE_LOG
        gate.USAGE_LOG = self.log
        os.environ.pop("PROMPTIMIZE_DISABLED", None)
        os.environ.pop("PROMPTIMIZE_THRESHOLD", None)
        # The legacy hook tests below were written against the noul decider's
        # 0.35 posture; pin it. Head-decider tests set their own env.
        os.environ["PROMPTIMIZE_DECIDER"] = "noul"

    def tearDown(self):
        gate.USAGE_LOG = self._old_log
        os.environ.pop("PROMPTIMIZE_DECIDER", None)
        self.dir.cleanup()

    def run_hook(self, prompt, prob):
        with mock.patch.object(gate, "decide", return_value=prob):
            with mock.patch.object(gate, "write_question",
                                   return_value=("Which endpoint is slow?", {"input_tokens": 1500, "output_tokens": 30})):
                return gate.run(hook_input(prompt))

    def test_clear_prompt_produces_no_output(self):
        self.assertIsNone(self.run_hook("Add an index on created_at to the orders table", 0.9))

    def test_flagged_prompt_yields_nudge_json(self):
        out = self.run_hook("make the search endpoint somehow faster", 0.2)
        self.assertIsNotNone(out)
        hso = out["hookSpecificOutput"]
        self.assertEqual(hso["hookEventName"], "UserPromptSubmit")
        self.assertIn("Which endpoint is slow?", hso["additionalContext"])
        self.assertIn("Promptimize", hso["additionalContext"])

    def test_short_prompts_and_slash_commands_skip(self):
        with mock.patch.object(gate, "decide") as decide:
            self.assertIsNone(gate.run(hook_input("hi")))
            self.assertIsNone(gate.run(hook_input("/help fix the thing")))
            decide.assert_not_called()

    def test_decision_failure_fails_open(self):
        with mock.patch.object(gate, "decide", side_effect=RuntimeError("boom")):
            # run() raises; main() must swallow. Exercise main via stdin patch.
            with mock.patch("sys.stdin") as stdin:
                stdin.read.return_value = hook_input("make the search endpoint somehow faster")
                with mock.patch("sys.stdout") as stdout:
                    with self.assertRaises(SystemExit) as cm:
                        gate.main()
        self.assertEqual(cm.exception.code, 0)
        stdout.write.assert_not_called()

    def test_budget_exhaustion_passes_through_silently(self):
        with mock.patch.object(gate, "decide", return_value=0.1):
            with mock.patch.object(gate, "write_question", side_effect=BudgetExhausted("cap")):
                self.assertIsNone(gate.run(hook_input("make the search endpoint somehow faster")))

    def test_question_failure_passes_through_silently(self):
        with mock.patch.object(gate, "decide", return_value=0.1):
            with mock.patch.object(gate, "write_question", side_effect=RuntimeError("http")):
                self.assertIsNone(gate.run(hook_input("make the search endpoint somehow faster")))

    def test_usage_log_never_contains_prompt_text(self):
        secret_prompt = "make the frobnicate endpoint faster please"
        self.run_hook(secret_prompt, 0.2)
        with open(self.log) as f:
            raw = f.read()
        self.assertNotIn(secret_prompt, raw)
        self.assertNotIn("frobnicate", raw)

    def test_question_declined_by_haiku_passes_silently_and_logs(self):
        with mock.patch.object(gate, "decide", return_value=0.1):
            with mock.patch.object(gate, "write_question", return_value=(None, {"input_tokens": 900, "output_tokens": 11})):
                self.assertIsNone(gate.run(hook_input("make the search endpoint somehow faster")))
        with open(self.log) as f:
            self.assertIn("question_declined", f.read())

    def test_haiku_mode_decides_and_asks_without_local_model(self):
        os.environ["PROMPTIMIZE_MODE"] = "haiku"
        try:
            with mock.patch.object(gate, "decide") as decide:
                with mock.patch("haiku_gate.call_gate") as call_gate:
                    call_gate.return_value = (
                        {"can_execute": False, "clarifying_question": "Which endpoint?", "ambiguity_type": "missing_scope"},
                        {"input_tokens": 1000, "output_tokens": 30},
                    )
                    with mock.patch("budget_ledger.BudgetLedger"):
                        out = gate.run(hook_input("make the search endpoint somehow faster"))
            decide.assert_not_called()
            self.assertIn("Which endpoint?", out["hookSpecificOutput"]["additionalContext"])
        finally:
            os.environ.pop("PROMPTIMIZE_MODE", None)

    def test_haiku_mode_clear_prompt_is_silent(self):
        os.environ["PROMPTIMIZE_MODE"] = "haiku"
        try:
            with mock.patch("haiku_gate.call_gate") as call_gate:
                call_gate.return_value = ({"can_execute": True, "clarifying_question": None, "ambiguity_type": None},
                                          {"input_tokens": 1000, "output_tokens": 11})
                with mock.patch("budget_ledger.BudgetLedger"):
                    self.assertIsNone(gate.run(hook_input("Add an index on created_at to the orders table")))
        finally:
            os.environ.pop("PROMPTIMIZE_MODE", None)

    def test_disabled_env_bypasses_gate(self):
        os.environ["PROMPTIMIZE_DISABLED"] = "1"
        with mock.patch.object(gate, "decide") as decide:
            self.assertIsNone(gate.run(hook_input("make the search endpoint somehow faster")))
            decide.assert_not_called()


class DeciderTests(unittest.TestCase):
    """The trained T5 head decider (default when plugin/gate_head.json is
    present) and the decider-selection / blind-spot logging around it."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.dir.name, "usage.jsonl")
        self._old_log = gate.USAGE_LOG
        self._old_head, self._old_head_path = gate._head, gate._head_path
        gate.USAGE_LOG = self.log
        for var in ("PROMPTIMIZE_DISABLED", "PROMPTIMIZE_THRESHOLD",
                    "PROMPTIMIZE_DECIDER", "PROMPTIMIZE_GATE_HEAD"):
            os.environ.pop(var, None)

    def tearDown(self):
        gate.USAGE_LOG = self._old_log
        gate._head, gate._head_path = self._old_head, self._old_head_path
        for var in ("PROMPTIMIZE_DECIDER", "PROMPTIMIZE_GATE_HEAD"):
            os.environ.pop(var, None)
        self.dir.cleanup()

    def write_head(self, w, b, mu, sd, thr=0.5):
        path = os.path.join(self.dir.name, "head.json")
        with open(path, "w") as f:
            json.dump({"format": "promptimize-gate-head/v1", "w": w, "b": b,
                       "mu": mu, "sd": sd, "threshold": thr}, f)
        return path

    def test_auto_decider_picks_head_when_artifact_present(self):
        # plugin/gate_head.json ships in the repo, so auto resolves to head.
        self.assertEqual(gate.decider_name(), "head")

    def test_auto_decider_falls_back_to_noul_without_artifact(self):
        os.environ["PROMPTIMIZE_GATE_HEAD"] = os.path.join(self.dir.name, "nope.json")
        self.assertEqual(gate.decider_name(), "noul")

    def test_decider_env_override_wins(self):
        os.environ["PROMPTIMIZE_DECIDER"] = "noul"
        self.assertEqual(gate.decider_name(), "noul")

    def test_default_threshold_is_per_decider(self):
        self.assertEqual(gate.default_threshold("head"), "0.95")
        self.assertEqual(gate.default_threshold("noul"), "0.35")

    def test_score_head_math(self):
        # emb standardized: ((1-0)/1)*2 + ((2-2)/1)*3 + ((4-3)/2)*(-1) = 1.5
        # z = 1.5 - 0.5 = 1.0 -> sigmoid(1) ~ 0.731059
        head = {"w": [2.0, 3.0, -1.0], "b": -0.5, "mu": [0.0, 2.0, 3.0], "sd": [1.0, 1.0, 2.0]}
        self.assertAlmostEqual(gate.score_head([1.0, 2.0, 4.0], head), 0.7310585786, places=6)

    def test_score_head_dim_mismatch_raises(self):
        head = {"w": [1.0], "b": 0.0, "mu": [0.0], "sd": [1.0]}
        with self.assertRaises(ValueError):
            gate.score_head([1.0, 2.0], head)

    def test_load_head_rejects_unknown_format(self):
        path = os.path.join(self.dir.name, "bad.json")
        with open(path, "w") as f:
            json.dump({"format": "other/v9", "w": [], "b": 0, "mu": [], "sd": []}, f)
        with self.assertRaises(ValueError):
            gate.load_head(path)

    def test_decide_head_uses_normalization_and_inverts(self):
        path = self.write_head([1.0], b=0.0, mu=[0.0], sd=[1.0], thr=0.5)
        os.environ["PROMPTIMIZE_DECIDER"] = "head"
        os.environ["PROMPTIMIZE_GATE_HEAD"] = path
        with mock.patch.object(gate, "embed_prompt", return_value=[2.0]) as emb:
            p_can = gate.decide("fix src/a.py")
        # score_head -> sigmoid(2) ~ 0.8808 p_ambiguous; P(can_execute) = 1 - that
        self.assertAlmostEqual(p_can, 1.0 - 0.8807970780, places=6)
        emb.assert_called_once_with("fix src/a.py")

    def test_head_decider_threshold_default_095(self):
        os.environ["PROMPTIMIZE_DECIDER"] = "head"
        with mock.patch.object(gate, "decide", return_value=0.93):
            with mock.patch.object(gate, "write_question",
                                   return_value=("Which endpoint?", {"input_tokens": 1, "output_tokens": 1})):
                with mock.patch("budget_ledger.BudgetLedger"):
                    out = gate.run(hook_input("make the search endpoint somehow faster"))
        self.assertIsNotNone(out)  # 0.93 < 0.95 -> asks under the head posture

    def test_head_decider_clear_prompt_silent(self):
        os.environ["PROMPTIMIZE_DECIDER"] = "head"
        with mock.patch.object(gate, "decide", return_value=0.97):
            self.assertIsNone(gate.run(hook_input("Add an index on created_at to the orders table")))

    def test_blind_spot_flagged_in_decision_log(self):
        long_prompt = "make this endpoint faster " + "with more detail " * 20
        self.assertGreaterEqual(len(long_prompt), 200)
        with mock.patch.object(gate, "decide", return_value=0.9):
            gate.run(hook_input(long_prompt))
        with open(self.log) as f:
            entries = [json.loads(l) for l in f if '"decision"' in l]
        self.assertTrue(entries[0]["blind_spot"])
        self.assertEqual(entries[0]["decider"], "head")

    def test_short_prompt_not_in_blind_spot(self):
        with mock.patch.object(gate, "decide", return_value=0.9):
            gate.run(hook_input("make the search endpoint somehow faster"))
        with open(self.log) as f:
            entries = [json.loads(l) for l in f if '"decision"' in l]
        self.assertFalse(entries[0]["blind_spot"])


if __name__ == "__main__":
    unittest.main()
