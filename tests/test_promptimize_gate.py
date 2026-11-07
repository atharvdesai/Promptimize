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

    def tearDown(self):
        gate.USAGE_LOG = self._old_log
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


if __name__ == "__main__":
    unittest.main()
