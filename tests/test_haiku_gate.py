import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import haiku_gate
from budget_ledger import BudgetLedger
from haiku_gate import GateCallError, call_gate


class HaikuGateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        ledger_path = os.path.join(self.dir, "budget.json")
        self.ledger = BudgetLedger(ledger_path, cap_usd=2.0).provision("test")
        self.key_file = os.path.join(self.dir, "key")
        with open(self.key_file, "w") as f:
            f.write("sk-ant-test-not-a-real-key")
        os.chmod(self.key_file, 0o600)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fake_post(self, url, body, key, timeout):
        if url == haiku_gate.COUNT_URL:
            return {"input_tokens": 1111}
        return {
            "content": [{"type": "text", "text": '{"can_execute": true}'}],
            "usage": {"input_tokens": 1100, "output_tokens": 12,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    def test_reserve_uses_exact_provider_count(self):
        with mock.patch.object(haiku_gate, "_post_json", side_effect=self._fake_post):
            result, usage = call_gate("do the thing", self.ledger, "test",
                                      key_file=self.key_file)
        self.assertTrue(result["can_execute"])
        # settle replaced the reservation with actual usage; the settle event
        # shows actuals, and no reservation lingers
        self.assertEqual(self.ledger._state["reserved"], {})
        settle = [e for e in self.ledger._state["events"] if e["kind"] == "settle"][0]
        self.assertEqual(settle["input_tokens"], 1100)

    def test_count_unavailable_refuses_before_any_paid_call(self):
        def boom(url, body, key, timeout):
            raise OSError("no network")

        with mock.patch.object(haiku_gate, "_post_json", side_effect=boom):
            with self.assertRaises(GateCallError):
                call_gate("do the thing", self.ledger, "test", key_file=self.key_file)
        # nothing reserved, nothing spent
        self.assertEqual(self.ledger.committed_usd(), 0.0)

    def test_group_readable_key_file_refused(self):
        os.chmod(self.key_file, 0o644)
        with self.assertRaises(GateCallError) as ctx:
            haiku_gate._read_key(self.key_file)
        self.assertIn("chmod 600", str(ctx.exception))

    def test_key_command_stdout_used(self):
        with mock.patch.dict(os.environ,
                             {"PROMPTIMIZE_ANTHROPIC_KEY_CMD": "printf sk-ant-from-cmd"}):
            self.assertEqual(haiku_gate._read_key(), "sk-ant-from-cmd")


if __name__ == "__main__":
    unittest.main()
