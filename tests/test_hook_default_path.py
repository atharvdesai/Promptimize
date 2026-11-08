import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import promptimize_gate as gate


def hook_input(prompt):
    return json.dumps({
        "session_id": "s1",
        "hook_event_name": "UserPromptSubmit",
        "cwd": "/tmp",
        "prompt": prompt,
    })


class HookDefaultPathTests(unittest.TestCase):
    """End-to-end through the hook's DEFAULT decider resolution (no
    PROMPTIMIZE_DECIDER override, real shipped gate_head.json), with only
    the ONNX embedding mocked. Guards against threshold/polarity
    inversion between the evaluator convention (ask iff p_ambiguous >=
    0.05) and the hook rule (P(can_execute) vs threshold)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self._old_log = gate.USAGE_LOG
        self._old_head, self._old_head_path = gate._head, gate._head_path
        gate.USAGE_LOG = os.path.join(self.dir.name, "usage.jsonl")
        for var in ("PROMPTIMIZE_DISABLED", "PROMPTIMIZE_THRESHOLD",
                    "PROMPTIMIZE_DECIDER", "PROMPTIMIZE_GATE_HEAD"):
            os.environ.pop(var, None)
        self.head = gate.load_head()  # the real shipped artifact

    def tearDown(self):
        gate.USAGE_LOG = self._old_log
        gate._head, gate._head_path = self._old_head, self._old_head_path
        self.dir.cleanup()

    def craft_emb(self, p_amb_target):
        """An embedding whose head score is exactly p_amb_target:
        standardized emb = t * w, so z = t * (w.w) + b = logit(p)."""
        w, b, mu, sd = self.head["w"], self.head["b"], self.head["mu"], self.head["sd"]
        ww = sum(x * x for x in w)
        logit = math.log(p_amb_target / (1.0 - p_amb_target))
        t = (logit - b) / ww
        return [m + t * wi * s for m, wi, s in zip(mu, w, sd)]

    def run_hook(self, prompt):
        with mock.patch.object(gate, "write_question",
                               return_value=("Which endpoint?", {"input_tokens": 1, "output_tokens": 1})):
            with mock.patch("budget_ledger.BudgetLedger"):
                return gate.run(hook_input(prompt))

    def test_default_decider_is_head_with_shipped_artifact(self):
        self.assertEqual(gate.decider_name(), "head")

    def test_polarity_clear_prompt_passes(self):
        # p_ambiguous 0.01 -> P(can_execute) 0.99 -> above 0.95 -> silent
        with mock.patch.object(gate, "embed_prompt", return_value=self.craft_emb(0.01)):
            self.assertIsNone(self.run_hook("Add an index on created_at to the orders table"))

    def test_polarity_ambiguous_prompt_asks(self):
        # p_ambiguous 0.9 -> P(can_execute) 0.1 -> below 0.95 -> nudge
        with mock.patch.object(gate, "embed_prompt", return_value=self.craft_emb(0.9)):
            out = self.run_hook("make the search endpoint somehow faster")
        self.assertIsNotNone(out)
        self.assertIn("Which endpoint?", out["hookSpecificOutput"]["additionalContext"])

    def test_boundary_matches_evaluator_convention(self):
        # Evaluator: ask iff p_ambiguous >= 0.05, i.e. P(can_execute) <= 0.95.
        # At EXACTLY the threshold the hook must ask (metrics() uses >=).
        with mock.patch.object(gate, "decide", return_value=0.95):
            out = self.run_hook("make the search endpoint somehow faster")
        self.assertIsNotNone(out)
        # And just above it, silent.
        with mock.patch.object(gate, "decide", return_value=0.9500001):
            self.assertIsNone(self.run_hook("make the search endpoint somehow faster"))

    def test_val_replay_through_hook_rule_matches_recorded_counts(self):
        """Replay the frozen val split through the HOOK's decision rule
        (not a standalone scorer) and require the recorded D1-errata
        counts: 43/44, one false-ask, zero misses. Any threshold or
        polarity inversion in the shipped default path breaks this."""
        emb_path = os.path.join(os.path.dirname(__file__), "..",
                                "experiments", "t5_embeddings.jsonl")
        if not os.path.exists(emb_path):
            self.skipTest("t5_embeddings.jsonl not present")
        rows = [json.loads(l) for l in open(emb_path)]
        val = [r for r in rows if r["split"] == "val"]
        self.assertEqual(len(val), 44)
        thr = float(gate.default_threshold(gate.decider_name()))
        self.assertEqual(thr, 0.95)
        correct = fp = fn = 0
        for r in val:
            p_can = 1.0 - gate.score_head(r["emb"], self.head)
            asks = p_can <= thr  # the hook rule for the head decider
            truth = bool(r["label"])  # 1 = ambiguous = should ask
            if asks == truth:
                correct += 1
            elif asks:
                fp += 1
            else:
                fn += 1
        self.assertEqual((correct, fp, fn), (43, 1, 0))
        self.assertEqual(round(fp / 21, 3), 0.048)  # per-class: 21 executable rows


if __name__ == "__main__":
    unittest.main()
