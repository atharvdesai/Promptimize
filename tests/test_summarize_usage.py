import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import summarize_usage as su


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "usage.jsonl")

    def write(self, entries, raw_lines=()):
        with open(self.path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            for l in raw_lines:
                f.write(l + "\n")

    def test_missing_file_exits_zero(self):
        with mock.patch("sys.argv", ["summarize_usage.py", self.path]):
            with mock.patch("sys.stdout"):
                with self.assertRaises(SystemExit) as cm:
                    su.main()
        self.assertEqual(cm.exception.code, 0)

    def test_decision_aggregates(self):
        self.write([
            {"ts": 1000, "kind": "decision", "decider": "head", "via": "server",
             "prob": 0.99, "threshold": 0.95, "latency_ms": 100, "blind_spot": False},
            {"ts": 2000, "kind": "decision", "decider": "head", "via": "local",
             "prob": 0.90, "threshold": 0.95, "latency_ms": 300, "blind_spot": True},
            {"ts": 3000, "kind": "decision", "decider": "noul", "via": "local",
             "prob": 0.35, "threshold": 0.35, "latency_ms": 200, "blind_spot": False},
        ])
        s = su.summarize(self.path)
        # head: prob <= thr asks (0.90 <= 0.95 asks, 0.99 doesn't);
        # noul: strict < (0.35 < 0.35 does NOT ask)
        self.assertEqual(s["decisions"]["n"], 3)
        self.assertAlmostEqual(s["decisions"]["ask_rate"], round(1 / 3, 3))
        self.assertEqual(s["decisions"]["via"], {"server": 1, "local": 2})
        self.assertEqual(s["decisions"]["latency_ms"]["p50"], 200)
        self.assertEqual(s["decisions"]["latency_ms"]["max"], 300)
        self.assertAlmostEqual(s["decisions"]["blind_spot_share"], round(1 / 3, 3))

    def test_question_pipeline_and_cost(self):
        self.write([
            {"ts": 1, "kind": "question", "input_tokens": 1000, "output_tokens": 100,
             "cache_read_tokens": 0, "cache_write_tokens": 0},
            {"ts": 2, "kind": "question_declined"},
            {"ts": 3, "kind": "question_skipped", "reason": "budget_exhausted"},
            {"ts": 4, "kind": "question_failed", "error": "RateLimit"},
            {"ts": 5, "kind": "error", "error": "RateLimit"},
        ])
        s = su.summarize(self.path)
        q = s["questions"]
        self.assertEqual((q["drafted"], q["declined_by_haiku"],
                          q["skipped_budget"], q["failed"]), (1, 1, 1, 1))
        self.assertEqual(q["tokens"]["input_tokens"], 1000)
        self.assertGreater(q["est_cost_usd"], 0)
        self.assertEqual(s["errors"], {"RateLimit": 1})

    def test_malformed_lines_counted(self):
        self.write([{"ts": 1, "kind": "decision", "decider": "head",
                     "prob": 0.5, "threshold": 0.95}],
                   raw_lines=["{not json", ""])
        s = su.summarize(self.path)
        self.assertEqual(s["malformed_lines"], 1)
        self.assertEqual(s["events"], 1)

    def test_privacy_violation_surfaced(self):
        self.write([{"ts": 1, "kind": "decision", "prompt": "secret text"}])
        s = su.summarize(self.path)
        self.assertEqual(s["privacy"]["prompt_field_violations"], 1)
        text = su.render(s)
        self.assertIn("PRIVACY REGRESSION", text)
        self.assertNotIn("secret text", text)

    def test_unknown_fields_flagged(self):
        self.write([{"ts": 1, "kind": "decision", "brand_new_field": 1}])
        s = su.summarize(self.path)
        self.assertEqual(s["privacy"]["unknown_fields"], {"brand_new_field": 1})

    def test_per_day_bucketing(self):
        day1 = 1700000000  # fixed epoch
        self.write([
            {"ts": day1, "kind": "decision", "decider": "head", "prob": 0.5,
             "threshold": 0.95},
            {"ts": day1 + 90000, "kind": "decision", "decider": "head",
             "prob": 0.99, "threshold": 0.95},
            {"ts": day1, "kind": "question"},
        ])
        s = su.summarize(self.path)
        days = sorted(s["per_day"])
        self.assertEqual(len(days), 2)
        import time
        d1 = time.strftime("%Y-%m-%d", time.localtime(day1))
        self.assertEqual(s["per_day"][d1], {"decisions": 1, "asks": 1, "questions": 1})

    def test_empty_log(self):
        self.write([])
        s = su.summarize(self.path)
        self.assertEqual(s["decisions"]["n"], 0)
        self.assertIsNone(s["decisions"]["ask_rate"])
        self.assertIn("events: 0", su.render(s))

    def test_asked_conventions(self):
        self.assertTrue(su.asked({"decider": "head", "prob": 0.95, "threshold": 0.95}))
        self.assertFalse(su.asked({"decider": "noul", "prob": 0.35, "threshold": 0.35}))
        self.assertIsNone(su.asked({"decider": "head"}))


if __name__ == "__main__":
    unittest.main()
