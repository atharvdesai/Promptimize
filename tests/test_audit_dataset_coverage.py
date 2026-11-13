import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import audit_dataset_coverage as ac


def row(prompt, can_execute, ambiguity_type=None):
    return {"prompt": prompt, "can_execute": can_execute,
            "ambiguity_type": ambiguity_type, "ambiguities": [],
            "missing_information": [], "clarifying_question": None,
            "confidence": 0.9}


class BucketTests(unittest.TestCase):
    def test_bucket_edges(self):
        self.assertEqual(ac.bucket_of(0), "0-49")
        self.assertEqual(ac.bucket_of(49), "0-49")
        self.assertEqual(ac.bucket_of(50), "50-99")
        self.assertEqual(ac.bucket_of(199), "150-199")
        self.assertEqual(ac.bucket_of(200), "200-299")
        self.assertEqual(ac.bucket_of(499), "300-499")
        self.assertEqual(ac.bucket_of(500), "500-999")
        self.assertEqual(ac.bucket_of(1000), "1000+")
        self.assertEqual(ac.bucket_of(5000), "1000+")

    def test_buckets_cover_full_range(self):
        self.assertEqual(ac.BUCKETS[0], "0-49")
        self.assertEqual(ac.BUCKETS[-1], "1000+")
        self.assertEqual(len(ac.BUCKETS), 8)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "data.jsonl")

    def write(self, rows, raw_lines=()):
        with open(self.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            for line in raw_lines:
                f.write(line + "\n")

    def test_histogram_and_long_tail(self):
        rows = ([row("x" * 30, True), row("x" * 250, True)]
                + [row("x" * 29, False), row("x" * 45, False, "missing_scope")])
        self.write(rows)
        a = ac.audit_dataset(self.path)
        self.assertEqual(a["rows"], 4)
        self.assertEqual(a["histogram"]["executable"]["0-49"], 1)
        self.assertEqual(a["histogram"]["executable"]["200-299"], 1)
        self.assertEqual(a["histogram"]["ambiguous"]["0-49"], 2)
        self.assertEqual(a["long_tail"]["executable"]["200"], 1)
        self.assertEqual(a["long_tail"]["ambiguous"]["200"], 0)
        self.assertEqual(a["long_tail"]["ambiguous"]["max"], 45)

    def test_asymmetric_gap_detection(self):
        rows = [row("x" * 250, True), row("x" * 30, False)]
        self.write(rows)
        a = ac.audit_dataset(self.path)
        gaps = a["gaps"]["asymmetric"]
        # both directions flag: covered-by-one, empty-in-the-other
        self.assertIn({"bucket": "200-299", "executable": 1, "ambiguous": 0}, gaps)
        self.assertIn({"bucket": "0-49", "executable": 0, "ambiguous": 1}, gaps)
        # a bucket covered by both is not a gap
        rows = [row("x" * 30, True), row("x" * 30, False)]
        self.write(rows)
        a = ac.audit_dataset(self.path)
        self.assertNotIn({"bucket": "0-49", "executable": 1, "ambiguous": 1},
                         a["gaps"]["asymmetric"])

    def test_unmeasured_bucket(self):
        rows = [row("x" * 30, True), row("x" * 30, False)]
        self.write(rows)
        a = ac.audit_dataset(self.path)
        self.assertIn("200-299", a["gaps"]["unmeasured"])

    def test_type_by_bucket(self):
        rows = [row("x" * 30, False, "missing_scope"),
                row("x" * 40, False, "missing_scope"),
                row("x" * 210, False, "missing_file"),
                row("x" * 100, True)]
        self.write(rows)
        a = ac.audit_dataset(self.path)
        table = a["ambiguity_type_by_bucket"]
        self.assertEqual(table["missing_scope"]["0-49"], 2)
        self.assertEqual(table["missing_file"]["200-299"], 1)
        self.assertNotIn("0-49", table["missing_file"])

    def test_malformed_lines_counted_not_fatal(self):
        self.write([row("ok", True)], raw_lines=["{not json", '{"prompt": 3, "can_execute": true}'])
        a = ac.audit_dataset(self.path)
        self.assertEqual(a["rows"], 1)
        self.assertEqual(a["malformed"], 2)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "data.jsonl")

    def write(self, rows):
        with open(self.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_gate_fails_below_minimum(self):
        self.write([row("x" * 30, False)] * 5 + [row("x" * 250, False)] * 3)
        rc = ac.main([self.path, "--min-long-ambiguous", "4"])
        self.assertEqual(rc, 1)

    def test_gate_passes_at_minimum(self):
        self.write([row("x" * 30, False)] * 5 + [row("x" * 250, False)] * 4)
        rc = ac.main([self.path, "--min-long-ambiguous", "4"])
        self.assertEqual(rc, 0)

    def test_gate_respects_long_threshold(self):
        self.write([row("x" * 350, False)] * 2)
        rc = ac.main([self.path, "--min-long-ambiguous", "3", "--long-threshold", "300"])
        self.assertEqual(rc, 1)
        rc = ac.main([self.path, "--min-long-ambiguous", "2", "--long-threshold", "300"])
        self.assertEqual(rc, 0)

    def test_missing_file_exits_1(self):
        rc = ac.main([os.path.join(self.dir.name, "nope.jsonl")])
        self.assertEqual(rc, 1)

    def test_json_output_shape(self):
        self.write([row("x" * 30, True), row("x" * 30, False, "missing_scope")])
        a = ac.audit_dataset(self.path)
        blob = json.dumps(a)  # must round-trip for --json consumers
        back = json.loads(blob)
        self.assertEqual(set(back), {"path", "rows", "malformed", "histogram",
                                     "long_tail", "ambiguity_type_by_bucket", "gaps"})


if __name__ == "__main__":
    unittest.main()
