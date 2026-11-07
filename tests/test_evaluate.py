"""Contract tests for scripts/evaluate.py (black-box, stdlib only)."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", MODULE_PATH)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def gold_executable(prompt="run the tests"):
    return {"prompt": prompt, "can_execute": True, "confidence": 0.95,
            "ambiguity_type": None, "ambiguities": [], "missing_information": [],
            "clarifying_question": None}


def gold_ambiguous(prompt="make it faster", ambiguity_type="unclear_goal",
                   question="What is slow, and how fast should it be?"):
    return {"prompt": prompt, "can_execute": False, "confidence": 0.9,
            "ambiguity_type": ambiguity_type, "ambiguities": ["no target"],
            "missing_information": ["which part"], "clarifying_question": question}


def sample_gold():
    return [gold_executable("run the tests"),
            gold_executable("rename the helper function"),
            gold_ambiguous("make it faster", "unclear_goal"),
            gold_ambiguous("add auth", "missing_requirement",
                           "Which auth provider should protect which routes?")]


def pred(prompt, can_execute, question=None, ambiguity_type=None):
    row = {"prompt": prompt, "can_execute": can_execute,
           "clarifying_question": question}
    if ambiguity_type is not None:
        row["ambiguity_type"] = ambiguity_type
    return row


def perfect_predictions():
    return [pred("run the tests", True),
            pred("rename the helper function", True),
            pred("make it faster", False, "What is slow, and how fast should it be?"),
            pred("add auth", False, "Which auth provider should protect which routes?")]


class TokenF1Tests(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertAlmostEqual(evaluate.token_f1("Which API is slow?",
                                                 "which api is slow"), 1.0)

    def test_disjoint_strings_score_zero(self):
        self.assertEqual(evaluate.token_f1("what is slow?", "who owns auth?"), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        f1 = evaluate.token_f1("which api is slow", "which api should change")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_empty_string_scores_zero(self):
        self.assertEqual(evaluate.token_f1("", "anything?"), 0.0)


class EvaluateMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        r = evaluate.evaluate(sample_gold(), perfect_predictions())
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["accuracy"], 1.0)
        self.assertAlmostEqual(r["over_asking_rate"], 0.0)
        self.assertAlmostEqual(r["under_asking_rate"], 0.0)
        self.assertAlmostEqual(r["question_coverage"], 1.0)
        self.assertAlmostEqual(r["question_token_f1"], 1.0)
        self.assertEqual(r["missing_predictions"], [])
        self.assertEqual(r["extra_predictions"], [])

    def test_always_execute_baseline(self):
        preds = [pred(g["prompt"], True) for g in sample_gold()]
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertAlmostEqual(r["accuracy"], 0.5)
        self.assertAlmostEqual(r["over_asking_rate"], 0.0)
        self.assertAlmostEqual(r["under_asking_rate"], 1.0)
        self.assertAlmostEqual(r["majority_baseline"], 0.5)

    def test_always_ask_flags_everything(self):
        preds = [pred(g["prompt"], False, "Can you clarify?") for g in sample_gold()]
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertAlmostEqual(r["over_asking_rate"], 1.0)
        self.assertAlmostEqual(r["under_asking_rate"], 0.0)
        self.assertAlmostEqual(r["question_coverage"], 1.0)

    def test_missing_prediction_counts_as_let_through(self):
        preds = perfect_predictions()[:3]  # drop the "add auth" prediction
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertEqual(r["missing_predictions"], ["add auth"])
        self.assertEqual(r["confusion"]["fn"], 1)
        self.assertAlmostEqual(r["under_asking_rate"], 0.5)
        self.assertAlmostEqual(r["accuracy"], 0.75)

    def test_extra_prediction_reported_and_ignored(self):
        preds = perfect_predictions() + [pred("unrelated prompt", True)]
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertEqual(r["extra_predictions"], ["unrelated prompt"])
        self.assertAlmostEqual(r["accuracy"], 1.0)

    def test_prompt_matching_normalizes_case_and_whitespace(self):
        preds = [pred("  Run The Tests ", True),
                 pred("rename the helper function", True),
                 pred("MAKE IT FASTER", False, "What is slow?"),
                 pred("add   auth", False, "Which auth provider?")]
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertAlmostEqual(r["accuracy"], 1.0)
        self.assertEqual(r["missing_predictions"], [])

    def test_question_without_question_mark_lowers_coverage(self):
        preds = perfect_predictions()
        preds[2]["clarifying_question"] = "tell me more"
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertAlmostEqual(r["question_coverage"], 0.5)
        self.assertEqual(r["question_token_f1_n"], 1)

    def test_ambiguity_type_accuracy_when_provided(self):
        preds = perfect_predictions()
        preds[2]["ambiguity_type"] = "unclear_goal"      # correct
        preds[3]["ambiguity_type"] = "missing_file"      # wrong
        r = evaluate.evaluate(sample_gold(), preds)
        self.assertAlmostEqual(r["type_accuracy"], 0.5)
        self.assertEqual(r["type_accuracy_n"], 2)

    def test_per_type_breakdown(self):
        r = evaluate.evaluate(sample_gold(), perfect_predictions())
        self.assertEqual(r["per_type"]["unclear_goal"], {"n": 1, "detected": 1})
        self.assertEqual(r["per_type"]["missing_requirement"], {"n": 1, "detected": 1})

    def test_malformed_prediction_raises(self):
        with self.assertRaises(ValueError):
            evaluate.evaluate(sample_gold(), [{"prompt": "run the tests"}])
        with self.assertRaises(ValueError):
            evaluate.evaluate(sample_gold(), [{"can_execute": True}])
        with self.assertRaises(ValueError):
            evaluate.evaluate(sample_gold(), [pred("x", "yes")])


class CliTests(unittest.TestCase):
    def _write_jsonl(self, rows):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.close()
        return fh.name

    def test_main_prints_report_and_returns_zero(self):
        gold_path = self._write_jsonl(sample_gold())
        pred_path = self._write_jsonl(perfect_predictions())
        out = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(out):
            code = evaluate.main([pred_path, "--gold", gold_path])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("accuracy:            1.000", text)
        self.assertIn("over-asking rate:    0.000", text)
        self.assertIn("under-asking rate:   0.000", text)
        self.assertIn("detection by ambiguity_type:", text)

    def test_main_reports_missing_and_extras(self):
        gold_path = self._write_jsonl(sample_gold())
        pred_path = self._write_jsonl(perfect_predictions()[:3]
                                      + [pred("stray", True)])
        out = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(out):
            code = evaluate.main([pred_path, "--gold", gold_path])
        self.assertEqual(code, 0)
        self.assertIn("1 missing predictions", out.getvalue())
        self.assertIn("warning: 1 predictions match no gold row", out.getvalue())

    def test_json_flag_emits_parseable_metrics(self):
        gold_path = self._write_jsonl(sample_gold())
        pred_path = self._write_jsonl(perfect_predictions())
        out = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(out):
            code = evaluate.main([pred_path, "--gold", gold_path, "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["accuracy"], 1.0)
        self.assertEqual(data["over_asking_rate"], 0.0)
        self.assertEqual(data["under_asking_rate"], 0.0)
        self.assertEqual(data["confusion"], {"tp": 2, "fp": 0, "fn": 0, "tn": 2})
        self.assertIn("per_type", data)
        self.assertIn("missing_predictions", data)

    def test_json_flag_reflects_errors_like_text_report(self):
        gold_path = self._write_jsonl(sample_gold())
        pred_path = self._write_jsonl(perfect_predictions()[:3]
                                      + [pred("stray", True)])
        out = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(out):
            code = evaluate.main([pred_path, "--gold", gold_path, "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["missing_predictions"], ["add auth"])
        self.assertEqual(data["extra_predictions"], ["stray"])
        self.assertEqual(data["under_asking_rate"], 0.5)

    def test_main_rejects_malformed_json(self):
        gold_path = self._write_jsonl(sample_gold())
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        fh.write("not json\n")
        fh.close()
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = evaluate.main([fh.name, "--gold", gold_path])
        self.assertEqual(code, 1)
        self.assertIn("invalid JSON", err.getvalue())


if __name__ == "__main__":
    unittest.main()
