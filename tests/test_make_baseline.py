"""Contract tests for scripts/make_baseline.py."""
import json
import subprocess
import sys

import pytest

sys.path.insert(0, "scripts")
import evaluate  # noqa: E402
import make_baseline  # noqa: E402

GOLD = "dataset/splits/test.jsonl"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "scripts/make_baseline.py", *args],
        capture_output=True, text=True)


@pytest.fixture(scope="module")
def gold_rows():
    return make_baseline.load_jsonl(GOLD)


def scored(strategy, gold_rows):
    rows = [make_baseline.baseline_row(r, strategy) for r in gold_rows]
    return evaluate.evaluate(gold_rows, rows)


def test_strategies_cover_every_gold_prompt(gold_rows):
    for strategy in ("never-ask", "always-ask", "oracle"):
        rows = [make_baseline.baseline_row(r, strategy) for r in gold_rows]
        assert len(rows) == len(gold_rows)
        for row in rows:
            evaluate.validate_prediction(row, strategy)


def test_oracle_scores_perfectly(gold_rows):
    r = scored("oracle", gold_rows)
    assert r["accuracy"] == 1.0
    assert r["over_asking_rate"] == 0.0
    assert r["under_asking_rate"] == 0.0
    assert r["question_coverage"] == 1.0


def test_never_ask_is_pure_under_asking(gold_rows):
    r = scored("never-ask", gold_rows)
    assert r["over_asking_rate"] == 0.0
    assert r["under_asking_rate"] == 1.0
    executable_share = sum(1 for g in gold_rows if g["can_execute"]) / len(gold_rows)
    assert r["accuracy"] == pytest.approx(executable_share)


def test_always_ask_is_pure_over_asking(gold_rows):
    r = scored("always-ask", gold_rows)
    assert r["over_asking_rate"] == 1.0
    assert r["under_asking_rate"] == 0.0
    assert r["question_coverage"] == 1.0


def test_cli_writes_file_and_stdout(tmp_path):
    out = tmp_path / "preds.jsonl"
    r = run_cli("never-ask", "--gold", GOLD, "-o", str(out))
    assert r.returncode == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows and all(row["can_execute"] is True for row in rows)

    r = run_cli("always-ask", "--gold", GOLD)
    assert r.returncode == 0
    rows = [json.loads(l) for l in r.stdout.splitlines()]
    assert rows and all(row["can_execute"] is False for row in rows)


def test_cli_rejects_unknown_strategy_and_bad_gold(tmp_path):
    r = run_cli("sometimes-ask", "--gold", GOLD)
    assert r.returncode != 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    r = run_cli("never-ask", "--gold", str(bad))
    assert r.returncode == 2
