#!/usr/bin/env python3
"""Validate the clarifying-question dataset and print summary stats.

Usage: python scripts/validate_dataset.py [path/to/file.jsonl]
Exits non-zero if any row violates the schema or labeling rules.
"""
import json
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "dataset/clarifying_question_dataset.jsonl"

FIELDS = {"prompt", "can_execute", "confidence", "ambiguity_type",
          "ambiguities", "missing_information", "clarifying_question"}
TYPES = {"missing_scope", "missing_requirement", "missing_file", "missing_component",
         "missing_technology", "missing_constraint", "missing_acceptance_criteria",
         "multiple_interpretations", "missing_context", "unclear_goal"}
GENERIC = ["can you provide more details", "can you clarify", "what do you mean"]


def check(row):
    assert set(row) == FIELDS, f"fields differ by {set(row) ^ FIELDS}"
    assert isinstance(row["prompt"], str) and row["prompt"].strip(), "empty prompt"
    assert 0.0 <= row["confidence"] <= 1.0, "confidence out of range"
    if row["can_execute"]:
        assert row["ambiguity_type"] is None, "executable row has ambiguity_type"
        assert row["ambiguities"] == [] and row["missing_information"] == [], \
            "executable row has ambiguity details"
        assert row["clarifying_question"] is None, "executable row has a question"
    else:
        assert row["ambiguity_type"] in TYPES, f"bad ambiguity_type {row['ambiguity_type']}"
        assert row["ambiguities"] and row["missing_information"], "missing details"
        q = row["clarifying_question"]
        assert q and "?" in q, "question missing or has no '?'"
        assert not any(g in q.lower() for g in GENERIC), "generic question"


def main():
    errors, seen, dupes = [], set(), []
    can, types = Counter(), Counter()
    for n, line in enumerate(open(PATH, encoding="utf-8"), 1):
        try:
            row = json.loads(line)
            check(row)
        except Exception as e:  # noqa: BLE001 - report every bad row
            errors.append((n, str(e)))
            continue
        key = row["prompt"].strip().lower()
        if key in seen:
            dupes.append(n)
        seen.add(key)
        can[row["can_execute"]] += 1
        if not row["can_execute"]:
            types[row["ambiguity_type"]] += 1

    print(f"rows: {sum(can.values())}  executable: {can[True]}  ambiguous: {can[False]}")
    for t, c in types.most_common():
        print(f"  {t}: {c}")
    for n, msg in errors:
        print(f"ERROR line {n}: {msg}")
    for n in dupes:
        print(f"ERROR line {n}: duplicate prompt")
    sys.exit(1 if errors or dupes else 0)


if __name__ == "__main__":
    main()
