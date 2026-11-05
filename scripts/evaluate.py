#!/usr/bin/env python3
"""Score clarifying-question predictions against a frozen split.

Usage: python3 scripts/evaluate.py predictions.jsonl [--gold dataset/splits/test.jsonl]

Predictions are JSONL, one object per prompt:

    {"prompt": "make the api public", "can_execute": false,
     "clarifying_question": "Which API, and who should see it?",
     "ambiguity_type": "missing_scope"}

Required fields: "prompt" (matched to gold after whitespace/case
normalization) and "can_execute" (JSON true/false). Optional:
"clarifying_question" and "ambiguity_type".

Metrics follow the repo README roadmap: over-asking rate (interrupting an
executable prompt, the costly error), under-asking rate (letting an
ambiguous prompt through), and a token-overlap F1 between predicted and
gold questions as a rough quality proxy. Question quality still needs
human or LLM-judge review; exact match is not meaningful.

A gold prompt with no prediction counts as "predicted executable" - the
realistic behavior of a pre-flight gate that stays silent is to let the
prompt through - so missing predictions raise the under-asking rate.
Predictions whose prompt matches no gold row are reported as extras and
ignored. Exits non-zero on malformed input.
"""
import argparse
import json
import re
import sys
from collections import Counter

WORD = re.compile(r"[a-z0-9]+")
DEFAULT_GOLD = "dataset/splits/test.jsonl"


def norm(prompt):
    return " ".join(str(prompt).split()).lower()


def token_f1(a, b):
    """Multiset token-overlap F1 between two question strings."""
    ta, tb = WORD.findall(a.lower()), WORD.findall(b.lower())
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(ta), common / len(tb)
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError("%s line %d: invalid JSON: %s" % (path, n, e))
    return rows


def validate_prediction(row, where):
    if not isinstance(row, dict):
        raise ValueError("%s: not a JSON object" % where)
    if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
        raise ValueError("%s: missing or empty 'prompt'" % where)
    if not isinstance(row.get("can_execute"), bool):
        raise ValueError("%s: 'can_execute' must be true or false" % where)
    q = row.get("clarifying_question")
    if q is not None and not isinstance(q, str):
        raise ValueError("%s: 'clarifying_question' must be a string or null" % where)
    t = row.get("ambiguity_type")
    if t is not None and not isinstance(t, str):
        raise ValueError("%s: 'ambiguity_type' must be a string or null" % where)


def evaluate(gold_rows, pred_rows):
    """Return a metrics dict comparing predictions to gold rows."""
    for n, row in enumerate(pred_rows, 1):
        validate_prediction(row, "prediction %d" % n)
    preds = {}
    for row in pred_rows:
        preds[norm(row["prompt"])] = row

    gold_keys = [norm(r["prompt"]) for r in gold_rows]
    missing = [r["prompt"] for r, k in zip(gold_rows, gold_keys) if k not in preds]
    extras = [r["prompt"] for r in pred_rows if norm(r["prompt"]) not in set(gold_keys)]

    tp = fp = fn = tn = 0          # positive class = ambiguous (can_execute false)
    coverage_n = coverage_hit = 0  # asked a real question when one was needed
    f1_sum = f1_n = 0
    type_hit = type_n = 0
    per_type = {}
    for row in gold_rows:
        pred = preds.get(norm(row["prompt"])) or {"can_execute": True}
        gold_amb, pred_amb = not row["can_execute"], not pred["can_execute"]
        if gold_amb and pred_amb:
            tp += 1
        elif not gold_amb and pred_amb:
            fp += 1
        elif gold_amb and not pred_amb:
            fn += 1
        else:
            tn += 1
        if gold_amb:
            t = row.get("ambiguity_type") or "unknown"
            stats = per_type.setdefault(t, [0, 0])
            stats[0] += 1
            stats[1] += 1 if pred_amb else 0
            if pred_amb:
                coverage_n += 1
                q = pred.get("clarifying_question")
                if isinstance(q, str) and "?" in q:
                    coverage_hit += 1
                    if row.get("clarifying_question"):
                        f1_sum += token_f1(q, row["clarifying_question"])
                        f1_n += 1
                pt = pred.get("ambiguity_type")
                if pt is not None:
                    type_n += 1
                    type_hit += 1 if pt == row.get("ambiguity_type") else 0

    n = len(gold_rows)
    gold_amb_n = tp + fn
    gold_exe_n = tn + fp
    majority = max(gold_amb_n, gold_exe_n) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "majority_baseline": majority,
        "over_asking_rate": fp / gold_exe_n if gold_exe_n else 0.0,
        "under_asking_rate": fn / gold_amb_n if gold_amb_n else 0.0,
        "ambiguous_precision": prec,
        "ambiguous_recall": rec,
        "ambiguous_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "question_coverage": coverage_hit / coverage_n if coverage_n else 0.0,
        "question_coverage_hit": coverage_hit,
        "question_coverage_n": coverage_n,
        "question_token_f1": f1_sum / f1_n if f1_n else None,
        "question_token_f1_n": f1_n,
        "type_accuracy": type_hit / type_n if type_n else None,
        "type_accuracy_n": type_n,
        "per_type": {t: {"n": s[0], "detected": s[1]}
                     for t, s in sorted(per_type.items())},
        "missing_predictions": missing,
        "extra_predictions": extras,
    }


def print_report(result, gold_path, out=sys.stdout):
    c = result["confusion"]
    w = lambda *a: print(*a, file=out)
    w("gold: %s (%d rows, %d missing predictions counted as executable)"
      % (gold_path, result["n"], len(result["missing_predictions"])))
    if result["extra_predictions"]:
        w("warning: %d predictions match no gold row (ignored): %s"
          % (len(result["extra_predictions"]),
             ", ".join(repr(p[:60]) for p in result["extra_predictions"][:5])))
    w("")
    w("accuracy:            %.3f   (majority-class baseline %.3f)"
      % (result["accuracy"], result["majority_baseline"]))
    w("over-asking rate:    %.3f   (%d of %d executable prompts interrupted)"
      % (result["over_asking_rate"], c["fp"], c["tn"] + c["fp"]))
    w("under-asking rate:   %.3f   (%d of %d ambiguous prompts let through)"
      % (result["under_asking_rate"], c["fn"], c["tp"] + c["fn"]))
    w("ambiguous class:     precision %.3f  recall %.3f  f1 %.3f"
      % (result["ambiguous_precision"], result["ambiguous_recall"],
         result["ambiguous_f1"]))
    w("question coverage:   %.3f   (%d of %d detected ambiguities got a '?')"
      % (result["question_coverage"], result["question_coverage_hit"],
         result["question_coverage_n"]))
    if result["question_token_f1"] is not None:
        w("question token-f1:   %.3f   (n=%d; rough proxy only - real question"
          " quality needs human or LLM-judge review)"
          % (result["question_token_f1"], result["question_token_f1_n"]))
    if result["type_accuracy"] is not None:
        w("ambiguity-type acc:  %.3f   (n=%d, where detected and typed)"
          % (result["type_accuracy"], result["type_accuracy_n"]))
    if result["per_type"]:
        w("")
        w("detection by ambiguity_type:")
        for t, s in result["per_type"].items():
            w("  %-30s %d/%d detected" % (t, s["detected"], s["n"]))
    if result["missing_predictions"]:
        w("")
        w("missing predictions (counted as let-through):")
        for p in result["missing_predictions"][:10]:
            w("  %s" % p[:80])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions", help="JSONL of model predictions")
    ap.add_argument("--gold", default=DEFAULT_GOLD,
                    help="gold JSONL split (default: %(default)s)")
    args = ap.parse_args(argv)
    try:
        gold = load_jsonl(args.gold)
        preds = load_jsonl(args.predictions)
        result = evaluate(gold, preds)
    except (OSError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if not gold:
        print("error: no gold rows in %s" % args.gold, file=sys.stderr)
        return 1
    print_report(result, args.gold, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
