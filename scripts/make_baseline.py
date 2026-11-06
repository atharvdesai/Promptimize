#!/usr/bin/env python3
"""Generate trivial baseline predictions for a frozen split.

Usage: python3 scripts/make_baseline.py STRATEGY [--gold dataset/splits/test.jsonl]
                                      [-o predictions.jsonl]

Strategies:
  never-ask   predict every prompt executable (can_execute true)
  always-ask  predict every prompt ambiguous, with a generic question
  oracle      copy the gold labels and questions - a sanity check that
              should score perfectly through scripts/evaluate.py

These anchor model scores: a real model should beat never-ask and
always-ask on accuracy and on the over/under-asking tradeoff, and the
oracle must hit 1.0 accuracy or the evaluator itself is broken.

Output is JSONL on stdout, or to -o PATH. Exits non-zero on an unknown
strategy or malformed gold file.
"""
import argparse
import json
import sys

GENERIC_QUESTION = "Could you clarify what you need?"


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


def baseline_row(gold, strategy):
    if strategy == "never-ask":
        return {"prompt": gold["prompt"], "can_execute": True}
    if strategy == "always-ask":
        return {"prompt": gold["prompt"], "can_execute": False,
                "clarifying_question": GENERIC_QUESTION}
    if strategy == "oracle":
        return {"prompt": gold["prompt"],
                "can_execute": gold["can_execute"],
                "clarifying_question": gold.get("clarifying_question"),
                "ambiguity_type": gold.get("ambiguity_type")}
    raise ValueError("unknown strategy: %s" % strategy)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("strategy", choices=["never-ask", "always-ask", "oracle"])
    p.add_argument("--gold", default="dataset/splits/test.jsonl")
    p.add_argument("-o", "--out", default=None,
                   help="output path (default: stdout)")
    args = p.parse_args(argv)

    try:
        gold = load_jsonl(args.gold)
        rows = [baseline_row(r, args.strategy) for r in gold]
    except (ValueError, KeyError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2

    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %d %s predictions to %s"
              % (len(rows), args.strategy, args.out), file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
