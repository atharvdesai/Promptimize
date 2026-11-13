#!/usr/bin/env python3
"""Audit prompt-length coverage of the clarifying-question dataset.

Motivation: P2/P3 showed the shipped decider misses long ambiguous prompts
(0/8 at >=200 chars, replicating on a cross-encoder) and T6 showed no
threshold policy can fix that blind spot, so the mitigation must be upstream
in training data. This audit quantifies the data hole behind it and gives a
future expanded dataset a pre-registration gate to pass BEFORE retraining:

- per-class prompt-length histograms (executable vs ambiguous)
- asymmetric coverage gaps: length buckets one class covers and the other
  has zero of
- long-tail counts at 200/500 chars per class
- per-split coverage (--splits): whether val/test could even measure a
  long-ambiguous fix
- --min-long-ambiguous N gate: exit 1 unless the dataset has at least N
  ambiguous prompts of length >= --long-threshold (default 200)

Usage:
  python scripts/audit_dataset_coverage.py [path.jsonl ...]
  python scripts/audit_dataset_coverage.py --splits
  python scripts/audit_dataset_coverage.py expanded.jsonl --min-long-ambiguous 40

Exits 0 for a readable audit, 1 if any input is unreadable or a gate fails.
"""
import argparse
import json
import sys
from collections import Counter

EDGES = [0, 50, 100, 150, 200, 300, 500, 1000]
BUCKETS = [f"{lo}-{hi - 1}" for lo, hi in zip(EDGES, EDGES[1:])] + [f"{EDGES[-1]}+"]
LONG_THRESHOLDS = (200, 500)
DEFAULT_SPLITS = ["dataset/splits/train.jsonl", "dataset/splits/val.jsonl",
                  "dataset/splits/test.jsonl"]


def bucket_of(n):
    for lo, hi in zip(EDGES, EDGES[1:]):
        if lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return f"{EDGES[-1]}+"


def load_rows(path):
    """Return (rows, error_count). Malformed lines are counted, not fatal."""
    rows, errors = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                assert isinstance(row["prompt"], str)
                assert isinstance(row["can_execute"], bool)
            except Exception:  # noqa: BLE001 - count and skip, never guess
                errors += 1
                continue
            rows.append(row)
    return rows, errors


def histogram(rows, can_execute):
    hist = Counter()
    for row in rows:
        if row["can_execute"] is can_execute:
            hist[bucket_of(len(row["prompt"]))] += 1
    return {b: hist.get(b, 0) for b in BUCKETS}


def long_tail(rows, can_execute, thresholds=LONG_THRESHOLDS):
    lens = [len(r["prompt"]) for r in rows if r["can_execute"] is can_execute]
    out = {str(t): sum(1 for n in lens if n >= t) for t in thresholds}
    out["max"] = max(lens) if lens else 0
    out["n"] = len(lens)
    return out


def type_by_bucket(rows):
    table = {}
    for row in rows:
        if row["can_execute"]:
            continue
        t = row.get("ambiguity_type") or "unknown"
        table.setdefault(t, Counter())[bucket_of(len(row["prompt"]))] += 1
    return {t: {b: c.get(b, 0) for b in BUCKETS if c.get(b, 0)}
            for t, c in sorted(table.items())}


def find_gaps(hist_exec, hist_amb):
    """Buckets covered by one class but empty in the other (asymmetric),
    plus buckets empty in both (unmeasured region for either class)."""
    asymmetric, unmeasured = [], []
    for b in BUCKETS:
        e, a = hist_exec[b], hist_amb[b]
        if e == 0 and a == 0:
            unmeasured.append(b)
        elif e == 0 or a == 0:
            asymmetric.append({"bucket": b, "executable": e, "ambiguous": a})
    return asymmetric, unmeasured


def audit_dataset(path):
    rows, errors = load_rows(path)
    hist_e = histogram(rows, True)
    hist_a = histogram(rows, False)
    asymmetric, unmeasured = find_gaps(hist_e, hist_a)
    return {
        "path": path,
        "rows": len(rows),
        "malformed": errors,
        "histogram": {"executable": hist_e, "ambiguous": hist_a},
        "long_tail": {"executable": long_tail(rows, True),
                      "ambiguous": long_tail(rows, False)},
        "ambiguity_type_by_bucket": type_by_bucket(rows),
        "gaps": {"asymmetric": asymmetric, "unmeasured": unmeasured},
    }


def render(audit):
    lines = [f"== {audit['path']}  rows={audit['rows']}"
             + (f"  malformed={audit['malformed']}" if audit["malformed"] else "")]
    head = "bucket        " + "".join(f"{b:>9}" for b in BUCKETS)
    lines.append(head)
    for cls in ("executable", "ambiguous"):
        h = audit["histogram"][cls]
        lines.append(f"{cls:<14}" + "".join(f"{h[b]:>9}" for b in BUCKETS))
    for cls in ("executable", "ambiguous"):
        lt = audit["long_tail"][cls]
        lines.append(f"{cls}: n={lt['n']} max={lt['max']} "
                     + " ".join(f">={t}:{lt[t]}" for t in map(str, LONG_THRESHOLDS)))
    for gap in audit["gaps"]["asymmetric"]:
        lines.append(f"ASYMMETRIC GAP {gap['bucket']}: "
                     f"executable={gap['executable']} ambiguous={gap['ambiguous']}")
    if audit["gaps"]["unmeasured"]:
        lines.append("unmeasured buckets (both classes empty): "
                     + ", ".join(audit["gaps"]["unmeasured"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*",
                    help="jsonl files to audit (default: the gold dataset)")
    ap.add_argument("--splits", action="store_true",
                    help="audit dataset/splits/{train,val,test}.jsonl separately")
    ap.add_argument("--min-long-ambiguous", type=int, default=None, metavar="N",
                    help="exit 1 unless ambiguous prompts >= --long-threshold >= N")
    ap.add_argument("--long-threshold", type=int, default=200,
                    help="char length defining 'long' for the gate (default 200)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.splits:
        paths = DEFAULT_SPLITS
    elif args.paths:
        paths = args.paths
    else:
        paths = ["dataset/clarifying_question_dataset.jsonl"]

    audits, failed = [], False
    for path in paths:
        try:
            audits.append(audit_dataset(path))
        except OSError as e:
            print(f"ERROR {path}: {e}", file=sys.stderr)
            failed = True

    for a in audits:
        print(json.dumps(a, indent=1) if args.json else render(a))
        print()

    if args.min_long_ambiguous is not None:
        for a in audits:
            have = sum(c for b, c in a["histogram"]["ambiguous"].items()
                       if int(b.rstrip("+").split("-")[0]) >= args.long_threshold)
            status = "PASS" if have >= args.min_long_ambiguous else "FAIL"
            print(f"GATE {status} {a['path']}: ambiguous >= "
                  f"{args.long_threshold} chars: {have} "
                  f"(required {args.min_long_ambiguous})")
            if status == "FAIL":
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
