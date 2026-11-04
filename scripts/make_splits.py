#!/usr/bin/env python3
"""Create frozen, stratified train/validation/test splits of the JSONL dataset.

Roadmap step 1 (see README): an 80/10/10 split, stratified by `can_execute`
and `ambiguity_type`, with the test set frozen. Re-running with the same seed
and the same dataset reproduces byte-identical split files, which is what
makes the test set safe to evaluate against repeatedly.

Usage:
    python3 scripts/make_splits.py                      # defaults below
    python3 scripts/make_splits.py --seed 7 --out /tmp/splits

Do NOT change the seed for an experiment series: a new seed reshuffles the
test set and quietly invalidates every earlier evaluation. Grow the dataset
and re-split only when a new frozen test set is intended.
"""
import argparse
import random
import hashlib
import json
import sys
from pathlib import Path

FIELDS = {"prompt", "can_execute", "confidence", "ambiguity_type",
          "ambiguities", "missing_information", "clarifying_question"}

# Frozen default. Changing this reshuffles the test set; see module docstring.
DEFAULT_SEED = 20261003


def stratum_key(row):
    """Stratification label: executable rows in one stratum, ambiguous rows per type."""
    if row["can_execute"]:
        return "executable"
    return "ambiguous:%s" % row["ambiguity_type"]


def allocation(n, test_frac, val_frac):
    """Return (n_train, n_val, n_test) for a stratum of size n.

    Every stratum with at least 3 rows contributes at least one row to both
    held-out sets, so evaluation covers every ambiguity type. Smaller strata
    stay entirely in train: a 1- or 2-row stratum cannot be split meaningfully.
    """
    if n < 3:
        return n, 0, 0
    n_test = max(1, int(n * test_frac + 0.5))
    n_val = max(1, int(n * val_frac + 0.5))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    return n - n_test - n_val, n_val, n_test


def make_splits(rows, seed, test_frac=0.10, val_frac=0.10):
    """Split rows deterministically. Returns {"train": [...], "val": [...], "test": [...]}.

    Rows keep their original dataset order inside each split. The shuffle uses
    one seeded RNG over strata iterated in sorted-key order, so output depends
    only on (rows, seed, fractions), not on dict or filesystem ordering.
    """
    strata = {}
    for index, row in enumerate(rows):
        if set(row) != FIELDS:
            raise ValueError("row %d has unexpected fields: %s" % (index, sorted(row)))
        strata.setdefault(stratum_key(row), []).append((index, row))

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        members = list(strata[key])
        rng.shuffle(members)
        n_train, n_val, n_test = allocation(len(members), test_frac, val_frac)
        splits["test"].extend(members[:n_test])
        splits["val"].extend(members[n_test:n_test + n_val])
        splits["train"].extend(members[n_test + n_val:])

    for name in splits:
        splits[name] = [row for _, row in sorted(splits[name], key=lambda p: p[0])]
    return splits


def write_splits(splits, out_dir):
    """Write train/val/test JSONL files; return {name: sha256} of what was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("train", "val", "test"):
        path = out_dir / ("%s.jsonl" % name)
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                       for row in splits[name])
        path.write_text(text, encoding="utf-8")
        hashes[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashes


def build_manifest(rows, splits, hashes, seed, test_frac, val_frac):
    strata = {}
    for row in rows:
        strata.setdefault(stratum_key(row), 0)
        strata[stratum_key(row)] += 1
    return {
        "seed": seed,
        "fractions": {"train": round(1 - test_frac - val_frac, 4),
                      "val": val_frac, "test": test_frac},
        "total_rows": len(rows),
        "split_rows": {name: len(splits[name]) for name in ("train", "val", "test")},
        "strata_rows": dict(sorted(strata.items())),
        "sha256": hashes,
        "frozen": "test.jsonl is a frozen evaluation set. Regenerate only with "
                  "the same seed and dataset; a different seed is a new test set.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="dataset/clarifying_question_dataset.jsonl")
    parser.add_argument("--out", default="dataset/splits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args(argv)

    if not 0 < args.test_frac < 1 or not 0 < args.val_frac < 1 \
            or args.test_frac + args.val_frac >= 1:
        parser.error("fractions must be in (0, 1) and sum to less than 1")

    rows = []
    with open(args.data, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        parser.error("no rows found in %s" % args.data)

    splits = make_splits(rows, args.seed, args.test_frac, args.val_frac)
    hashes = write_splits(splits, args.out)
    manifest = build_manifest(rows, splits, hashes, args.seed,
                              args.test_frac, args.val_frac)
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    print("Wrote %d rows to %s (seed %d)" % (len(rows), args.out, args.seed))
    for name in ("train", "val", "test"):
        print("  %-5s %4d rows  sha256 %s" % (name, len(splits[name]), hashes[name][:12]))
    for key, count in sorted(manifest["strata_rows"].items()):
        held_out = sum(1 for name in ("val", "test")
                       for row in splits[name] if stratum_key(row) == key)
        print("  %-38s %4d rows (%d held out)" % (key, count, held_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
