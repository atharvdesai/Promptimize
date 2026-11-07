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

Minimal pairs and near-duplicates (roadmap step 3): rows that are deliberately
alike must never land in different splits, or the evaluation leaks. Record
them in dataset/groups.json (see README); every group is assigned atomically
to one split. Adding groups changes split membership even at the same seed,
so it is only done when a new frozen test set is intended.
"""
import argparse
import json
import random
import sys
from pathlib import Path

FIELDS = {"prompt", "can_execute", "confidence", "ambiguity_type",
          "ambiguities", "missing_information", "clarifying_question"}

# Frozen default. Changing this reshuffles the test set; see module docstring.
DEFAULT_SEED = 20261003
DEFAULT_GROUPS = "dataset/groups.json"


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


def load_groups(path):
    """Load dataset/groups.json: {"groups": [["prompt a", "prompt b"], ...]}.

    Returns {prompt: group_index}. Groups are minimal pairs / near-duplicates
    that must share one split. Every group needs at least two distinct prompts
    and groups may not overlap; both are labeling mistakes, not valid input.
    """
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or not isinstance(doc.get("groups"), list):
        raise ValueError("%s must be an object with a 'groups' list" % path)
    group_of = {}
    for gid, members in enumerate(doc["groups"]):
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("group %d needs at least two prompts" % gid)
        if len(set(members)) != len(members):
            raise ValueError("group %d repeats a prompt" % gid)
        for prompt in members:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("group %d has an empty prompt" % gid)
            if prompt in group_of:
                raise ValueError("prompt %r is in two groups" % prompt[:60])
            group_of[prompt] = gid
    return group_of


def make_splits(rows, seed, test_frac=0.10, val_frac=0.10, groups=None):
    """Split rows deterministically. Returns {"train": [...], "val": [...], "test": [...]}.

    Rows keep their original dataset order inside each split. The shuffle uses
    one seeded RNG over strata iterated in sorted-key order, so output depends
    only on (rows, seed, fractions, groups), not on dict or filesystem ordering.
    With no groups the output is unchanged from the original splitter, so the
    frozen splits stay reproducible until a groups file is deliberately added.

    `groups` maps a prompt to a group id; each group's rows are assigned
    atomically to one split (greedy deficit balancing after the stratified
    pass), because near-duplicate rows in different splits leak evaluation.
    """
    group_of = groups or {}
    prompts = {row["prompt"] for row in rows}
    unknown = [p for p in group_of if p not in prompts]
    if unknown:
        raise ValueError("groups name prompts missing from the dataset: %s"
                         % [p[:60] for p in unknown[:3]])

    strata = {}
    grouped = {}
    for index, row in enumerate(rows):
        if set(row) != FIELDS:
            raise ValueError("row %d has unexpected fields: %s" % (index, sorted(row)))
        gid = group_of.get(row["prompt"])
        if gid is None:
            strata.setdefault(stratum_key(row), []).append((index, row))
        else:
            grouped.setdefault(gid, []).append((index, row))

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        members = list(strata[key])
        rng.shuffle(members)
        n_train, n_val, n_test = allocation(len(members), test_frac, val_frac)
        splits["test"].extend(members[:n_test])
        splits["val"].extend(members[n_test:n_test + n_val])
        splits["train"].extend(members[n_test + n_val:])

    if grouped:
        order = ("train", "val", "test")
        total = len(rows)
        targets = {"test": int(total * test_frac + 0.5),
                   "val": int(total * val_frac + 0.5)}
        targets["train"] = total - targets["test"] - targets["val"]
        gids = sorted(grouped)
        rng.shuffle(gids)
        for gid in gids:
            # Largest remaining deficit wins; ties break in train/val/test order.
            best = max(order, key=lambda s: (targets[s] - len(splits[s]),
                                             -order.index(s)))
            splits[best].extend(grouped[gid])

    for name in splits:
        splits[name] = [row for _, row in sorted(splits[name], key=lambda p: p[0])]
    return splits


def write_splits(splits, out_dir):
    """Write train/val/test JSONL files; return {name: sha256} of what was written."""
    import hashlib
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


def build_manifest(rows, splits, hashes, seed, test_frac, val_frac, groups_info=None):
    strata = {}
    for row in rows:
        strata.setdefault(stratum_key(row), 0)
        strata[stratum_key(row)] += 1
    manifest = {
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
    if groups_info:
        manifest["groups"] = groups_info
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="dataset/clarifying_question_dataset.jsonl")
    parser.add_argument("--out", default="dataset/splits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--groups", default=None,
                        help="groups JSON file; defaults to %s when it exists"
                             % DEFAULT_GROUPS)
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

    groups_path = args.groups
    if groups_path is None and Path(DEFAULT_GROUPS).is_file():
        groups_path = DEFAULT_GROUPS
    groups = None
    groups_info = None
    if groups_path is not None:
        if not Path(groups_path).is_file():
            parser.error("groups file not found: %s" % groups_path)
        groups = load_groups(groups_path)
        groups_info = {"file": groups_path,
                       "group_count": len(set(groups.values())),
                       "grouped_rows": len(groups)}

    splits = make_splits(rows, args.seed, args.test_frac, args.val_frac, groups)
    hashes = write_splits(splits, args.out)
    manifest = build_manifest(rows, splits, hashes, args.seed,
                              args.test_frac, args.val_frac, groups_info)
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    print("Wrote %d rows to %s (seed %d)" % (len(rows), args.out, args.seed))
    for name in ("train", "val", "test"):
        print("  %-5s %4d rows  sha256 %s" % (name, len(splits[name]), hashes[name][:12]))
    if groups_info:
        print("  %d groups held together (%d rows)"
              % (groups_info["group_count"], groups_info["grouped_rows"]))
    for key, count in sorted(manifest["strata_rows"].items()):
        held_out = sum(1 for name in ("val", "test")
                       for row in splits[name] if stratum_key(row) == key)
        print("  %-38s %4d rows (%d held out)" % (key, count, held_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
