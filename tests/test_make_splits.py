"""Contract tests for scripts/make_splits.py (black-box, stdlib only)."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "make_splits.py"
spec = importlib.util.spec_from_file_location("make_splits", MODULE_PATH)
make_splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_splits)

FIELDS = ("prompt", "can_execute", "confidence", "ambiguity_type",
          "ambiguities", "missing_information", "clarifying_question")


def executable_row(i):
    return {"prompt": "run the tests and fix failure %d" % i, "can_execute": True,
            "confidence": 0.95, "ambiguity_type": None, "ambiguities": [],
            "missing_information": [], "clarifying_question": None}


def ambiguous_row(i, ambiguity_type="missing_scope"):
    return {"prompt": "make %s part %d faster" % (ambiguity_type, i), "can_execute": False,
            "confidence": 0.9, "ambiguity_type": ambiguity_type,
            "ambiguities": ["no target"], "missing_information": ["which part"],
            "clarifying_question": "Which part is slow?"}


def sample_rows():
    rows = [executable_row(i) for i in range(60)]
    rows += [ambiguous_row(i, "missing_scope") for i in range(30)]
    rows += [ambiguous_row(i, "missing_file") for i in range(10)]
    return rows


def prompts(split):
    return {row["prompt"] for row in split}


class MakeSplitsContractTests(unittest.TestCase):
    def test_every_row_appears_exactly_once(self):
        rows = sample_rows()
        splits = make_splits.make_splits(rows, seed=1)
        total = sum(len(splits[name]) for name in ("train", "val", "test"))
        self.assertEqual(total, len(rows))
        all_prompts = [row["prompt"] for name in ("train", "val", "test")
                       for row in splits[name]]
        self.assertEqual(len(all_prompts), len(set(all_prompts)))

    def test_rows_are_unmodified(self):
        rows = sample_rows()
        splits = make_splits.make_splits(rows, seed=1)
        by_prompt = {row["prompt"]: row for rows_ in splits.values() for row in rows_}
        for row in rows:
            self.assertEqual(by_prompt[row["prompt"]], row)
            self.assertEqual(tuple(sorted(by_prompt[row["prompt"]])), tuple(sorted(FIELDS)))

    def test_deterministic_for_same_seed(self):
        rows = sample_rows()
        first = make_splits.make_splits(rows, seed=42)
        second = make_splits.make_splits(rows, seed=42)
        self.assertEqual(first, second)

    def test_different_seed_changes_membership(self):
        rows = sample_rows()
        first = make_splits.make_splits(rows, seed=1)
        second = make_splits.make_splits(rows, seed=2)
        self.assertNotEqual(prompts(first["test"]), prompts(second["test"]))

    def test_split_sizes_follow_fractions(self):
        rows = sample_rows()
        splits = make_splits.make_splits(rows, seed=1, test_frac=0.10, val_frac=0.10)
        # 100 rows -> 80/10/10 with per-stratum rounding
        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["val"]), 10)
        self.assertEqual(len(splits["test"]), 10)

    def test_every_stratum_held_out(self):
        rows = sample_rows()
        splits = make_splits.make_splits(rows, seed=1)
        held_out = splits["val"] + splits["test"]
        for key in ("executable", "ambiguous:missing_scope", "ambiguous:missing_file"):
            self.assertTrue(any(make_splits.stratum_key(row) == key for row in held_out),
                            "stratum %s missing from held-out sets" % key)

    def test_allocation_tiny_stratum_stays_in_train(self):
        self.assertEqual(make_splits.allocation(1, 0.1, 0.1), (1, 0, 0))
        self.assertEqual(make_splits.allocation(2, 0.1, 0.1), (2, 0, 0))
        self.assertEqual(make_splits.allocation(3, 0.1, 0.1), (1, 1, 1))
        self.assertEqual(make_splits.allocation(14, 0.1, 0.1), (12, 1, 1))

    def test_rejects_rows_with_unexpected_fields(self):
        rows = sample_rows()
        rows[0] = {"prompt": "broken"}
        with self.assertRaises(ValueError):
            make_splits.make_splits(rows, seed=1)

    def test_written_files_match_manifest_hashes(self):
        rows = sample_rows()
        splits = make_splits.make_splits(rows, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            hashes = make_splits.write_splits(splits, tmp)
            manifest = make_splits.build_manifest(rows, splits, hashes, 1, 0.1, 0.1)
            for name in ("train", "val", "test"):
                text = (Path(tmp) / ("%s.jsonl" % name)).read_text(encoding="utf-8")
                self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 manifest["sha256"][name])
            self.assertEqual(manifest["total_rows"], len(rows))
            self.assertEqual(sum(manifest["split_rows"].values()), len(rows))

    def test_cli_end_to_end_and_reproducible(self):
        rows = sample_rows()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows),
                            encoding="utf-8")
            out1, out2 = Path(tmp) / "a", Path(tmp) / "b"
            self.assertEqual(make_splits.main(["--data", str(data),
                                               "--out", str(out1), "--seed", "9"]), 0)
            self.assertEqual(make_splits.main(["--data", str(data),
                                               "--out", str(out2), "--seed", "9"]), 0)
            for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"):
                self.assertEqual((out1 / name).read_bytes(), (out2 / name).read_bytes())
            self.assertEqual(len((out1 / "train.jsonl").read_text().splitlines()), 80)


if __name__ == "__main__":
    unittest.main()


class GroupedSplitTests(unittest.TestCase):
    def grouped_rows(self):
        rows = sample_rows()
        # A minimal pair: same intent, once ambiguous, once with the detail added.
        pair_amb = ambiguous_row(500, "missing_scope")
        pair_amb["prompt"] = "make the search faster"
        pair_exec = executable_row(500)
        pair_exec["prompt"] = ("make the search faster: the /search endpoint p95 "
                               "is 2s, target 300ms, see search/api.py")
        return rows + [pair_amb, pair_exec]

    def test_group_members_share_one_split(self):
        rows = self.grouped_rows()
        groups = {"make the search faster": 0,
                  "make the search faster: the /search endpoint p95 is 2s, target 300ms, see search/api.py": 0}
        for seed in (1, 2, 3, 7, 42):
            splits = make_splits.make_splits(rows, seed=seed, groups=groups)
            homes = [name for name in ("train", "val", "test")
                     for row in splits[name] if row["prompt"] in groups]
            self.assertEqual(len(homes), 2)
            self.assertEqual(homes[0], homes[1],
                             "group straddled splits at seed %d" % seed)

    def test_ungrouped_rows_still_stratified(self):
        rows = self.grouped_rows()
        groups = {"make the search faster": 0,
                  "make the search faster: the /search endpoint p95 is 2s, target 300ms, see search/api.py": 0}
        splits = make_splits.make_splits(rows, seed=1, groups=groups)
        ungrouped_held = [row for name in ("val", "test") for row in splits[name]
                          if row["prompt"] not in groups]
        for key in ("executable", "ambiguous:missing_scope", "ambiguous:missing_file"):
            self.assertTrue(any(make_splits.stratum_key(row) == key
                                for row in ungrouped_held))

    def test_grouped_split_is_deterministic(self):
        rows = self.grouped_rows()
        groups = {"make the search faster": 0,
                  "make the search faster: the /search endpoint p95 is 2s, target 300ms, see search/api.py": 0}
        self.assertEqual(make_splits.make_splits(rows, seed=5, groups=groups),
                         make_splits.make_splits(rows, seed=5, groups=groups))

    def test_no_groups_matches_empty_groups(self):
        rows = sample_rows()
        self.assertEqual(make_splits.make_splits(rows, seed=1),
                         make_splits.make_splits(rows, seed=1, groups={}))

    def test_unknown_grouped_prompt_rejected(self):
        with self.assertRaises(ValueError):
            make_splits.make_splits(sample_rows(), seed=1,
                                    groups={"no such prompt": 0, "also missing": 0})

    def test_load_groups_validation(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            bad_overlap = Path(tmp) / "g.json"
            bad_overlap.write_text(json.dumps({"groups": [["a", "b"], ["b", "c"]]}))
            with self.assertRaises(ValueError):
                make_splits.load_groups(bad_overlap)
            bad_single = Path(tmp) / "h.json"
            bad_single.write_text(json.dumps({"groups": [["a"]]}))
            with self.assertRaises(ValueError):
                make_splits.load_groups(bad_single)
            good = Path(tmp) / "ok.json"
            good.write_text(json.dumps({"groups": [["a", "b"]]}))
            self.assertEqual(make_splits.load_groups(good), {"a": 0, "b": 0})


class FrozenSplitGuardTests(unittest.TestCase):
    """The committed test set is frozen: the real dataset at the default seed
    must reproduce the committed hashes, or any change is a deliberate re-split."""

    def test_real_dataset_reproduces_committed_splits(self):
        root = MODULE_PATH.parent.parent
        rows = [json.loads(line) for line in
                (root / "dataset" / "clarifying_question_dataset.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads((root / "dataset" / "splits" / "manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], make_splits.DEFAULT_SEED)
        splits = make_splits.make_splits(rows, manifest["seed"])
        with tempfile.TemporaryDirectory() as tmp:
            hashes = make_splits.write_splits(splits, tmp)
        self.assertEqual(hashes, manifest["sha256"],
                         "dataset or splitter changed: committed splits no longer "
                         "reproduce; only change with a deliberate new frozen test set")
