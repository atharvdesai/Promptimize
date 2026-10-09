"""Contract tests for the host-agent-summarized, cached repo context."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.repo_context_cache import RepoContextCache, RepoContextCacheError


def fake_summarizer(calls):
    def summarize(path, content):
        calls.append(path)
        return {"keywords": [path.split("/")[-1]], "description": f"Summary of {path} ({len(content)} chars)"}

    return summarize


class RepoContextCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "repo_context_cache.json"

    def test_new_files_are_summarized_and_persisted(self):
        cache = RepoContextCache(self.cache_path)
        calls = []

        cache.refresh({"a.py": "print('hi')"}, fake_summarizer(calls))

        self.assertEqual(calls, ["a.py"])
        self.assertTrue(self.cache_path.exists())
        on_disk = json.loads(self.cache_path.read_text())
        self.assertIn("a.py", on_disk)
        self.assertEqual(on_disk["a.py"]["keywords"], ["a.py"])

    def test_unchanged_file_is_not_re_summarized(self):
        cache = RepoContextCache(self.cache_path)
        calls = []
        cache.refresh({"a.py": "print('hi')"}, fake_summarizer(calls))

        cache.refresh({"a.py": "print('hi')"}, fake_summarizer(calls))

        self.assertEqual(calls, ["a.py"])  # second refresh made zero new calls

    def test_changed_file_is_re_summarized(self):
        cache = RepoContextCache(self.cache_path)
        calls = []
        cache.refresh({"a.py": "print('hi')"}, fake_summarizer(calls))

        cache.refresh({"a.py": "print('bye')"}, fake_summarizer(calls))

        self.assertEqual(calls, ["a.py", "a.py"])

    def test_files_dropped_from_the_call_are_dropped_from_the_cache(self):
        cache = RepoContextCache(self.cache_path)
        calls = []
        cache.refresh({"a.py": "x", "b.py": "y"}, fake_summarizer(calls))

        result = cache.refresh({"a.py": "x"}, fake_summarizer(calls))

        self.assertEqual(set(result), {"a.py"})

    def test_to_repo_context_compacts_cache_entries(self):
        cache = RepoContextCache(self.cache_path)
        calls = []
        cache.refresh({"auth/jwt.py": "token stuff"}, fake_summarizer(calls))

        repo_context = cache.to_repo_context()

        self.assertEqual(len(repo_context), 1)
        entry = repo_context[0]
        self.assertEqual(entry["id"], "auth/jwt.py")
        self.assertIn("jwt.py", entry["content"])
        self.assertIn("Summary of auth/jwt.py", entry["content"])
        self.assertEqual(entry["relevance"], 0.5)

    def test_to_repo_context_accepts_a_custom_relevance_function(self):
        cache = RepoContextCache(self.cache_path)
        calls = []
        cache.refresh({"a.py": "x", "b.py": "y"}, fake_summarizer(calls))

        repo_context = cache.to_repo_context(relevance=lambda path, entry: 1.0 if path == "a.py" else 0.0)

        by_id = {entry["id"]: entry["relevance"] for entry in repo_context}
        self.assertEqual(by_id, {"a.py": 1.0, "b.py": 0.0})

    def test_invalid_summarizer_result_is_rejected(self):
        cache = RepoContextCache(self.cache_path)
        with self.assertRaises(RepoContextCacheError):
            cache.refresh({"a.py": "x"}, lambda path, content: {"keywords": "not-a-list", "description": "d"})
        with self.assertRaises(RepoContextCacheError):
            cache.refresh({"a.py": "x"}, lambda path, content: {"keywords": [], "description": ""})

    def test_corrupt_cache_file_raises_a_clear_error(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text("not json")
        cache = RepoContextCache(self.cache_path)

        with self.assertRaises(RepoContextCacheError):
            cache.to_repo_context()


if __name__ == "__main__":
    unittest.main()
