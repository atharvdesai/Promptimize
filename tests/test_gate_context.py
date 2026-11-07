"""Integration tests tying continuity scoring, the repo cache, and cascading
context together through build_gate_context -- the one entry point a host
agent calls to build everything the decision model needs."""

import tempfile
import unittest
from pathlib import Path

from scripts.gate_context import build_gate_context


def character_counter(text):
    return len(text)


def fake_summarizer(path, content):
    return {"keywords": ["auth", "jwt"], "description": "Handles JWT-based session auth."}


class GateContextTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "cache.json"
        self.repo_file = Path(self._tmp.name) / "auth.py"
        self.repo_file.write_text("def issue_token(): ...")

    def test_builds_cascade_and_continuity_together(self):
        recent = [
            {"role": "user", "content": "Sessions keep expiring too early."},
            {"role": "user", "content": "It's the JWT expiry setting."},
        ]

        result = build_gate_context(
            "Extend the JWT expiry to 24 hours.",
            character_counter,
            recent_context=recent,
            repo_files=[str(self.repo_file)],
            summarizer=fake_summarizer,
            cache_path=self.cache_path,
        )

        self.assertIn("tiers", result)
        self.assertGreaterEqual(len(result["tiers"]), 1)
        self.assertIn("topic_continuity", result)
        self.assertGreater(result["topic_continuity"], 0.0)
        all_repo_ids = [r["id"] for tier in result["tiers"] for r in tier["packet"]["repo_context"]]
        self.assertIn(str(self.repo_file), all_repo_ids)

    def test_unrelated_prompt_has_low_continuity(self):
        recent = [{"role": "user", "content": "Let's redesign the settings page layout."}]

        result = build_gate_context(
            "Fix the flaky integration test for payment retries.",
            character_counter,
            recent_context=recent,
        )

        self.assertLess(result["topic_continuity"], 0.3)

    def test_no_recent_context_gives_no_continuity_signal(self):
        result = build_gate_context("Fix the bug.", character_counter)

        self.assertIsNone(result["topic_continuity"])

    def test_second_call_without_summarizer_reuses_the_cache(self):
        build_gate_context(
            "Extend the JWT expiry.",
            character_counter,
            repo_files=[str(self.repo_file)],
            summarizer=fake_summarizer,
            cache_path=self.cache_path,
        )

        # No summarizer passed this time -- must reuse what's cached, not error.
        result = build_gate_context(
            "Shorten the JWT expiry instead.",
            character_counter,
            repo_files=[str(self.repo_file)],
            cache_path=self.cache_path,
        )

        all_repo_ids = [r["id"] for tier in result["tiers"] for r in tier["packet"]["repo_context"]]
        self.assertIn(str(self.repo_file), all_repo_ids)

    def test_no_repo_files_means_no_repo_context(self):
        result = build_gate_context("Fix the bug.", character_counter)

        all_repo = [r for tier in result["tiers"] for r in tier["packet"]["repo_context"]]
        self.assertEqual(all_repo, [])


if __name__ == "__main__":
    unittest.main()
