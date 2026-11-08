import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gate_normalize import BLIND_SPOT_MIN_CHARS, in_measured_blind_spot, normalize


class NormalizeTests(unittest.TestCase):
    """Pins the T5 train/inference normalization contract. Experiments and
    the shipped plugin import this same module - if these change, the
    trained head artifact (plugin/gate_head.json) no longer matches."""

    def test_strips_path_like_tokens(self):
        self.assertEqual(
            normalize("fix the bug in src/auth/login.py please"),
            "fix the bug in please",
        )

    def test_strips_common_extensions(self):
        self.assertEqual(
            normalize("update config.yaml and README.md now"),
            "update and now",
        )

    def test_keeps_plain_words_and_numbers(self):
        self.assertEqual(
            normalize("make the search endpoint faster v2"),
            "make the search endpoint faster v2",
        )

    def test_collapses_whitespace(self):
        self.assertEqual(normalize("add   an\n\t index"), "add an index")

    def test_all_path_prompt_normalizes_to_empty(self):
        self.assertEqual(normalize("src/a.py tests/b_test.py"), "")

    def test_word_with_dot_but_no_extension_is_kept(self):
        self.assertEqual(normalize("handle v1.2 rollback"), "handle v1.2 rollback")


class BlindSpotTests(unittest.TestCase):
    def test_below_threshold(self):
        self.assertFalse(in_measured_blind_spot("x" * (BLIND_SPOT_MIN_CHARS - 1)))

    def test_at_threshold(self):
        self.assertTrue(in_measured_blind_spot("x" * BLIND_SPOT_MIN_CHARS))


if __name__ == "__main__":
    unittest.main()
