"""Contract tests for the cheap, local topic-continuity signal."""

import unittest

from scripts.continuity import score_topic_continuity


def message(content):
    return {"role": "user", "content": content}


class TopicContinuityTests(unittest.TestCase):
    def test_no_recent_context_returns_none(self):
        self.assertIsNone(score_topic_continuity("Add retries to the payment call.", []))

    def test_unrelated_prompt_scores_low(self):
        recent = [message("Let's update the login page styling."), message("Use the new brand colors.")]

        score = score_topic_continuity("Fix the database migration for the orders table.", recent)

        self.assertIsNotNone(score)
        self.assertLess(score, 0.2)

    def test_related_prompt_scores_high(self):
        recent = [
            message("The payment endpoint keeps timing out under load."),
            message("It's the Stripe charge call specifically."),
        ]

        score = score_topic_continuity("Add retries to the Stripe charge call.", recent)

        self.assertGreater(score, 0.2)

    def test_only_considers_the_configured_window(self):
        recent = [message("database migration orders table")] + [message("irrelevant filler")] * 5

        full_window = score_topic_continuity("Fix the database migration.", recent, window=6)
        narrow_window = score_topic_continuity("Fix the database migration.", recent, window=1)

        self.assertGreater(full_window, narrow_window)

    def test_stopwords_alone_do_not_inflate_similarity(self):
        recent = [message("The quick brown fox.")]

        score = score_topic_continuity("Is it for the of and.", recent)

        self.assertIsNone(score)

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            score_topic_continuity("Continue.", [message("x")], window=0)

    def test_empty_recent_messages_are_ignored(self):
        recent = [{"role": "user", "content": ""}, {"role": "assistant", "content": "   "}]

        self.assertIsNone(score_topic_continuity("Fix the bug.", recent))


if __name__ == "__main__":
    unittest.main()
