"""Black-box contract tests for Promptimize's v1 context builder.

These tests intentionally use a fake character counter. They verify that the
builder obeys the token-counting interface and budget without assuming any
particular model tokenizer. Production integration must supply the selected
decision model's real tokenizer.

The public contract under test is:
    build_context(input_data, token_counter, *, token_budget=4096,
                  section_targets=..., spillover_order=...)

Input has a full current prompt, chronological recent messages, and retrieved
memory candidates with relevance and provenance. The result has a serialized
``packet``, its ``token_count``, the effective budget, and lists of omitted
items. Section targets are soft shares of the capacity remaining after the
mandatory prompt. Unused capacity and the flexible reserve spill over in the
configured order. The default shares are a tunable starting point, not Jev's
model limit.
"""

import json
import copy
import unittest

from scripts.context_builder import (
    ContextBuildError,
    PromptExceedsBudget,
    build_context,
)


def character_counter(text):
    """Deterministic stand-in for a model tokenizer in unit tests."""
    return len(text)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def memory(
    memory_id,
    content,
    *,
    relevance=0.5,
    status="user_confirmed",
    source="conversation:thread-1/message-1",
    updated_at="2026-10-01T12:00:00Z",
):
    return {
        "id": memory_id,
        "content": content,
        "source": source,
        "status": status,
        "relevance": relevance,
        "updated_at": updated_at,
    }


class ContextBuilderContractTests(unittest.TestCase):
    def build(
        self,
        data,
        *,
        budget=4096,
        counter=character_counter,
        section_targets=None,
        spillover_order=None,
    ):
        options = {"token_budget": budget}
        if section_targets is not None:
            options["section_targets"] = section_targets
        if spillover_order is not None:
            options["spillover_order"] = spillover_order
        return build_context(data, counter, **options)

    def test_prompt_only_input_produces_a_packet_with_prompt_unchanged(self):
        prompt = "Fix the pagination bug in the listings endpoint."

        result = self.build({"prompt": prompt})

        self.assertEqual(result["packet"]["prompt"], prompt)
        self.assertEqual(result["packet"]["recent_context"], [])
        self.assertEqual(result["packet"]["memories"], [])

    def test_prompt_is_never_truncated_to_make_room_for_context(self):
        prompt = "Keep every word of this request, including this exact suffix: DO NOT DROP"
        data = {
            "prompt": prompt,
            "recent_context": [{"role": "user", "content": "irrelevant " * 100}],
        }

        result = self.build(data, budget=len(compact_json({
            "version": 1, "prompt": prompt, "recent_context": [], "memories": []
        })))

        self.assertEqual(result["packet"]["prompt"], prompt)
        self.assertEqual(result["packet"]["recent_context"], [])

    def test_prompt_alone_over_budget_fails_explicitly_instead_of_truncating(self):
        prompt = "A sufficiently long request that cannot fit in this tiny budget."

        with self.assertRaises(PromptExceedsBudget):
            self.build({"prompt": prompt}, budget=10)

    def test_default_budget_is_4096(self):
        result = build_context({"prompt": "Check the failing test."}, character_counter)

        self.assertEqual(result["token_budget"], 4096)

    def test_packet_count_uses_the_supplied_counter_and_stays_within_budget(self):
        data = {
            "prompt": "Fix this endpoint.",
            "recent_context": [{"role": "user", "content": "It is the orders endpoint."}],
        }

        result = self.build(data, budget=180)

        self.assertEqual(result["token_count"], character_counter(result["serialized"]))
        self.assertEqual(result["serialized"], compact_json(result["packet"]))
        self.assertLessEqual(result["token_count"], result["token_budget"])

    def test_messages_that_fit_are_preserved_with_roles_and_in_chronological_order(self):
        data = {
            "prompt": "Implement the agreed change.",
            "recent_context": [
                {"role": "user", "content": "First, update the export."},
                {"role": "assistant", "content": "I found the export code."},
                {"role": "user", "content": "CSV is the required format."},
            ],
        }

        result = self.build(data)

        self.assertEqual(result["packet"]["recent_context"], data["recent_context"])

    def test_developer_role_is_preserved_in_recent_context(self):
        message = {"role": "developer", "content": "Follow the repository conventions."}

        result = self.build({"prompt": "Update the endpoint.", "recent_context": [message]})

        self.assertEqual(result["packet"]["recent_context"], [message])

    def test_recent_context_selection_prefers_newer_messages_when_budget_is_tight(self):
        prompt = "Continue the task."
        older = {"role": "user", "content": "An unrelated old topic."}
        newer = {"role": "user", "content": "The target is the billing page."}
        expected_packet = {
            "version": 1,
            "prompt": prompt,
            "recent_context": [newer],
            "memories": [],
        }

        result = self.build(
            {"prompt": prompt, "recent_context": [older, newer]},
            budget=len(compact_json(expected_packet)),
        )

        self.assertEqual(result["packet"]["recent_context"], [newer])
        self.assertIn("recent:0", result["omitted_recent"])

    def test_retrieved_memories_that_fit_keep_their_provenance_and_status(self):
        item = memory(
            "decision-42",
            "The billing page uses the existing Stripe Checkout flow.",
            source="conversation:thread-7/message-18",
            status="user_confirmed",
        )

        result = self.build({"prompt": "Update the billing page.", "memories": [item]})

        retained = result["packet"]["memories"][0]
        self.assertEqual(retained["id"], item["id"])
        self.assertEqual(retained["source"], item["source"])
        self.assertEqual(retained["status"], item["status"])
        self.assertEqual(retained["content"], item["content"])

    def test_memory_selection_prefers_more_relevant_retrieval_when_budget_is_tight(self):
        prompt = "Update billing."
        low = memory("low", "An old preference about the footer.", relevance=0.2)
        high = memory("high", "Use the Stripe Checkout flow.", relevance=0.95)
        expected_packet = {
            "version": 1,
            "prompt": prompt,
            "recent_context": [],
            "memories": [{
                "id": high["id"],
                "content": high["content"],
                "source": high["source"],
                "status": high["status"],
                "relevance": high["relevance"],
                "updated_at": high["updated_at"],
            }],
        }

        result = self.build(
            {"prompt": prompt, "memories": [low, high]},
            budget=len(compact_json(expected_packet)),
        )

        self.assertEqual([m["id"] for m in result["packet"]["memories"]], ["high"])
        self.assertIn("low", result["omitted_memories"])

    def test_raw_finite_retrieval_scores_are_accepted(self):
        negative = memory("negative", "A memory with a negative cosine score.", relevance=-0.25)
        high = memory("bm25", "A memory with an unnormalized retrieval score.", relevance=4.2)

        result = self.build({"prompt": "Continue.", "memories": [negative, high]})

        self.assertEqual([item["id"] for item in result["packet"]["memories"]], ["bm25", "negative"])

    def test_recent_context_and_memories_can_both_be_included(self):
        recent = [{"role": "user", "content": "Use the existing admin page."}]
        long_term = memory("pref-1", "The user prefers minimal UI changes.")

        result = self.build({
            "prompt": "Update admin.",
            "recent_context": recent,
            "memories": [long_term],
        })

        self.assertEqual(result["packet"]["recent_context"], recent)
        self.assertEqual(result["packet"]["memories"][0]["id"], "pref-1")

    def test_soft_targets_do_not_act_as_hard_section_caps(self):
        recent = [
            {"role": "user", "content": "The target is the billing page."},
            {"role": "user", "content": "Keep the existing Stripe flow."},
        ]
        memory_item = memory("billing-pref", "Avoid changing payment providers.")

        result = self.build(
            {"prompt": "Update billing.", "recent_context": recent, "memories": [memory_item]},
            section_targets={"recent_context": 0.10, "memories": 0.10},
        )

        self.assertEqual(result["packet"]["recent_context"], recent)
        self.assertEqual(result["packet"]["memories"][0]["id"], "billing-pref")

    def test_unused_section_target_spills_to_other_section(self):
        prompt = "Continue."
        recent = [
            {"role": "user", "content": "The target is the settings page."},
            {"role": "user", "content": "Keep the current behavior."},
        ]
        expected_packet = {
            "version": 1,
            "prompt": prompt,
            "recent_context": recent,
            "memories": [],
        }

        result = self.build(
            {"prompt": prompt, "recent_context": recent, "memories": []},
            budget=len(compact_json(expected_packet)),
            section_targets={"recent_context": 0.20, "memories": 0.50},
            spillover_order=("recent_context", "memories"),
        )

        self.assertEqual(result["packet"]["recent_context"], recent)
        self.assertEqual(result["packet"]["memories"], [])

    def test_spillover_order_is_configurable_when_sections_compete(self):
        prompt = "Continue."
        recent = {"role": "user", "content": "Recent conversation target."}
        memory_item = memory("long-term", "Long-term project decision.")
        expected_packet = {
            "version": 1,
            "prompt": prompt,
            "recent_context": [],
            "memories": [{
                "id": memory_item["id"],
                "content": memory_item["content"],
                "source": memory_item["source"],
                "status": memory_item["status"],
                "relevance": memory_item["relevance"],
                "updated_at": memory_item["updated_at"],
            }],
        }

        result = self.build(
            {"prompt": prompt, "recent_context": [recent], "memories": [memory_item]},
            budget=len(compact_json(expected_packet)),
            section_targets={"recent_context": 0.0, "memories": 0.0},
            spillover_order=("memories", "recent_context"),
        )

        self.assertEqual(result["packet"]["recent_context"], [])
        self.assertEqual([m["id"] for m in result["packet"]["memories"]], ["long-term"])

    def test_section_targets_must_be_valid_shares(self):
        with self.assertRaises(ContextBuildError):
            self.build(
                {"prompt": "Continue."},
                section_targets={"recent_context": 0.8, "memories": 0.5},
            )

    def test_unconfirmed_assistant_memory_is_never_promoted_to_confirmed_context(self):
        unconfirmed = memory(
            "guess-1",
            "The user probably wants a dark theme.",
            status="assistant_unconfirmed",
            source="assistant:turn-5",
        )

        result = self.build({"prompt": "Update the settings page.", "memories": [unconfirmed]})

        self.assertEqual(result["packet"]["memories"], [])
        self.assertEqual(result["excluded_unconfirmed_memories"], ["guess-1"])

    def test_build_is_deterministic_for_identical_inputs(self):
        data = {
            "prompt": "Update billing.",
            "recent_context": [{"role": "user", "content": "Keep the current provider."}],
            "memories": [memory("pref", "Use Stripe.", relevance=0.9)],
        }

        first = self.build(data)
        second = self.build(data)

        self.assertEqual(first, second)

    def test_build_does_not_mutate_caller_input(self):
        data = {
            "prompt": "Update billing.",
            "recent_context": [{"role": "user", "content": "Keep the current provider."}],
            "memories": [memory("pref", "Use Stripe.", relevance=0.9)],
        }
        original = copy.deepcopy(data)

        self.build(data)

        self.assertEqual(data, original)

    def test_invalid_prompt_and_context_records_fail_with_clear_errors(self):
        invalid_cases = [
            ({"prompt": "   "}, "prompt"),
            ({"prompt": "Do it.", "recent_context": [{"role": "robot", "content": "x"}]}, "role"),
            ({"prompt": "Do it.", "memories": [memory("x", "y", relevance=float("nan"))]}, "relevance"),
            ({"prompt": "Do it.", "memories": [memory("x", "y", status=[])]}, "status"),
        ]

        for data, expected_message in invalid_cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ContextBuildError, expected_message):
                    self.build(data)

    def test_token_counter_must_return_a_nonnegative_integer(self):
        for invalid_count in (-1, 1.5, True):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaisesRegex(ContextBuildError, "token_counter"):
                    self.build({"prompt": "Do it."}, counter=lambda _: invalid_count)

    def test_spillover_order_must_list_both_sections_once(self):
        invalid_orders = [
            ("recent_context",),
            ("recent_context", "recent_context"),
            "recent_context,memories",
        ]
        for order in invalid_orders:
            with self.subTest(order=order):
                with self.assertRaisesRegex(ContextBuildError, "spillover_order"):
                    self.build({"prompt": "Do it."}, spillover_order=order)


if __name__ == "__main__":
    unittest.main()
