"""Black-box contract tests for Promptimize's cascading context windows.

This is step 1 of testing the "importance-ordered tiers" idea described in the
project plan: instead of one packed context window, split prompt-adjacent chat,
memories, and repo facts into strictly disjoint, importance-ordered tiers. A
decision model is meant to look at tier 0 first, escalate to tier 1 only if
tier 0 is insufficient, and so on -- so once something is placed in an earlier
(more important) tier, it must never resurface in a later one.

Like tests/test_context_builder.py, these use a fake character counter so the
tests describe the packing *contract*, not any particular tokenizer. No model
or escalation loop is exercised here -- that is explicitly a later step.

The public contract under test is:
    build_cascading_context(data, token_counter, *, token_budget=4096,
                             section_targets=..., spillover_order=...,
                             max_tiers=10)
"""

import copy
import json
import unittest

from scripts.context_builder import (
    ContextBuildError,
    PromptExceedsBudget,
    build_cascading_context,
)


def character_counter(text):
    """Deterministic stand-in for a model tokenizer in unit tests."""
    return len(text)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def message(role, content):
    return {"role": role, "content": content}


def memory(memory_id, content, *, relevance=0.5, status="user_confirmed",
           source="conversation:thread-1/message-1", updated_at="2026-10-01T12:00:00Z"):
    return {
        "id": memory_id,
        "content": content,
        "source": source,
        "status": status,
        "relevance": relevance,
        "updated_at": updated_at,
    }


def repo_fact(fact_id, content, *, relevance=0.5, updated_at=None):
    item = {"id": fact_id, "content": content, "relevance": relevance}
    if updated_at is not None:
        item["updated_at"] = updated_at
    return item


def empty_packet_size(prompt):
    return len(compact_json(
        {"version": 1, "prompt": prompt, "recent_context": [], "memories": [], "repo_context": []}
    ))


def packet_with(prompt, *, recent=(), memories=(), repo=()):
    return {
        "version": 1,
        "prompt": prompt,
        "recent_context": list(recent),
        "memories": list(memories),
        "repo_context": list(repo),
    }


class CascadingContextContractTests(unittest.TestCase):
    def build(self, data, *, budget=4096, counter=character_counter,
              section_targets=None, spillover_order=None, max_tiers=10):
        options = {"token_budget": budget, "max_tiers": max_tiers}
        if section_targets is not None:
            options["section_targets"] = section_targets
        if spillover_order is not None:
            options["spillover_order"] = spillover_order
        return build_cascading_context(data, counter, **options)

    # -- basic shape -----------------------------------------------------

    def test_prompt_only_input_produces_a_single_tier_with_prompt_unchanged(self):
        prompt = "Fix the pagination bug in the listings endpoint."

        result = self.build({"prompt": prompt})

        self.assertEqual(len(result["tiers"]), 1)
        self.assertEqual(result["tiers"][0]["packet"]["prompt"], prompt)
        self.assertEqual(result["tiers"][0]["packet"]["recent_context"], [])
        self.assertEqual(result["tiers"][0]["packet"]["memories"], [])
        self.assertEqual(result["tiers"][0]["packet"]["repo_context"], [])
        self.assertEqual(result["unplaceable"], {"recent_context": [], "memories": [], "repo_context": []})

    def test_prompt_is_never_truncated_to_make_room_for_context(self):
        prompt = "Keep every word of this request, including this exact suffix: DO NOT DROP"
        data = {"prompt": prompt, "recent_context": [message("user", "irrelevant " * 100)]}

        result = self.build(data, budget=empty_packet_size(prompt))

        self.assertEqual(result["tiers"][0]["packet"]["prompt"], prompt)

    def test_prompt_alone_over_budget_fails_explicitly_instead_of_truncating(self):
        prompt = "A sufficiently long request that cannot fit in this tiny budget."

        with self.assertRaises(PromptExceedsBudget):
            self.build({"prompt": prompt}, budget=10)

    def test_default_budget_is_4096(self):
        result = build_cascading_context({"prompt": "Check the failing test."}, character_counter)

        self.assertEqual(result["tiers"][0]["token_budget"], 4096)

    def test_every_tier_stays_within_its_token_budget(self):
        prompt = "Continue."
        data = {
            "prompt": prompt,
            "recent_context": [message("user", f"message {i} " * 3) for i in range(10)],
            "memories": [memory(f"m{i}", f"memory fact {i} " * 3, relevance=i / 10) for i in range(10)],
            "repo_context": [repo_fact(f"r{i}", f"repo fact {i} " * 3, relevance=i / 10) for i in range(10)],
        }

        result = self.build(data, budget=200)

        for tier in result["tiers"]:
            self.assertEqual(tier["token_count"], character_counter(tier["serialized"]))
            self.assertLessEqual(tier["token_count"], tier["token_budget"])

    # -- disjointness and importance ordering -----------------------------

    def test_tiers_are_strictly_disjoint_and_ordered_newest_first(self):
        prompt = "Continue."
        contents = [f"message-{i}-xxxxxxxxxx" for i in range(4)]  # equal length, distinguishable
        recent = [message("user", c) for c in contents]
        tier0_expected = packet_with(prompt, recent=[recent[2], recent[3]])
        budget = len(compact_json(tier0_expected))

        result = self.build({"prompt": prompt, "recent_context": recent}, budget=budget)

        self.assertEqual(len(result["tiers"]), 2)
        self.assertEqual(result["tiers"][0]["packet"]["recent_context"], [recent[2], recent[3]])
        self.assertEqual(result["tiers"][1]["packet"]["recent_context"], [recent[0], recent[1]])
        self.assertEqual(result["unplaceable"], {"recent_context": [], "memories": [], "repo_context": []})

        # No message appears in more than one tier.
        seen = []
        for tier in result["tiers"]:
            seen.extend(id(m) for m in tier["packet"]["recent_context"])
            seen_contents = [m["content"] for m in tier["packet"]["recent_context"]]
        all_contents = [m["content"] for tier in result["tiers"] for m in tier["packet"]["recent_context"]]
        self.assertEqual(sorted(all_contents), sorted(contents))
        self.assertEqual(len(all_contents), len(set(all_contents)))

    def test_memories_and_repo_facts_cascade_by_relevance_highest_first(self):
        prompt = "Continue."
        low_mem = memory("low-mem", "An old preference.", relevance=0.1)
        high_mem = memory("high-mem", "The current requirement.", relevance=0.9)
        low_repo = repo_fact("low-repo", "A minor detail about the repo.", relevance=0.1)
        high_repo = repo_fact("high-repo", "The main architecture fact.", relevance=0.9)
        tier0_expected = packet_with(prompt, memories=[high_mem], repo=[high_repo])
        budget = len(compact_json(tier0_expected))

        result = self.build(
            {"prompt": prompt, "memories": [low_mem, high_mem], "repo_context": [low_repo, high_repo]},
            budget=budget,
        )

        self.assertEqual([m["id"] for m in result["tiers"][0]["packet"]["memories"]], ["high-mem"])
        self.assertEqual([r["id"] for r in result["tiers"][0]["packet"]["repo_context"]], ["high-repo"])
        self.assertEqual([m["id"] for m in result["tiers"][1]["packet"]["memories"]], ["low-mem"])
        self.assertEqual([r["id"] for r in result["tiers"][1]["packet"]["repo_context"]], ["low-repo"])

    def test_three_sections_mix_and_cascade_together(self):
        # recent_context, memories, and repo_context don't cost the same number of
        # tokens per item (memories carry source/status/updated_at; repo facts
        # don't), so a tight budget won't necessarily split "one of each" per
        # tier. What must still hold: every candidate is placed exactly once,
        # nothing is left unplaceable, and within each section the higher-priority
        # item never lands in a *later* tier than the lower-priority one.
        prompt = "Continue."
        older = message("user", "AAAAAAAAAAAAAAAAAAAA")
        newer = message("user", "BBBBBBBBBBBBBBBBBBBB")
        low_mem = memory("mem-lo", "CCCCCCCCCCCCCCCCCCCC", relevance=0.2)
        high_mem = memory("mem-hi", "DDDDDDDDDDDDDDDDDDDD", relevance=0.9)
        low_repo = repo_fact("repo-lo", "EEEEEEEEEEEEEEEEEEEE", relevance=0.2)
        high_repo = repo_fact("repo-hi", "FFFFFFFFFFFFFFFFFFFF", relevance=0.9)
        budget = len(compact_json(packet_with(prompt, recent=[newer], memories=[high_mem])))

        result = self.build(
            {
                "prompt": prompt,
                "recent_context": [older, newer],
                "memories": [low_mem, high_mem],
                "repo_context": [low_repo, high_repo],
            },
            budget=budget,
        )

        self.assertGreaterEqual(len(result["tiers"]), 2)
        self.assertEqual(result["unplaceable"], {"recent_context": [], "memories": [], "repo_context": []})

        def tier_index_of(section, item_id_or_content, key):
            for i, tier in enumerate(result["tiers"]):
                if any(item[key] == item_id_or_content for item in tier["packet"][section]):
                    return i
            self.fail(f"{item_id_or_content!r} was never placed in any tier")

        all_recent_contents = [
            m["content"] for tier in result["tiers"] for m in tier["packet"]["recent_context"]
        ]
        all_memory_ids = [m["id"] for tier in result["tiers"] for m in tier["packet"]["memories"]]
        all_repo_ids = [r["id"] for tier in result["tiers"] for r in tier["packet"]["repo_context"]]
        self.assertEqual(sorted(all_recent_contents), sorted([older["content"], newer["content"]]))
        self.assertEqual(sorted(all_memory_ids), ["mem-hi", "mem-lo"])
        self.assertEqual(sorted(all_repo_ids), ["repo-hi", "repo-lo"])

        self.assertLessEqual(
            tier_index_of("recent_context", newer["content"], "content"),
            tier_index_of("recent_context", older["content"], "content"),
        )
        self.assertLessEqual(
            tier_index_of("memories", "mem-hi", "id"), tier_index_of("memories", "mem-lo", "id")
        )
        self.assertLessEqual(
            tier_index_of("repo_context", "repo-hi", "id"), tier_index_of("repo_context", "repo-lo", "id")
        )

    # -- termination and unplaceable items --------------------------------

    def test_cascading_stops_once_every_candidate_is_placed_no_trailing_empty_tier(self):
        prompt = "Continue."
        contents = [f"message-{i}-xxxxxxxxxx" for i in range(4)]
        recent = [message("user", c) for c in contents]
        tier0_expected = packet_with(prompt, recent=[recent[2], recent[3]])
        budget = len(compact_json(tier0_expected))

        result = self.build({"prompt": prompt, "recent_context": recent}, budget=budget, max_tiers=10)

        self.assertEqual(len(result["tiers"]), 2)  # not padded with a trailing empty tier

    def test_a_candidate_too_large_to_ever_fit_alone_is_reported_unplaceable(self):
        prompt = "Continue."
        small = message("user", "ok")
        huge = message("user", "X" * 5000)
        budget = empty_packet_size(prompt) + 50

        result = self.build({"prompt": prompt, "recent_context": [huge, small]}, budget=budget)

        all_recent = [m for tier in result["tiers"] for m in tier["packet"]["recent_context"]]
        self.assertEqual(all_recent, [small])
        self.assertEqual(result["unplaceable"]["recent_context"], ["recent:0"])

    def test_max_tiers_caps_cascading_and_reports_the_rest_as_unplaceable(self):
        prompt = "Continue."
        contents = [f"message-{i}-" + "C" * 50 for i in range(3)]
        recent = [message("user", c) for c in contents]
        one_message_packet = packet_with(prompt, recent=[recent[0]])
        budget = len(compact_json(one_message_packet))

        result = self.build({"prompt": prompt, "recent_context": recent}, budget=budget, max_tiers=2)

        self.assertEqual(len(result["tiers"]), 2)
        placed = [m["content"] for tier in result["tiers"] for m in tier["packet"]["recent_context"]]
        self.assertEqual(placed, [contents[2], contents[1]])
        self.assertEqual(result["unplaceable"]["recent_context"], ["recent:0"])

    def test_unconfirmed_assistant_memory_is_excluded_from_every_tier(self):
        unconfirmed = memory("guess-1", "The user probably wants dark mode.", status="assistant_unconfirmed")

        result = self.build({"prompt": "Continue.", "memories": [unconfirmed]})

        all_memories = [m for tier in result["tiers"] for m in tier["packet"]["memories"]]
        self.assertEqual(all_memories, [])
        self.assertEqual(result["excluded_unconfirmed_memories"], ["guess-1"])

    # -- validation ---------------------------------------------------------

    def test_repo_context_schema_errors_are_rejected(self):
        invalid_cases = [
            ({"prompt": "Do it.", "repo_context": [{"content": "x", "relevance": 0.5}]}, "id"),
            ({"prompt": "Do it.", "repo_context": [{"id": "r1", "relevance": 0.5}]}, "content"),
            ({"prompt": "Do it.", "repo_context": [{"id": "r1", "content": "x"}]}, "relevance"),
            ({"prompt": "Do it.", "repo_context": [{"id": "r1", "content": "x", "relevance": "high"}]}, "relevance"),
            ({"prompt": "Do it.", "repo_context": [{"id": "r1", "content": "x", "relevance": float("nan")}]}, "relevance"),
            ({"prompt": "Do it.", "repo_context": "not-a-list"}, "repo_context"),
        ]
        for data, expected_message in invalid_cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ContextBuildError, expected_message):
                    self.build(data)

    def test_section_targets_must_cover_all_three_cascade_sections(self):
        with self.assertRaises(ContextBuildError):
            self.build({"prompt": "Continue."}, section_targets={"recent_context": 0.5, "memories": 0.5})

    def test_max_tiers_must_be_a_positive_integer(self):
        with self.assertRaisesRegex(ContextBuildError, "max_tiers"):
            self.build({"prompt": "Continue."}, max_tiers=0)

    # -- determinism and purity ----------------------------------------------

    def test_build_is_deterministic_for_identical_inputs(self):
        data = {
            "prompt": "Update billing.",
            "recent_context": [message("user", "Keep the current provider.")],
            "memories": [memory("pref", "Use Stripe.", relevance=0.9)],
            "repo_context": [repo_fact("arch-1", "Billing lives in billing/.", relevance=0.7)],
        }

        first = self.build(data)
        second = self.build(data)

        self.assertEqual(first, second)

    def test_build_does_not_mutate_caller_input(self):
        data = {
            "prompt": "Update billing.",
            "recent_context": [message("user", "Keep the current provider.")],
            "memories": [memory("pref", "Use Stripe.", relevance=0.9)],
            "repo_context": [repo_fact("arch-1", "Billing lives in billing/.", relevance=0.7)],
        }
        original = copy.deepcopy(data)

        self.build(data)

        self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
