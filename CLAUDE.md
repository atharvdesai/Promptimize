# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Promptimize is a prototype for a cheap pre-flight gate that sits in front of a coding agent: given a developer prompt, decide whether the agent can execute it as-is (`can_execute`), and if not, surface exactly one clarifying question. There is no trained model or chat/memory integration yet — this repo currently contains a hand-written seed dataset (420 labeled examples), validation tooling for that dataset, and a deterministic context-packing module meant for the eventual decision model's input.

Two independent pieces live here, joined only by the end goal:

1. **`dataset/`** — the labeled training data for the gate's classifier, plus a PDF rendering for human review.
2. **`scripts/context_builder.py`** — a standalone, dependency-free packer that assembles the `{prompt, recent_context, memories}` input the decision model will eventually see, within a token budget.

## Commands

```bash
# Validate the dataset (run before every commit that touches the JSONL)
python3 scripts/validate_dataset.py

# Regenerate the dataset PDF (needs reportlab; use a venv on macOS)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_pdf.py

# Run the context builder's contract test suite
python3 -m unittest discover -s tests -v

# Run a single test
python3 -m unittest tests.test_context_builder.ContextBuilderContractTests.test_build_is_deterministic_for_identical_inputs
```

## Dataset: the calibration rule

This is the one rule that governs every label in `dataset/clarifying_question_dataset.jsonl`, and it's the thing most likely to be gotten wrong when adding or editing rows:

A prompt is **executable** if a coding agent can discover whatever is missing by itself (run the tests, grep the repo, read the config, follow existing conventions). It is **ambiguous** only when the missing information exists only in the developer's head (intent, target environment, a business rule, an error message the agent can't see). Over-asking is the main failure mode to avoid — when a case is borderline, lower `confidence` rather than flipping the label.

The JSONL is the source of truth; never hand-edit the PDF. Workflow for any dataset change: edit the JSONL → `python3 scripts/validate_dataset.py` until it exits 0 → regenerate the PDF → commit both files together. Full field schema, the 10 `ambiguity_type` values, and labeling conventions (confidence, question style, realism) are in README.md and enforced by `scripts/validate_dataset.py` — read that script before writing new rows, since it's the actual spec.

## Context builder architecture

`scripts/context_builder.py` has one entry point, `build_context(data, token_counter, *, token_budget=4096, section_targets=None, spillover_order=None)`, and no model/retrieval/network dependency — the caller supplies already-retrieved memories and a tokenizer callback.

Key mechanics worth knowing before touching this file:

- The current `prompt` is **never truncated**. If the prompt plus empty packet framing alone exceeds `token_budget`, `build_context` raises `PromptExceedsBudget` rather than cutting the prompt.
- `section_targets` (default: 50% recent chat, 30% memories, out of capacity remaining after the prompt) are **soft** — they bound the first fill pass, but leftover capacity (unused targets + the unallocated shared reserve) spills over afterward in `spillover_order` (default: recent_context, then memories). This is why `test_soft_targets_do_not_act_as_hard_section_caps` and `test_unused_section_target_spills_to_other_section` both pass.
- Recent messages are filled newest-first (oldest-to-newest input, but priority favors the newest). Memories are filled by `relevance` descending, with `status` (`user_confirmed` > `tool_observed`) and `updated_at` as tiebreakers.
- `assistant_unconfirmed` memories are filtered out entirely before packing and reported separately as `excluded_unconfirmed_memories` — they must never reach the packet.
- `token_counter` is called on the full serialized packet (via `json.dumps(..., separators=(",", ":"))`), not on individual strings, so every `try_add` candidate re-serializes and re-counts the whole packet — this is what makes the budget check exact rather than additive/approximate.
- The 4096-token default is a conservative working budget for an 8192-token model input, not a hard platform limit — it's meant to be overridden by the caller per decision model.

When changing packing behavior, the contract tests in `tests/test_context_builder.py` are the spec to satisfy — they're explicitly black-box and use a fake character-counter, so don't assume any particular tokenizer behavior when editing the builder.
