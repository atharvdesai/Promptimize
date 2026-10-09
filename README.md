# Promptimize

The goal of this tool is to reduce costs in agentic coding by asking clarification questions to the user when the prompt is ambiguous.

> **Status: dataset and context-builder prototype.** There is no trained model or chat/memory integration yet. This repo contains a hand-written seed dataset (420 examples), validation tooling, a deterministic context packer, and tests for its contract.

## Why this exists

When a developer sends a coding agent (Claude Code, Codex, Cursor, Copilot Chat, Aider, ...) a prompt like *"fix the bug"* or *"make it faster"*, the agent usually guesses, burns tokens exploring the codebase and producing a large diff, and the developer then rejects it or corrects it and pays for a second run. One cheap question up front ("Which endpoint is slow, and what's the latency target?") is much cheaper than an expensive wrong attempt.

Promptimize is meant to be a small, fast **gate that sits in front of a coding agent**:

```
developer prompt ──► Promptimize ──┬─► can_execute = true  ──► forward the prompt to the agent unchanged
                                   └─► can_execute = false ──► show ONE clarifying question to the developer,
                                                               then forward the enriched prompt
```

The gate has to be cheap and fast (hence a small fine-tuned model rather than a frontier model on every prompt) and has to be **right about when not to interrupt**. A gate that asks too many questions is worse than no gate, because it adds friction and costs the developer time. See [Calibration rule](#the-calibration-rule-the-most-important-thing-in-this-repo).

## What the model should learn

Given a developer prompt, the model must:

1. Decide whether a coding agent can execute it reliably (`can_execute`).
2. If not, identify the kind of ambiguity (`ambiguity_type`), what is unclear (`ambiguities`), and what information is missing (`missing_information`).
3. Produce **exactly one** clarifying question that removes the most uncertainty.

Scope is software engineering only: coding, debugging, architecture, DevOps/infra, testing, AI/ML, databases, APIs, mobile, frontend, backend, security.

## Repo layout

```
dataset/
  clarifying_question_dataset.jsonl   # the dataset (source of truth), one JSON object per line
  clarifying_question_dataset.pdf     # generated, human-readable rendering for review
scripts/
  validate_dataset.py                 # schema + labeling-rule checks and summary stats
  build_pdf.py                        # regenerates the PDF from the JSONL
  context_builder.py                  # deterministic context packing with soft section budgets
tests/
  test_context_builder.py             # black-box contract tests for context packing
requirements.txt                      # only needed for build_pdf.py (reportlab)
```

**The JSONL is the source of truth. Never edit the PDF by hand.** Edit the JSONL, validate, then regenerate the PDF.

## Data format

One JSON object per line, all seven fields always present.

Executable example:

```json
{"prompt": "Add a --dry-run flag to the deploy script in scripts/deploy.sh that prints the commands it would run without executing them", "can_execute": true, "confidence": 0.97, "ambiguity_type": null, "ambiguities": [], "missing_information": [], "clarifying_question": null}
```

Ambiguous example:

```json
{"prompt": "make it faster", "can_execute": false, "confidence": 0.96, "ambiguity_type": "missing_scope", "ambiguities": ["No target identified for optimization", "No performance baseline or goal stated"], "missing_information": ["Which part of the system is slow", "What metric defines fast enough"], "clarifying_question": "What specifically feels slow — page load, an API endpoint, a database query, or the build/test cycle?"}
```

| Field | Type | Rules |
|---|---|---|
| `prompt` | string | A realistic developer prompt. Unique across the file (case-insensitive). |
| `can_execute` | bool | The core label. |
| `confidence` | float 0–1 | Confidence in the `can_execute` judgment (see below). |
| `ambiguity_type` | string or null | One of the 10 types below when `can_execute` is false, otherwise `null`. |
| `ambiguities` | string[] | Non-empty when ambiguous, `[]` when executable. |
| `missing_information` | string[] | Non-empty when ambiguous, `[]` when executable. |
| `clarifying_question` | string or null | Exactly one specific question containing `?` when ambiguous, otherwise `null`. |

### Ambiguity types

| Type | Meaning | Example prompt |
|---|---|---|
| `missing_context` | The prompt refers to something the agent has no way to see: an error, a symptom, earlier discussion, a "the thing". The biggest category in real usage. | "it doesn't work anymore" |
| `missing_requirement` | The feature is named but its behavior is undefined. | "add auth to the app" |
| `missing_scope` | Unbounded or unclear extent: which part, how much. | "make it faster" |
| `missing_technology` | A choice of tool, provider, or platform is needed and can't be inferred. | "set up error tracking" |
| `missing_file` | The target file or code is not identified. | "add docstrings to the file" |
| `missing_component` | A specific UI element, endpoint, service, or package is unidentified. | "remove the banner" |
| `missing_constraint` | A limit, threshold, or safety condition is unspecified, and guessing is risky. | "add retries to the payment call" |
| `missing_acceptance_criteria` | No way to tell when the work is done or good. | "make onboarding smoother" |
| `multiple_interpretations` | Two or more materially different readings. | "roll back the deploy" |
| `unclear_goal` | It's not clear what outcome the developer wants. | "help me with the database" |

Counts in the current file: `missing_context` 54, `missing_requirement` 26, `missing_scope` 25, `missing_technology` 16, `multiple_interpretations` 16, `unclear_goal` 15, `missing_component` 15, `missing_acceptance_criteria` 15, `missing_file` 14, `missing_constraint` 14. `missing_context` is deliberately over-represented because it dominates real assistant usage.

## Labeling conventions

Read this before adding or editing any example. Consistency here matters more than volume.

### The calibration rule (the most important thing in this repo)

A prompt is **executable** if a coding agent can discover whatever is missing by itself: run the tests, grep the repo, read the config, follow existing conventions. A prompt is **ambiguous** only when the missing information exists only in the developer's head: intent, target environment, a business rule, an error message the agent can't see.

- "Run the test suite and fix whatever's failing." → **executable** (the agent can run the tests).
- "fix the bug" → **ambiguous** (nothing says what the bug is).
- "run the linter and fix what it finds" → **executable**, even though it is 7 words long. Short does not mean ambiguous, and the dataset includes terse executable prompts on purpose so the model doesn't learn length as a shortcut.
- "add retries to the payment call" → **ambiguous**: retrying a non-idempotent charge can double-bill, so the one question that matters is about idempotency keys.

Over-asking is the main failure mode to avoid. When in doubt about a borderline case, give it a lower `confidence` rather than flipping the label.

### Confidence

`confidence` is how sure we are about the `can_execute` label, not how "safe" the prompt is. Executable rows currently range 0.90–0.98; ambiguous rows 0.55–0.97. Genuinely borderline prompts get low values (for example, "the docker build is broken" is 0.60 because an agent might reproduce it locally). **These values are hand-assigned estimates, not measured or calibrated**; treat them as a rough signal (see [Known limitations](#known-limitations)).

### Clarifying questions

- Exactly **one** question, specific and actionable.
- Name concrete options where possible ("page load, an API endpoint, a database query, or the build cycle?").
- Pick the question with the **largest reduction in uncertainty**, not the first thing that's missing.
- Never generic: no "Can you provide more details?", "Can you clarify?", "What do you mean?". The validator rejects these.

### Realism

Prompts are hand-written to sound like what developers actually type: terse senior engineers, beginners, founders ("an app like Uber for tutoring"), enterprise requests (SOC 2, SSO, GDPR), secondhand reports ("my cofounder says the code is bad"), mixed with detailed, well-specified prompts. Avoid textbook phrasing and avoid templated variations of one prompt.

## How to run it

Requires Python 3.9+. The validator uses only the standard library.

Validate the dataset (run this before every commit that touches it):

```bash
python3 scripts/validate_dataset.py
```

It checks the schema, the executable/ambiguous field rules, valid `ambiguity_type` values, that every question has a `?` and isn't generic, and that there are no duplicate prompts. It prints counts per type and exits non-zero on any problem.

Regenerate the PDF (needs `reportlab`; a virtualenv is recommended because system Python on macOS is externally managed):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_pdf.py
```

## Adding examples

1. Append lines to `dataset/clarifying_question_dataset.jsonl` (one JSON object per line, no trailing commas, UTF-8).
2. Follow the [calibration rule](#the-calibration-rule-the-most-important-thing-in-this-repo) and keep ambiguity types balanced; the under-represented ones are the best place to add.
3. Run `python3 scripts/validate_dataset.py` until it exits 0.
4. Regenerate the PDF and commit both files together.

## Known limitations

Be honest about these when evaluating anything built on this data:

- **Small.** 420 examples (210 executable / 210 ambiguous) is enough for a proof of concept, not for a reliable model. The original target was about 1,000, and a few thousand is more realistic for a dependable classifier.
- **Single-author, synthetic.** All prompts were written by one model-assisted author, so they share quirks. Notably, executable prompts tend to be longer and contain file paths while ambiguous ones are shorter, so a model could learn those surface features instead of reasoning about what's missing. Real developer prompts are needed to break this.
- **Labels depend on context the model won't see.** Several rows are marked executable because an agent could discover the answer by running or reading something. At inference time the model sees only the prompt text. Either supply a short repo-context field in the future, or tighten those rows.
- **`confidence` is not calibrated.** It is a hand-assigned estimate. Consider dropping it as a training target or collapsing it into coarse buckets (high/medium/low).
- **No train/validation/test split yet.** Nothing has been evaluated.
- **Single-turn only.** Prompts like "apply the changes we discussed" are labeled ambiguous because no history is available; a real integration would have conversation context.

## Roadmap

Suggested order of work:

1. **Create splits.** 80/10/10, stratified by `ambiguity_type` and `can_execute`, with the test set frozen.
2. **Baseline first.** Run a strong prompted model (few-shot) on the test set before fine-tuning anything. It may match a small fine-tune at this data size and will expose label disagreements.
3. **Grow the data** toward 1,500–3,000 examples, including:
   - real developer prompts (scrubbed of secrets and private code), labeled by hand;
   - **minimal pairs**: the same prompt written once ambiguous and once with the missing detail added, so the model has to learn what's missing rather than surface features;
   - optional repo-context field for rows whose label depends on discoverable context.
4. **Fine-tune** a small instruct model (LoRA on a 7–8B model, or a hosted fine-tuning API) with the output schema as the target.
5. **Evaluate** on the frozen test set. The metrics that matter most:
   - **Over-asking rate**: how often it interrupts an executable prompt (the costly error);
   - **Under-asking rate**: how often it lets an ambiguous prompt through;
   - **Question quality**: does the question target the real missing information? This needs human or LLM-judge review; exact match is not meaningful.
6. **Integrate** as a pre-flight hook for a coding agent and measure the real goal: tokens/cost per completed task with and without the gate.

## Context builder prototype

The context builder accepts the current prompt, recent conversation messages, and already-retrieved memory records. It preserves the prompt, applies configurable soft token targets to recent chat and long-term memories, and spills unused capacity in a configurable order. It does not retrieve memories or call a model.

The default packing budget is **4,096 tokens**, a conservative working target for an 8,192-token ModernBERT-base input. This is not Jev's context limit. The default soft targets are 50% for recent chat and 30% for retrieved memories, leaving 20% shared; unused capacity spills to recent chat first, then memories. These settings are configurable. The caller supplies the tokenizer counter. Assistant-unconfirmed memories are excluded; retained memories carry source and confirmation metadata. Run the contract suite with `python3 -m unittest discover -s tests -v`.

## Contributing

- Keep the JSONL as the source of truth, and keep commits focused (data changes separate from tooling changes).
- Do not commit secrets, customer data, or private code in example prompts. Scrub anything taken from real usage.
- If you change a labeling convention, update this README in the same commit so the next person isn't working from stale rules.
