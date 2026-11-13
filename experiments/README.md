# Experiment loop

Protocol for the research-build-test loop (approved 2026-11-07, iMessage:
"Just loop it ... max $10 to test ... always commit to GitHub for each
trial").

## Rules

- **Selection on train, evaluation on val, test stays reserved.** Model,
  prompt, and threshold choices are made on `dataset/splits/train.jsonl`
  (332 rows). The winner is evaluated once on `dataset/splits/val.jsonl`
  (44 rows). `test.jsonl` has recorded sanity exposure and is never used
  to guide modeling. Splits are frozen (seed 20261003, group-aware).
- **Every trial gets one commit**: hypothesis, config, data/model
  revisions, and results - including failures. No empty churn.
- **Val reuse across trials is selection exposure**: the loop re-reads
  val once per trial to compare approaches, so val numbers are
  DIRECTIONAL, not an independent final evaluation. A clean final claim
  needs a fresh never-developed-against holdout; the test split (recorded
  sanity exposure) is the last resort and still limits the claim.
- **Probe governance**: `t2_probes.jsonl` is a DEVELOPMENT diagnostic -
  its failures may guide remediation, so gains on it afterwards are
  remediation evidence, not independent generalization. Each new
  diagnostic family is authored AFTER the fix it evaluates, frozen
  (SHA256 in trials.jsonl) before scoring, and kept separate from any
  eventual independently sourced test set. Model-authored probes never
  support a real-prompt generalization claim.
- **Augmentation labels are inspected, not assumed**: counterfactual
  surface augmentation may only inherit a label when the transformation
  provably preserves discoverability (shortening must not remove needed
  context; added paths must not make missing info discoverable). Every
  augmented row carries a per-row rationale; sample both flagged and
  unflagged cases for review. Augmentation touches train only - never
  val, never probes.
- **Preserve everything**: raw results, failed variants, prior model
  configs and counts are kept (git or Drive archive); no result is
  silently overwritten.
- **Metrics with honest denominators**: accuracy, over-asking rate
  (false interrupts over executable rows), under-asking rate (missed
  ambiguity over ambiguous rows), Wilson 95% CIs on the actual n.
- **Budget**: $10.00 TOTAL API cap across the whole experiment
  (ledger `~/.promptimize/budget.json`, fail-closed, conservative
  reserves). Local compute is free; sandbox RAM is 2 GB.
- **Licenses**: LAYA is Apache 2.0 (fine-tuning allowed). Anthropic
  Commercial Terms D.4 prohibit using Claude outputs to train competing
  AI models, so Claude is an evaluation COMPARATOR only - never a
  training-data teacher.
- **Dataset provenance caveat (unresolved)**: the README calls the
  dataset "single-author, synthetic ... model-assisted", and the
  assisting model is NOT identified. If Claude assisted, its gold
  questions are Claude outputs and cannot train the question writer
  (that would launder teacher output through the JSONL). Until the user
  confirms the authoring model and its terms, gold questions are
  EVALUATION targets only; question-writer training data must come from
  a license-clean source (open model with a checked license, templates,
  or confirmed-clean authorship).

## Acceptance / stopping rule

The loop stops at a locally-runnable gate that, measured on validation:
1. accuracy >= 0.886 (the Haiku 5.5 comparator's val accuracy),
2. over-asking <= 0.05 (comparator: 0.000), under-asking <= 0.22
   (comparator: 0.217),
3. question quality within a clear margin of the comparator (token-F1
   vs gold questions plus judge comparison),
4. decision latency < 1 s per prompt on a laptop CPU, no per-prompt API
   call,
5. OR the loop is stopped when 3 consecutive structurally different
   approaches fail to close the gap - that negative result gets reported
   with the same rigor.

Checkpoint to the user (via main) after every 2-3 trials or daily,
whichever comes first.
