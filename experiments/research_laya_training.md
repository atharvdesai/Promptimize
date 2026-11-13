# Research: training LAYA - data scale, routes, compute (2026-11-07)

Sources fetched 2026-11-07:
- https://laya-ai.com/guides/fine-tune-laya (official guide, updated Oct 11 2026)
- https://huggingface.co/convaiinnovations/laya (model card)
- https://huggingface.co/convaiinnovations/laya-typed-decisions (fine-tuned reference checkpoint card)

## How many examples? (source-backed reference points)
- Upstream reference fine-tune (`laya-typed-decisions`): trained on the benchmark's
  **1,200-case training split (~30k typed questions)**, 4 epochs, ~4-5 h on free 2xT4
  (Kaggle). Result: 0.362 -> 0.766 accuracy on the 2,000-decision benchmark; base
  checkpoints score near/below majority-class zero-shot. NOTE: this is the reference
  for UPSTREAM's typed-decisions task, NOT a recommended sample count for our
  ambiguity task - our LC1 curve shows the count question is task/data dependent.
- Community specialisation (cklxx/laya-browser, linked by upstream): single 16 GB GPU,
  no paid API; element top-1 0.10 -> 0.66, task success 0% -> 62%.
- Frozen-encoder head route (community "stuntd" project): trained per-decision heads on
  the frozen encoder from the user's own labeled rows; author reports 89.5% -> 100% on a
  12-label intent task (their figures, their task).
- Calibration needs its own held-out slice: >= 10 calibration items per question type
  minimum, never fitted on training data.
- Our empirical answer will come from learning curves (head trained on grouped
  subsamples 32..332) rather than any single magic number.

## Official training path (installed: laya 0.4.28)
`laya-train --data x.csv --text-column body --label-column label --base english
--out ./ckpt --eval held_out.csv`
- Devices: CUDA, MPS (Apple Silicon via finetune_*_mps.py), or CPU
  (`finetune_single_device.py`, 0.3.22+).
- `--freeze-encoder`: trains ONLY the decision head - fits small GPU or CPU.
- `--dry-run`: reports rows/questions/optimizer updates without training.
- Losses: RLCD (default; proper scoring rule + REINFORCE w/ group-mean baseline) or
  soft-ce; --label-smoothing, --shuffle-options available.
- Calibration: refits one temperature per (type, option count) on a held-out slice;
  warns if <10 items. NEVER gate on min_confidence without fitted calibration.
- Dataset design rules (guide's synthesis): mirror production instructions exactly;
  avoid boolean-word choice labels (use semantic keys); include hard negatives and
  ambiguous cases; choice questions under ~20 options.

## Compute feasibility for Promptimize (421M encoder, honest)
- FULL encoder fine-tune (RLCD, unfrozen): NOT feasible in the 2GB sandbox
  (weights ~1.7 GB fp32 + optimizer states several GB + activations). Routes:
  his Mac (MPS script exists), Kaggle free 2xT4 notebook (his account), or paid GPU
  (needs separate approval - outside current $10 scope unless he approves).
- `--freeze-encoder` head-only training on CPU: memory ~= inference (~1.5 GB RSS
  measured) + small head grads - PLAUSIBLE in the sandbox; must be verified with
  --dry-run then a small real run. Naming: "official-path decision-head fine-tune
  with frozen encoder", NOT full LAYA tuning.
- numpy logistic head on extracted embeddings (T2) remains the lightest comparator.

## Data expansion (user request: generate less-biased examples / pull public data)
- Synthetic expansion: counterfactual surface augmentation (T3-debias) - labels by
  construction, inspected per-row; diagnostic-only caveat applies.
- Public corpora of real developer prompts: candidates to verify for license /
  consent / provenance / duplicates BEFORE use (research pending).
- Any newly sourced data reserved as far as possible for FINAL eval, not development.

## Measured sandbox feasibility (2026-11-07, T4 trial)
- `laya-train --freeze-encoder --device cpu` (laya 0.4.28): dry-run OK (332 rows,
  auto calibration holdout 33 items), but the real run was OOM-SIGKILLed during
  torch model load: fp32 421M weights (~1.7 GB) + torch runtime exceed the 2GB
  sandbox. Official-path head training is NOT feasible here even frozen-encoder.
- Feasible local equivalent: numpy head on ONNX-extracted embeddings (T2, LC1).
- LC1 learning curves (same day): flat - ~32 train rows saturate val acc (0.986).
  Data realism/diversity is the binding constraint, not quantity.
