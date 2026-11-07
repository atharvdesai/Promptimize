# Promptimize gate (Claude Code hook, experimental)

A `UserPromptSubmit` hook that watches your prompts before Claude Code
processes them. A local decision model (LAYA, ONNX, on your machine)
scores whether the request is executable as-is; when it scores below the
threshold, Haiku drafts one clarifying question and the hook injects it
as context so Claude asks you natively. It never blocks your prompt.

**Status: experimental command-hook prototype with manual settings.json
wiring. This is not an installed or packaged Claude Code plugin, and the
native Claude Code integration has not been tested yet (developed in a
Linux sandbox without the Claude Code CLI; macOS dogfooding is the first
real test). Decision quality measured on the frozen validation split
(n=44) - real-world usefulness is unproven until used on real work.**

## Setup

```bash
pip install laya onnxruntime
# ~850 MB one-time download:
curl -L -o plugin/model.onnx \
  https://huggingface.co/codenamev/laya-onnx/resolve/main/english/model.onnx
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [{"type": "command",
        "command": "python3 /path/to/Promptimize/plugin/promptimize_gate.py"}]}
    ]
  }
}
```

**API key (never paste it into chat, settings.json, or the repo):**

Preferred, macOS Keychain - run this once (it prompts nothing; replace
the placeholder yourself, the key is never echoed back by the hook):

```bash
security add-generic-password -s promptimize-anthropic -a "$USER" -w 'sk-ant-...'
export PROMPTIMIZE_ANTHROPIC_KEY_CMD='security find-generic-password -s promptimize-anthropic -w'
```

Or a permission-locked file (the hook REFUSES a group/world-readable
key file):

```bash
install -m 600 /dev/null ~/.promptimize/anthropic_key   # then edit in an editor
export PROMPTIMIZE_ANTHROPIC_KEY_FILE=~/.promptimize/anthropic_key
```

The key is billed to your Anthropic API account, separate from a Claude
subscription. It is read at call time, never logged or echoed.

**Budget provisioning (required once per machine):** the approved spend
cap is a TOTAL across the whole experiment, not per install or per
machine. The ledger refuses every paid call until explicitly
provisioned with the REMAINING allowance - approved total minus
everything already spent (and minus any outstanding reservations). This
experiment has already spent $0.004489 of the approved $2.00 (as of
2026-11-07), so the remaining allowance is $1.995511 - check first,
spend may have moved:

```bash
python3 scripts/budget_ledger.py status   # on the machine that already spent
# on any new machine, provision ONLY the remaining allowance:
python3 scripts/budget_ledger.py init --cap 1.995511 --note 'remaining of approved $2'
```

Running on several machines at once: either keep the gate on ONE
machine under the single shared cap, or explicitly PARTITION the
remaining allowance between them (e.g. $1.00 on the laptop and
$0.995511 on the desktop). Never provision the same remaining amount on
two machines - that would double the approved total.

Before each paid call the hook asks the provider's free token-counting
endpoint for the input tokens of the exact request and reserves that
count plus a safety margin (10% + 16 tokens - the provider calls its
count an estimate that can differ by a small amount) at the cache-write
rate plus full max output; if counting is unavailable, the paid call is
refused (the hook still fails open for your work). This is conservative
stop-before-cap accounting, not an exact billing guarantee.

## What it costs and stores

- Decision: local, free, ~0.3 s, offline. No network.
- Question drafting (only on flagged prompts): one Haiku 5.5 call,
  ~1,500 tokens in worst case. Hard spend cap via
  `PROMPTIMIZE_BUDGET_USD` (default `$2.00`); the cap fails closed (calls
  stop) while the hook fails open (your prompts always go through).
- Logs usage totals only (`~/.promptimize/usage.jsonl`): tokens,
  estimated cost, latency. **No prompt text is logged.** Your prompt
  leaves the machine only inside the question-drafting call, only when
  flagged.
- `PROMPTIMIZE_DISABLED=1` bypasses the gate entirely.

## Modes and latency

`PROMPTIMIZE_MODE`:
- `hybrid` (default): LAYA decides locally, Haiku writes the question on
  flags. Note the hook is a fresh process per prompt, so LAYA's model
  load (~4-7 s on first run, ~4 s warm) is paid on every prompt; a
  resident `laya-serve` daemon is the follow-up if that feels slow.
- `haiku`: Haiku makes the decision AND writes the question (~1 s, one
  API call per prompt, needs network; prompts leave the machine on every
  prompt, not only flagged ones). This is the comparison arm from the
  frozen-validation results - see the main README.

## Tuning

`PROMPTIMIZE_THRESHOLD` (default `0.35`): lower values ask less often.
On the frozen validation split (n=44), 0.35 traded over-asking 0.238 for
under-asking 0.348, while 0.50 gave 0.524 / 0.130. This is a starting
posture for dogfooding, not a validated default.
