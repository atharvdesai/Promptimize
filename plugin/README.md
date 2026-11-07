# Promptimize gate (Claude Code hook, experimental)

A `UserPromptSubmit` hook that watches your prompts before Claude Code
processes them. A local decision model (LAYA, ONNX, on your machine)
scores whether the request is executable as-is; when it scores below the
threshold, Haiku drafts one clarifying question and the hook injects it
as context so Claude asks you natively. It never blocks your prompt.

**Status: experimental, dogfooding only. Tested in a Linux sandbox, not
yet on macOS. Decision quality measured on the frozen validation split
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

Set `PROMPTIMIZE_ANTHROPIC_KEY_FILE` to a file containing your Anthropic
API key (billed to your API account, separate from a Claude
subscription). The key file is read, never logged.

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
