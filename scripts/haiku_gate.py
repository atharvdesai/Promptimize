"""Shared Haiku client for the Promptimize gate.

Used by the few-shot baseline runner (decision + question per dataset row)
and by the Claude Code hook (question-writing only, after a local LAYA
flag). Every call goes through BudgetLedger, which fails closed at the
cap; callers decide how to fail open.

The API key is read from a file whose path comes from
PROMPTIMIZE_ANTHROPIC_KEY_FILE (fallback: ANTHROPIC_KEY_FILE). The key is
never logged, printed, or written to this repo. Only usage totals reach
the ledger; prompt text is never logged.

Model and rates: claude-haiku-5-5 at $0.10/MTok input, $0.50/MTok output,
$0.125/MTok 5m cache write, $0.01/MTok cache read (prompts <= 100k),
pinned 2026-11-07 from
https://docs.claude.com/en/docs/about-claude/pricing
Model ID from https://docs.claude.com/en/docs/about-claude/models/overview
Token counting endpoint: free to use, separate rate limits, and billing
reflects only request content (not system-added tokens), per
https://docs.claude.com/en/docs/build-with-claude/token-counting (fetched
2026-11-07) - so its count for the exact request body is the reserved
input bound. If counting is unavailable, the paid call is refused.
Prompt-caching minimum for Haiku 5.5 is 512 tokens
(https://docs.claude.com/en/docs/build-with-claude/prompt-caching), so
this static few-shot prefix is cacheable; reserve math never assumes it.
"""

import json
import os
import re
import urllib.error
import urllib.request

from budget_ledger import MODEL, BudgetLedger

API_URL = "https://api.anthropic.com/v1/messages"
COUNT_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_VERSION = "2026-10-21"
DEFAULT_MAX_TOKENS = 200  # thinking is disabled; text-only output stays well under this

GATE_INSTRUCTIONS = """You are the Promptimize gate: you decide whether a coding agent can execute a user's request as-is, or whether exactly one clarifying question is needed first.

Labeling rule:
- A request is EXECUTABLE (can_execute=true) when the agent can discover whatever is missing by itself: run the tests, grep the repo, read the config, follow existing conventions.
- A request is AMBIGUOUS (can_execute=false) ONLY when the missing information exists solely in the user's head: intent, target environment, a business rule, an error message the agent cannot see.
- Over-asking is the main failure mode. When borderline, answer can_execute=true.

Keep the clarifying_question under 40 words.

Output: strict JSON only, no markdown, no explanation:
{"can_execute": true} 
or
{"can_execute": false, "clarifying_question": "<one specific question>", "ambiguity_type": "<one of: missing_context, missing_requirement, missing_scope, missing_file, missing_technology, missing_constraint, multiple_interpretations, unclear_goal, missing_component, missing_acceptance_criteria>"}

Examples:
Request: Convert src/utils/dates.js from moment.js to date-fns and update the imports everywhere it's used
{"can_execute": true}

Request: The tests in tests/test_auth.py started failing after my last commit. Run them, figure out what broke, and fix it.
{"can_execute": true}

Request: Add a healthcheck endpoint GET /healthz to the FastAPI app in app/main.py that returns 200 with {"status": "ok"}
{"can_execute": true}

Request: make it faster
{"can_execute": false, "clarifying_question": "What specifically feels slow - page load, an API endpoint, a database query, or the build/test cycle?", "ambiguity_type": "missing_scope"}

Request: fix the bug
{"can_execute": false, "clarifying_question": "What's the incorrect behavior you're seeing, and is there an error message or a way to reproduce it?", "ambiguity_type": "missing_context"}

Request: update the api
{"can_execute": false, "clarifying_question": "Which endpoint needs updating, and what should change about it - the response shape, validation, behavior, or something else?", "ambiguity_type": "unclear_goal"}
"""

# Few-shot exemplars above come from dataset/splits/train.jsonl. The frozen
# splits (seed 20261003) are group-aware, so train rows are group-disjoint
# from the validation rows this prompt is evaluated on.


class GateCallError(RuntimeError):
    """The API call failed after budget accounting (charged as reserved)."""


class GateParseError(RuntimeError):
    """The response was not the strict JSON the gate requires."""


def _read_key(key_file=None):
    """Read the API key without ever printing or logging it.

    Preferred: PROMPTIMIZE_ANTHROPIC_KEY_CMD - a command whose stdout is
    the key, e.g. macOS Keychain:
      security find-generic-password -s promptimize-anthropic -w
    Fallback: a key FILE (PROMPTIMIZE_ANTHROPIC_KEY_FILE) that must be
    owner-only (mode 600/400); anything group/world-readable is refused.
    """
    cmd = os.environ.get("PROMPTIMIZE_ANTHROPIC_KEY_CMD")
    if cmd:
        import subprocess

        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0 or not out.stdout.strip():
            raise GateCallError("PROMPTIMIZE_ANTHROPIC_KEY_CMD returned no key")
        return out.stdout.strip()
    path = (
        key_file
        or os.environ.get("PROMPTIMIZE_ANTHROPIC_KEY_FILE")
        or os.environ.get("ANTHROPIC_KEY_FILE")
    )
    if not path:
        raise GateCallError(
            "no key source: set PROMPTIMIZE_ANTHROPIC_KEY_CMD (keychain) "
            "or PROMPTIMIZE_ANTHROPIC_KEY_FILE (mode-600 file)"
        )
    path = os.path.expanduser(path)
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise GateCallError(
            "key file %s is group/world-readable (mode %o); chmod 600 it" % (path, mode)
        )
    with open(path) as f:
        return f.read().strip()


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    return json.loads(text)


def _post_json(url, body, key, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def call_gate(user_prompt, ledger, purpose, key_file=None, max_tokens=DEFAULT_MAX_TOKENS, timeout=25):
    """Run one gated decision+question call. Returns (parsed, usage).

    Raises BudgetExhausted before any network activity when the reserve
    would exceed the cap, GateCallError on transport/HTTP failure, and
    GateParseError when the model did not answer with the required JSON.
    """
    system_text = GATE_INSTRUCTIONS
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        # Deterministic JSON output: with thinking left on, Haiku 5.5 can spend
        # the whole max_tokens budget on a thinking block and truncate the JSON.
        "thinking": {"type": "disabled"},
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "Request: " + user_prompt}],
    }
    key = _read_key(key_file)

    # Exact input-token bound: the provider's count_tokens endpoint on the
    # exact request content (free; billing reflects only content tokens).
    # No bound, no paid call - fail closed.
    count_body = {
        "model": MODEL,
        "system": [{"type": "text", "text": system_text}],
        "messages": body["messages"],
    }
    try:
        count_resp = _post_json(COUNT_URL, count_body, key, timeout)
        est_input = int(count_resp["input_tokens"])
    except Exception as e:
        raise GateCallError(
            "token counting unavailable (%s); refusing paid call" % type(e).__name__
        ) from e

    rid = ledger.reserve(purpose, est_input, max_tokens)

    try:
        data = _post_json(API_URL, body, key, timeout)
    except urllib.error.HTTPError as e:
        ledger.fail(rid, "http_%s" % e.code)
        raise GateCallError("HTTP %s from Messages API" % e.code) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        ledger.fail(rid, type(e).__name__)
        raise GateCallError("Messages API call failed: %s" % type(e).__name__) from e

    usage = data.get("usage", {})
    ledger.settle(rid, usage)
    try:
        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        parsed = _extract_json(text)
        result = {
            "can_execute": bool(parsed["can_execute"]),
            "clarifying_question": parsed.get("clarifying_question"),
            "ambiguity_type": parsed.get("ambiguity_type"),
        }
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise GateParseError("unparseable gate response") from e
    return result, usage
