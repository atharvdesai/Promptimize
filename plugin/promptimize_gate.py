#!/usr/bin/env python3
"""Promptimize gate: Claude Code UserPromptSubmit hook (nudge mode).

Reads the hook JSON on stdin, runs the local LAYA decision model (ONNX,
no network, ~0.3 s on a laptop CPU), and when the prompt scores below the
execute-as-is threshold, asks Haiku to draft ONE clarifying question and
injects it as additionalContext so Claude asks it natively in the
conversation. The hook never blocks: any error, timeout risk, or
exhausted budget exits 0 with no output and the prompt goes through
unchanged.

Privacy: prompt text is never logged. It leaves the machine only in the
single Haiku question-drafting call, and only when LAYA has flagged the
prompt. Usage totals (tokens, estimated cost, latency) go to
~/.promptimize/usage.jsonl; spend accounting to the budget ledger.

Config (env):
  PROMPTIMIZE_LAYA_ONNX       path to model.onnx (default: plugin/model.onnx)
  PROMPTIMIZE_DECIDER         "head", "noul", or "auto" (default auto:
                              head when plugin/gate_head.json exists, else noul)
  PROMPTIMIZE_GATE_HEAD       path to the head artifact (default: plugin/gate_head.json)
  PROMPTIMIZE_THRESHOLD       ask when P(can_execute) < this (defaults: 0.95
                              for the head decider, 0.35 for noul)
  PROMPTIMIZE_MODE            "hybrid" (local decider scores, Haiku writes the
                              question; default) or "haiku" (Haiku does
                              both - the comparison arm; no local model)
  PROMPTIMIZE_ANTHROPIC_KEY_FILE  file containing the Anthropic API key
  PROMPTIMIZE_BUDGET_USD      total spend cap, fail closed (default 2.00)
  PROMPTIMIZE_DISABLED        set to 1 to bypass the gate entirely

Deciders. "noul" is LAYA's zero-shot head with the instructions below:
measured val acc 0.682, over-asking 0.238 / under-asking 0.348 at 0.35
(n=44, frozen split). "head" is the trained numpy logistic head from
trial T5 on the same frozen LAYA embeddings with input normalization
(gate_normalize.normalize, applied identically at train and inference
time): measured val acc 0.977, over 0.048 / under 0.000 at p_ambiguous
0.05 (artifact as shipped, n=44, frozen split). Both figures are one validation split reused across trials -
directional, not validated on real usage. The head's measured blind
spot is LONG prompts (the probe failure zone, flagged >= 200 chars as
a heuristic, not a validated boundary): ambiguous-recall 0/8 on probe
families 1+2 (trials P2/P3 - replicates on a cross-encoder, so it is
likely dataset surface correlation, not the encoder). Decisions in the
blind-spot zone are logged with blind_spot=true so dogfooding data can
be stratified on it.

Noul threshold note: 0.35 comes from the frozen-validation sweep
(n=44): over-asking 0.238 / under-asking 0.348 there, versus
0.524 / 0.130 at 0.50. It is a dogfooding posture chosen to bias
against false interruptions, not a value validated on real usage.
Tune after real use.

Claude Code wiring (settings.json):
  "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command",
    "command": "python3 /path/to/repo/plugin/promptimize_gate.py"}]}]}
"""

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from gate_normalize import normalize, in_measured_blind_spot

USAGE_LOG = os.path.expanduser("~/.promptimize/usage.jsonl")
STATE_TMPL = 'A user gave a coding agent this request: "%s"'
NOUL_INSTRUCTIONS = (
    "Can a coding agent execute this request correctly without asking "
    "the user a clarifying question?"
)
MIN_PROMPT_CHARS = 25
HEAD_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_head.json")

_agent = None
_head = None
_head_path = None


def _log(kind, **fields):
    # Usage totals only. Prompt text must never reach this file.
    fields.pop("prompt", None)
    entry = {"ts": time.time(), "kind": kind}
    entry.update(fields)
    try:
        os.makedirs(os.path.dirname(USAGE_LOG), exist_ok=True)
        with open(USAGE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def decider_name():
    """Which local decider scores prompts. auto: head if the artifact exists."""
    d = os.environ.get("PROMPTIMIZE_DECIDER", "auto")
    if d == "auto":
        d = "head" if os.path.exists(os.environ.get("PROMPTIMIZE_GATE_HEAD", HEAD_PATH_DEFAULT)) else "noul"
    return d


def default_threshold(decider):
    # head: ask when P(can_execute) < 1 - 0.05 (T5 selected p_ambiguous thr).
    # noul: 0.35 dogfooding posture (see header note).
    return "0.95" if decider == "head" else "0.35"


def _load_agent():
    """Load the local LAYA checkpoint once. Raises on failure."""
    global _agent
    if _agent is None:
        import onnxruntime as ort

        real_session = ort.InferenceSession

        def memory_safe_session(path, sess_options=None, providers=None, **kw):
            # Default session options OOM below ~2 GB free RAM; arenas,
            # pattern pre-allocation and graph passes all spike the peak.
            so = sess_options or ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            return real_session(path, sess_options=so, providers=["CPUExecutionProvider"], **kw)

        ort.InferenceSession = memory_safe_session
        os.environ.setdefault("USE_TF", "0")
        import laya

        onnx_path = os.environ.get(
            "PROMPTIMIZE_LAYA_ONNX",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx"),
        )
        _agent = laya.load(
            "codenamev/laya-onnx", backend="onnx", subfolder="english", onnx_path=onnx_path
        )
    return _agent


def decide(prompt):
    """P(can_execute). Dispatches to the selected decider. Raises on failure."""
    if decider_name() == "head":
        return 1.0 - score_head(embed_prompt(prompt), load_head())
    return decide_noul(prompt)


def decide_noul(prompt):
    """P(can_execute) from LAYA's zero-shot noul head."""
    agent = _load_agent()
    result = agent.predict(
        STATE_TMPL % prompt, {"can_execute": {"type": "noul", "instructions": NOUL_INSTRUCTIONS}}
    )
    return float(result["answers"]["can_execute"]["noul"])


def load_head(path=None):
    """The trained T5 logistic head artifact (small numpy head, JSON)."""
    global _head, _head_path
    if path is None:
        path = os.environ.get("PROMPTIMIZE_GATE_HEAD", HEAD_PATH_DEFAULT)
    if _head is None or _head_path != path:
        with open(path) as f:
            artifact = json.load(f)
        if artifact.get("format") != "promptimize-gate-head/v1":
            raise ValueError("unsupported gate head format: %r" % artifact.get("format"))
        _head, _head_path = artifact, path
    return _head


def embed_prompt(prompt):
    """Frozen LAYA embedding of the NORMALIZED prompt (T5 contract: the
    same gate_normalize.normalize used to train the head)."""
    import numpy as np
    from laya.onnx_agent import _collate_items_numpy

    agent = _load_agent()
    question = {"can_execute": {"type": "noul", "instructions": NOUL_INSTRUCTIONS}}
    internal = {qid: agent._to_internal(q) for qid, q in question.items()}
    ids = list(question.keys())
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = agent._encode_state(STATE_TMPL % normalize(prompt), ids, internal, max_len, head_max_len)
    inputs = _collate_items_numpy([items], agent.tok.pad_token_id)
    hidden = agent.session.run(["last_hidden_state"], inputs)[0][0]  # (seq, dim)
    mask = inputs["attention_mask"][0].astype(np.float64)  # (seq,)
    return (hidden * mask[:, None]).sum(axis=0) / max(mask.sum(), 1.0)


def score_head(emb, head):
    """p_ambiguous = sigmoid(standardize(emb) @ w + b). Pure Python: the
    head is ~1k floats, no numpy needed on the hot path."""
    w, mu, sd, b = head["w"], head["mu"], head["sd"], head["b"]
    if len(emb) != len(w):
        raise ValueError("embedding dim %d != head dim %d" % (len(emb), len(w)))
    z = b
    for e, m, s, wi in zip(emb, mu, sd, w):
        z += (float(e) - m) / s * wi
    # numerically stable sigmoid
    if z >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -z))
    ez = pow(2.718281828459045, z)
    return ez / (1.0 + ez)


def write_question(prompt, ledger):
    """Draft the clarifying question via Haiku (budget-gated)."""
    from haiku_gate import call_gate

    result, usage = call_gate(prompt, ledger, purpose="hook-question", max_tokens=200)
    return result.get("clarifying_question"), usage


def run(stdin_text):
    data = json.loads(stdin_text or "{}")
    prompt = (data.get("prompt") or "").strip()
    if os.environ.get("PROMPTIMIZE_DISABLED") == "1":
        return None
    if len(prompt) < MIN_PROMPT_CHARS or prompt.startswith("/"):
        return None

    decider = decider_name()
    threshold = float(os.environ.get("PROMPTIMIZE_THRESHOLD", default_threshold(decider)))
    mode = os.environ.get("PROMPTIMIZE_MODE", "hybrid")
    t0 = time.time()

    from budget_ledger import BudgetExhausted, BudgetLedger

    ledger = BudgetLedger()

    if mode == "haiku":
        # Comparison arm: Haiku makes the decision AND writes the question.
        try:
            from haiku_gate import call_gate

            result, usage = call_gate(prompt, ledger, purpose="hook-decide+question", max_tokens=200)
        except BudgetExhausted:
            _log("question_skipped", reason="budget_exhausted", mode=mode)
            return None
        except Exception as e:
            _log("question_failed", error=type(e).__name__, mode=mode)
            return None
        _log(
            "decision",
            mode=mode,
            can_execute=result["can_execute"],
            latency_ms=int((time.time() - t0) * 1000),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        )
        if result["can_execute"] or not result.get("clarifying_question"):
            return None
        question = result["clarifying_question"]
    else:
        prob = decide(prompt)
        decision_ms = int((time.time() - t0) * 1000)
        _log("decision", mode=mode, decider=decider, prob=round(prob, 4),
             threshold=threshold, latency_ms=decision_ms,
             blind_spot=in_measured_blind_spot(prompt))
        if prob >= threshold:
            return None
        try:
            question, usage = write_question(prompt, ledger)
        except BudgetExhausted:
            _log("question_skipped", reason="budget_exhausted", mode=mode)
            return None
        except Exception as e:
            _log("question_failed", error=type(e).__name__, mode=mode)
            return None
        if not question:
            # Haiku's second opinion judged the prompt executable - the
            # disagreement rate is dogfooding signal, so record it.
            _log("question_declined", mode=mode, latency_ms=int((time.time() - t0) * 1000))
            return None
        _log(
            "question",
            mode=mode,
            latency_ms=int((time.time() - t0) * 1000),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        )
    if mode == "haiku":
        finding = "the Promptimize gate model judged it not executable as-is"
    else:
        finding = "a local decision model scored it below the execute-as-is threshold"
    context = (
        "Promptimize pre-flight gate finding: this coding request may be ambiguous ("
        + finding
        + "). Suggested clarifying question to ask the user before acting on the request: "
        + question
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }


def main():
    try:
        out = run(sys.stdin.read())
    except Exception as e:  # fail open, always
        _log("error", error=type(e).__name__)
        out = None
    if out is not None:
        sys.stdout.write(json.dumps(out))
    sys.exit(0)


if __name__ == "__main__":
    main()
