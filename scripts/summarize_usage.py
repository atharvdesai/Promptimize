#!/usr/bin/env python3
"""Summarize Promptimize dogfooding telemetry (~/.promptimize/usage.jsonl).

The hook logs usage EVENTS ONLY - never prompt text - so dogfooding can
answer "is the gate helping?" without reading anyone's prompts. This
script aggregates those events into the numbers that matter for the
dogfooding decision:

  python3 scripts/summarize_usage.py [path-to-usage.jsonl] [--json]

- decision volume and ask rate (recomputed per row from the recorded
  prob/threshold with the decider's own comparison convention)
- answer path: warm server vs in-process fallback, with decision latency
  percentiles
- blind-spot share (the >=200-char heuristic flag from trials P2/P3)
- question pipeline: drafted / declined by Haiku / skipped on budget /
  failed, with token totals and estimated cost (same rate card as the
  budget ledger)
- errors by exception type, malformed log lines
- privacy self-check: any entry carrying a "prompt" field is a
  regression in the hook's no-prompt-logging rule and is surfaced
  loudly (count only - the text itself is never read or printed)

Per-day rows keep trends visible. Zero network, zero cost.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from budget_ledger import cost_usd

LOG_DEFAULT = os.path.expanduser("~/.promptimize/usage.jsonl")

# Fields an entry may legitimately carry. Anything else is reported as
# an unknown field (schema drift detector); "prompt" specifically is a
# privacy regression.
KNOWN_FIELDS = {
    "ts", "kind", "mode", "decider", "via", "prob", "threshold",
    "latency_ms", "blind_spot", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "can_execute", "reason",
    "error",
}


def asked(row):
    """Recompute the ask decision from recorded fields, using the
    decider's own convention (head: prob <= threshold; noul: <)."""
    if "prob" not in row or "threshold" not in row:
        return None
    if row.get("decider") == "head":
        return row["prob"] <= row["threshold"]
    return row["prob"] < row["threshold"]


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return s[i]


def summarize(path):
    rows, malformed = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1

    decisions = [r for r in rows if r.get("kind") == "decision"]
    questions = [r for r in rows if r.get("kind") == "question"]
    declined = [r for r in rows if r.get("kind") == "question_declined"]
    skipped = [r for r in rows if r.get("kind") == "question_skipped"]
    failed = [r for r in rows if r.get("kind") == "question_failed"]
    errors = [r for r in rows if r.get("kind") == "error"]

    asks = [a for a in (asked(r) for r in decisions) if a is not None]
    via = Counter(r.get("via", "unknown") for r in decisions)
    lat = [r["latency_ms"] for r in decisions if "latency_ms" in r]
    blind = sum(1 for r in decisions if r.get("blind_spot"))

    tok = {k: sum(r.get(k, 0) for r in questions)
           for k in ("input_tokens", "output_tokens",
                     "cache_read_tokens", "cache_write_tokens")}
    est_cost = cost_usd(**tok) if any(tok.values()) else 0.0

    per_day = defaultdict(lambda: {"decisions": 0, "asks": 0, "questions": 0})
    for r in decisions:
        day = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
        per_day[day]["decisions"] += 1
        a = asked(r)
        if a:
            per_day[day]["asks"] += 1
    for r in questions:
        day = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
        per_day[day]["questions"] += 1

    privacy_violations = sum(1 for r in rows if "prompt" in r)
    unknown_fields = Counter(
        k for r in rows for k in r if k not in KNOWN_FIELDS and k != "prompt")

    return {
        "path": path,
        "window": {
            "first": min((r.get("ts", 0) for r in rows), default=None),
            "last": max((r.get("ts", 0) for r in rows), default=None),
        },
        "events": len(rows),
        "malformed_lines": malformed,
        "decisions": {
            "n": len(decisions),
            "ask_rate": round(sum(asks) / len(asks), 3) if asks else None,
            "via": dict(via),
            "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95),
                           "max": max(lat) if lat else None},
            "blind_spot_share": round(blind / len(decisions), 3) if decisions else None,
        },
        "questions": {
            "drafted": len(questions),
            "declined_by_haiku": len(declined),
            "skipped_budget": len(skipped),
            "failed": len(failed),
            "tokens": tok,
            "est_cost_usd": round(est_cost, 6),
        },
        "errors": dict(Counter(r.get("error", "?") for r in errors)),
        "per_day": {d: per_day[d] for d in sorted(per_day)},
        "privacy": {
            "prompt_field_violations": privacy_violations,
            "unknown_fields": dict(unknown_fields),
        },
    }


def render(s):
    out = []
    w = s["window"]
    if w["first"]:
        out.append("window: %s .. %s" % (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(w["first"])),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(w["last"]))))
    out.append("events: %d (%d malformed lines skipped)" % (s["events"], s["malformed_lines"]))
    d = s["decisions"]
    out.append("decisions: %d, ask rate %s, blind-spot share %s" % (
        d["n"],
        "%.1f%%" % (100 * d["ask_rate"]) if d["ask_rate"] is not None else "n/a",
        "%.1f%%" % (100 * d["blind_spot_share"]) if d["blind_spot_share"] is not None else "n/a"))
    out.append("  via: %s" % (", ".join("%s=%d" % kv for kv in sorted(d["via"].items())) or "n/a"))
    lat = d["latency_ms"]
    out.append("  decision latency ms: p50=%s p95=%s max=%s" % (lat["p50"], lat["p95"], lat["max"]))
    q = s["questions"]
    out.append("questions: %d drafted, %d declined by Haiku, %d skipped (budget), %d failed" % (
        q["drafted"], q["declined_by_haiku"], q["skipped_budget"], q["failed"]))
    out.append("  tokens: %s, est cost $%.6f" % (
        ", ".join("%s=%d" % kv for kv in sorted(q["tokens"].items())), q["est_cost_usd"]))
    if s["errors"]:
        out.append("errors: %s" % ", ".join("%s=%d" % kv for kv in sorted(s["errors"].items())))
    for day, row in s["per_day"].items():
        out.append("  %s: %d decisions, %d asks, %d questions" % (
            day, row["decisions"], row["asks"], row["questions"]))
    p = s["privacy"]
    if p["prompt_field_violations"]:
        out.append("PRIVACY REGRESSION: %d log entries contain a prompt field - "
                   "the hook must never log prompt text" % p["prompt_field_violations"])
    if p["unknown_fields"]:
        out.append("unknown fields (schema drift): %s" % ", ".join(
            "%s=%d" % kv for kv in sorted(p["unknown_fields"].items())))
    return "\n".join(out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else LOG_DEFAULT
    if not os.path.exists(path):
        print("no usage log at %s yet - the hook writes it on first decision" % path)
        sys.exit(0)
    s = summarize(path)
    if "--json" in sys.argv:
        print(json.dumps(s, indent=1))
    else:
        print(render(s))


if __name__ == "__main__":
    main()
