"""Fail-closed API budget ledger for Promptimize experiments.

Reserve-before-call accounting: every paid call first reserves its
worst-case cost (estimated input + max output at UNCACHED list rates).
After the call, the reservation settles to the actual billed token usage;
if the call fails, the full reservation is charged (errors can still bill
partial usage, and retries/errors must count toward the cap). A
reservation that would push committed spend past the cap is refused, so
the budget fails closed - while the hook itself always fails open for the
user's own work.

The ledger records usage totals only. Prompt text is never written here.

Rates pinned from https://docs.claude.com/en/docs/about-claude/pricing
(fetched 2026-11-07), Claude Haiku 5.5, prompts <= 100,000 tokens.
Re-pin from that page before relying on them later.
"""

import json
import os
import time
import uuid

MODEL = "claude-haiku-5-5"
RATES_USD_PER_MTOK = {
    "input": 0.10,
    "output": 0.50,
    "cache_write_5m": 0.125,
    "cache_read": 0.01,
}
DEFAULT_CAP_USD = 2.00


class BudgetExhausted(RuntimeError):
    """Raised when a reservation would exceed the cap. Callers fail open."""


def cost_usd(input_tokens=0, output_tokens=0, cache_write_tokens=0, cache_read_tokens=0):
    r = RATES_USD_PER_MTOK
    return (
        input_tokens * r["input"]
        + output_tokens * r["output"]
        + cache_write_tokens * r["cache_write_5m"]
        + cache_read_tokens * r["cache_read"]
    ) / 1_000_000


class BudgetLedger:
    def __init__(self, path=None, cap_usd=None):
        self.path = (
            path
            or os.environ.get("PROMPTIMIZE_BUDGET_FILE")
            or os.path.expanduser("~/.promptimize/budget.json")
        )
        self.cap_usd = float(
            cap_usd
            if cap_usd is not None
            else os.environ.get("PROMPTIMIZE_BUDGET_USD", DEFAULT_CAP_USD)
        )
        self._state = self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                state = json.load(f)
            state.setdefault("spent_usd", 0.0)
            state.setdefault("reserved", {})
            state.setdefault("events", [])
            return state
        except (OSError, json.JSONDecodeError):
            return {"spent_usd": 0.0, "reserved": {}, "events": []}

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def _event(self, kind, purpose, **fields):
        entry = {"ts": time.time(), "kind": kind, "purpose": purpose, "model": MODEL}
        entry.update(fields)
        self._state["events"].append(entry)

    def committed_usd(self):
        reserved = sum(r["reserve_usd"] for r in self._state["reserved"].values())
        return self._state["spent_usd"] + reserved

    def remaining_usd(self):
        return self.cap_usd - self.committed_usd()

    def reserve(self, purpose, est_input_tokens, max_output_tokens):
        """Reserve worst-case (uncached) cost. Returns a reservation id."""
        reserve = cost_usd(est_input_tokens, max_output_tokens)
        if reserve > self.remaining_usd() + 1e-12:
            self._event(
                "refused",
                purpose,
                needed_usd=round(reserve, 6),
                remaining_usd=round(self.remaining_usd(), 6),
            )
            self._save()
            raise BudgetExhausted(
                "reserving $%.6f would exceed the $%.2f cap (%.6f remaining)"
                % (reserve, self.cap_usd, self.remaining_usd())
            )
        rid = uuid.uuid4().hex[:16]
        self._state["reserved"][rid] = {
            "reserve_usd": reserve,
            "purpose": purpose,
            "est_input_tokens": est_input_tokens,
            "max_output_tokens": max_output_tokens,
            "ts": time.time(),
        }
        self._event("reserve", purpose, rid=rid, reserve_usd=round(reserve, 6))
        self._save()
        return rid

    def settle(self, rid, usage):
        """Replace a reservation with actual billed usage. Returns actual cost."""
        self._state["reserved"].pop(rid, None)
        actual = cost_usd(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        )
        self._state["spent_usd"] += actual
        self._event(
            "settle",
            self._purpose_of(rid),
            actual_usd=round(actual, 6),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            spent_total_usd=round(self._state["spent_usd"], 6),
        )
        self._save()
        return actual

    def fail(self, rid, error_kind):
        """Charge the full reservation on a failed call (conservative)."""
        r = self._state["reserved"].pop(rid, None)
        charged = r["reserve_usd"] if r else 0.0
        self._state["spent_usd"] += charged
        self._event(
            "fail",
            r["purpose"] if r else "unknown",
            charged_usd=round(charged, 6),
            error=error_kind,
            spent_total_usd=round(self._state["spent_usd"], 6),
        )
        self._save()
        return charged

    def _purpose_of(self, rid):
        # purpose was already popped in settle; look back at events for accuracy
        for entry in reversed(self._state["events"]):
            if entry["kind"] == "reserve" and entry.get("rid") == rid:
                return entry["purpose"]
        return "unknown"
