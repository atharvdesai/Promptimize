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

Safeguards:
- Atomic across processes: every mutation holds an exclusive flock on a
  sibling lock file and re-reads state inside the lock, so concurrent
  hook processes cannot both spend the same headroom.
- Corrupt ledger: loading a damaged file raises CorruptLedger (fail
  closed - no calls) instead of silently resetting spend to zero.
- Explicit provisioning: with neither state file present the ledger
  refuses every paid call (LedgerNotProvisioned). The approved total is
  per experiment, not per machine: provision a fresh machine with the
  REMAINING allowance only (approved total minus spend already accounted
  in the previous ledger / Anthropic console), via
  `python3 scripts/budget_ledger.py init --cap <remaining_usd>`.
- Reserve math uses the cache-WRITE rate for input tokens (higher than
  plain input), so caching can only ever cost less than reserved.
- settle() with missing usage data charges the full reservation; a
  response without usage counts as unknown billable usage.

Rates pinned from https://docs.claude.com/en/docs/about-claude/pricing
(fetched 2026-11-07), Claude Haiku 5.5, prompts <= 100,000 tokens.
Re-pin from that page before relying on them later.
"""

import fcntl
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


class CorruptLedger(RuntimeError):
    """The ledger file exists but is unreadable. Fail closed, keep the file."""


class LedgerNotProvisioned(RuntimeError):
    """No ledger state on this machine and no explicit provisioning yet.

    The cap is a total across the whole experiment, not per install, so a
    fresh machine must be provisioned with the remaining allowance (see
    module docstring) before any paid call is allowed."""


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

    def _read_state(self):
        """Read state from disk. Corrupt data fails closed, never resets."""
        try:
            with open(self.path) as f:
                state = json.load(f)
        except FileNotFoundError:
            bak = self.path + ".bak"
            if os.path.exists(bak):
                # Main file lost (reinstall, cleanup): restore the mirror
                # rather than resetting the approved remaining budget.
                with open(bak) as f:
                    state = json.load(f)
            else:
                return None  # unprovisioned - callers fail closed
        except json.JSONDecodeError as e:
            raise CorruptLedger(
                "budget ledger %s is corrupt; refusing paid calls until fixed" % self.path
            ) from e
        state.setdefault("spent_usd", 0.0)
        state.setdefault("reserved", {})
        state.setdefault("events", [])
        return state

    def _load(self):
        # None when unprovisioned; read methods and mutations fail closed.
        return self._read_state()

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
        try:
            with open(self.path + ".bak", "w") as f:
                json.dump(self._state, f, indent=1, sort_keys=True)
        except OSError:
            pass

    def _locked(self, fn):
        """Run fn(self) holding an exclusive cross-process lock, state fresh."""
        lock_path = self.path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._state = self._read_state()
                if self._state is None:
                    raise LedgerNotProvisioned(
                        "no budget ledger at %s: provision the remaining "
                        "allowance first (scripts/budget_ledger.py init)" % self.path
                    )
                result = fn(self)
                self._save()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return result

    def set_cap(self, new_cap_usd, note=""):
        """Record a cap change with provenance. Total across the experiment."""
        def _do(ledger):
            old = ledger._state.get("cap_usd", ledger.cap_usd)
            ledger._state["cap_usd"] = float(new_cap_usd)
            ledger.cap_usd = float(new_cap_usd)
            ledger._event(
                "cap_update",
                "cap change",
                old_cap_usd=old,
                new_cap_usd=float(new_cap_usd),
                note=note,
            )
            return float(new_cap_usd)
        return self._locked(_do)

    def _require_state(self):
        if self._state is None:
            raise LedgerNotProvisioned(
                "no budget ledger at %s: provision first" % self.path
            )
        return self._state

    def _event(self, kind, purpose, **fields):
        entry = {"ts": time.time(), "kind": kind, "purpose": purpose, "model": MODEL}
        entry.update(fields)
        self._state["events"].append(entry)

    def committed_usd(self):
        state = self._require_state()
        reserved = sum(r["reserve_usd"] for r in state["reserved"].values())
        return state["spent_usd"] + reserved

    def remaining_usd(self):
        state = self._require_state()
        return state.get("cap_usd", self.cap_usd) - self.committed_usd()

    def provision(self, note=""):
        """Explicitly initialize this machine's allowance. Fails if state
        already exists (use the existing ledger; never double-provision)."""
        if os.path.exists(self.path) or os.path.exists(self.path + ".bak"):
            raise CorruptLedger(
                "ledger already exists at %s; refusing to re-provision" % self.path
            )
        self._state = {
            "spent_usd": 0.0,
            "reserved": {},
            "cap_usd": self.cap_usd,
            "events": [
                {
                    "ts": time.time(),
                    "kind": "provision",
                    "purpose": note or "manual init",
                    "model": MODEL,
                    "cap_usd": self.cap_usd,
                }
            ],
        }
        self._save()
        return self

    def reserve(self, purpose, est_input_tokens, max_output_tokens):
        """Reserve worst-case cost. Returns a reservation id.

        Worst case = input tokens at the 5m cache-WRITE rate (higher than
        plain input) plus max output tokens, so a cache-eligible call can
        only cost less than reserved. Atomic across processes.
        """
        def _do(ledger):
            reserve = cost_usd(output_tokens=max_output_tokens,
                               cache_write_tokens=est_input_tokens)
            cap = ledger._state.get("cap_usd", ledger.cap_usd)
            remaining = cap - ledger.committed_usd()
            if reserve > remaining + 1e-12:
                ledger._event(
                    "refused",
                    purpose,
                    needed_usd=round(reserve, 6),
                    remaining_usd=round(remaining, 6),
                )
                ledger._save()  # keep the refusal on the audit trail
                raise BudgetExhausted(
                    "reserving $%.6f would exceed the $%.2f cap (%.6f remaining)"
                    % (reserve, cap, remaining)
                )
            rid = uuid.uuid4().hex[:16]
            ledger._state["reserved"][rid] = {
                "reserve_usd": reserve,
                "purpose": purpose,
                "est_input_tokens": est_input_tokens,
                "max_output_tokens": max_output_tokens,
                "ts": time.time(),
            }
            ledger._event("reserve", purpose, rid=rid, reserve_usd=round(reserve, 6))
            return rid
        return self._locked(_do)

    def settle(self, rid, usage):
        """Replace a reservation with actual billed usage. Returns actual cost.

        usage comes from the provider response. If it is missing or has no
        token fields, the bill is unknowable - charge the full reservation
        (conservative) instead of recording zero.
        """
        if not usage or not any(
            k in usage
            for k in ("input_tokens", "output_tokens",
                      "cache_creation_input_tokens", "cache_read_input_tokens")
        ):
            self.fail(rid, "missing_usage")
            return self._state["events"][-1].get("charged_usd", 0.0)

        def _do(ledger):
            ledger._state["reserved"].pop(rid, None)
            actual = cost_usd(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            )
            ledger._state["spent_usd"] += actual
            ledger._event(
                "settle",
                ledger._purpose_of(rid),
                actual_usd=round(actual, 6),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                spent_total_usd=round(ledger._state["spent_usd"], 6),
            )
            return actual
        return self._locked(_do)

    def fail(self, rid, error_kind):
        """Charge the full reservation on a failed call (conservative)."""
        def _do(ledger):
            r = ledger._state["reserved"].pop(rid, None)
            charged = r["reserve_usd"] if r else 0.0
            ledger._state["spent_usd"] += charged
            ledger._event(
                "fail",
                r["purpose"] if r else "unknown",
                charged_usd=round(charged, 6),
                error=error_kind,
                spent_total_usd=round(ledger._state["spent_usd"], 6),
            )
            return charged
        return self._locked(_do)

    def _purpose_of(self, rid):
        # purpose was already popped in settle; look back at events for accuracy
        for entry in reversed(self._state["events"]):
            if entry["kind"] == "reserve" and entry.get("rid") == rid:
                return entry["purpose"]
        return "unknown"


def _main(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description="Provision the Promptimize budget ledger on this machine."
    )
    parser.add_argument("command", choices=["init", "status", "setcap"])
    parser.add_argument(
        "--cap",
        type=float,
        default=None,
        help="Remaining allowance in USD for THIS machine. The approved "
        "total is per experiment: on a second machine, provision only "
        "approved_total minus spend already accounted elsewhere.",
    )
    parser.add_argument("--path", default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    ledger = BudgetLedger(path=args.path, cap_usd=args.cap)
    if args.command == "setcap":
        if args.cap is None:
            parser.error("setcap requires --cap (new total allowance in USD)")
        ledger.set_cap(args.cap, note=args.note)
        print("cap updated to $%.2f on %s" % (ledger.cap_usd, ledger.path))
    elif args.command == "init":
        if args.cap is None:
            parser.error("init requires --cap (remaining allowance in USD)")
        ledger.provision(note=args.note)
        print("provisioned %s with remaining allowance $%.2f" % (ledger.path, ledger.cap_usd))
    else:
        state = ledger._read_state()
        if state is None:
            print("unprovisioned: %s (no paid calls allowed)" % ledger.path)
        else:
            print(
                "spent $%.6f, reserved $%.6f, remaining $%.6f of $%.2f"
                % (
                    state["spent_usd"],
                    sum(r["reserve_usd"] for r in state["reserved"].values()),
                    state.get("cap_usd", ledger.cap_usd) - ledger.committed_usd(),
                    state.get("cap_usd", ledger.cap_usd),
                )
            )


if __name__ == "__main__":
    import sys

    _main(sys.argv[1:])
