"""Run the few-shot Haiku gate baseline over a frozen split.

Roadmap step 2: score a strong prompted model before fine-tuning anything.
Writes an evaluate.py-compatible predictions JSONL, then prints usage
totals (tokens and estimated cost) - never prompt text.

    PROMPTIMIZE_ANTHROPIC_KEY_FILE=~/.vault/.../password \
    python3 scripts/run_haiku_baseline.py \
        --split dataset/splits/val.jsonl --out /tmp/haiku_val_predictions.jsonl

Budget: every call reserves its worst-case uncached cost first and the
run stops (fail closed) when the cap in PROMPTIMIZE_BUDGET_USD is reached
(default $2.00). Rows a call failed for are skipped: evaluate.py counts a
missing prediction as let-through, matching the silent-gate semantics, and
the run summary reports how many were skipped so the score can be read
honestly.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from budget_ledger import BudgetExhausted, BudgetLedger
from haiku_gate import GateCallError, GateParseError, call_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dataset/splits/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    args = ap.parse_args()

    ledger = BudgetLedger()
    rows = [json.loads(l) for l in open(args.split) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                done.add(json.loads(line)["prompt"])

    calls = failures = skipped_budget = 0
    started_spend = ledger._state["spent_usd"]
    t0 = time.time()
    with open(args.out, "a") as fo:
        for i, row in enumerate(rows, 1):
            if row["prompt"] in done:
                continue
            try:
                result, usage = call_gate(row["prompt"], ledger, purpose="val-baseline")
                calls += 1
            except BudgetExhausted as e:
                print(f"[{i}/{len(rows)}] BUDGET EXHAUSTED, stopping: {e}", file=sys.stderr)
                skipped_budget = len(rows) - len(done) - calls - failures
                break
            except GateCallError as e:
                failures += 1
                print(f"[{i}/{len(rows)}] call failed: {e}", file=sys.stderr)
                continue
            except GateParseError:
                failures += 1
                print(f"[{i}/{len(rows)}] unparseable response, row skipped", file=sys.stderr)
                continue
            pred = {"prompt": row["prompt"], "can_execute": result["can_execute"]}
            if not result["can_execute"] and result.get("clarifying_question"):
                pred["clarifying_question"] = result["clarifying_question"]
                if result.get("ambiguity_type"):
                    pred["ambiguity_type"] = result["ambiguity_type"]
            fo.write(json.dumps(pred) + "\n")
            fo.flush()
            print(
                f"[{i}/{len(rows)}] can_execute={result['can_execute']} "
                f"in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
                f"cache_r={usage.get('cache_read_input_tokens')} cache_w={usage.get('cache_creation_input_tokens')}",
                file=sys.stderr,
            )

    spent = ledger._state["spent_usd"] - started_spend
    summary = {
        "rows": len(rows),
        "calls": calls,
        "failures": failures,
        "skipped_budget": skipped_budget,
        "run_cost_usd": round(spent, 6),
        "spent_total_usd": round(ledger._state["spent_usd"], 6),
        "cap_usd": ledger.cap_usd,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
