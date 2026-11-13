#!/usr/bin/env python3
"""Trial T1: tune the LAYA gate's noul instruction text.

Zero paid calls, fully local. For each candidate instruction phrasing,
score P(can_execute) for every TRAIN row, pick (variant, threshold) on
train (primary: accuracy; tie-break: lower over-asking), then evaluate
the single winner once on VAL. Resumable: probabilities are appended
per (variant, split, row) and existing keys are skipped on re-run.

LAYA: convaiinnovations/laya via third-party ONNX export
codenamev/laya-onnx @ 1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4
(parity with upstream torch checkpoint not yet established - see
experiments/trials.jsonl caveat T0). Weights live outside the repo.
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.environ.get(
    "PROMPTIMIZE_LAYA_ONNX",
    os.path.expanduser("~/work/laya_test/model.onnx"),
)
OUT = os.path.join(REPO, "experiments", "t1_probs.jsonl")

VARIANTS = {
    "v1_original": "Can a coding agent execute this request correctly without asking the user a clarifying question?",
    "v2_discover": "Can a coding agent complete this request by discovering anything it needs itself (reading code, running tests, grepping), without asking the user anything first?",
    "v3_userhead": "Does fulfilling this request require information that exists only in the user's head (intent, preferences, business rules, unseen errors)?",
    "v4_actionable": "Is this request specific and actionable enough for a coding agent to start work on it right now without any clarification?",
    "v5_confident": "Would a careful senior engineer start implementing this request immediately, confident they understood what was wanted?",
    "v6_oneliner": "Is this coding request unambiguous and ready to implement as written?",
}

STATE_TMPL = 'A user gave a coding agent this request: "%s"'


def load_agent():
    import onnxruntime as ort

    real_session = ort.InferenceSession

    def memory_safe_session(path, sess_options=None, providers=None, **kw):
        so = sess_options or ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return real_session(path, sess_options=so, providers=["CPUExecutionProvider"], **kw)

    ort.InferenceSession = memory_safe_session
    os.environ.setdefault("USE_TF", "0")
    import laya

    return laya.load("codenamev/laya-onnx", backend="onnx", subfolder="english", onnx_path=ONNX)


def existing_keys():
    keys = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                keys.add((r["variant"], r["split"], r["prompt"]))
    return keys


def score_split(agent, variant, instructions, split, rows, done, out_f):
    n = 0
    for row in rows:
        if (variant, split, row["prompt"]) in done:
            continue
        t0 = time.time()
        result = agent.predict(
            STATE_TMPL % row["prompt"],
            {"can_execute": {"type": "noul", "instructions": instructions}},
        )
        prob = float(result["answers"]["can_execute"]["noul"])
        out_f.write(json.dumps({
            "variant": variant, "split": split, "prompt": row["prompt"],
            "prob": prob, "gold_can_execute": bool(row["can_execute"]),
            "latency_s": round(time.time() - t0, 3),
        }) + "\n")
        out_f.flush()
        n += 1
    return n


def load_split(name):
    with open(os.path.join(REPO, "dataset", "splits", name)) as f:
        return [json.loads(l) for l in f]


def metrics(rows, threshold):
    tp = fp = tn = fn = 0  # positive class = "asks" (predicts cannot execute)
    for r in rows:
        pred_ask = r["prob"] < threshold
        gold_ask = not r["gold_can_execute"]
        if pred_ask and gold_ask: tp += 1
        elif pred_ask and not gold_ask: fp += 1
        elif not pred_ask and gold_ask: fn += 1
        else: tn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0
    over = fp / (fp + tn) if (fp + tn) else 0   # of executable rows, fraction interrupted
    under = fn / (fn + tp) if (fn + tp) else 0  # of ambiguous rows, fraction missed
    return {"acc": acc, "over": over, "under": under, "n": n, "threshold": threshold}


def select(records):
    variants = sorted({r["variant"] for r in records})
    best = None
    for v in variants:
        train = [r for r in records if r["variant"] == v and r["split"] == "train"]
        for thr in [x / 100 for x in range(15, 61, 5)]:
            m = metrics(train, thr)
            key = (m["acc"], -m["over"], -m["under"])
            if best is None or key > best[0]:
                best = (key, v, m)
    return best[1], best[2]


def main():
    only_val_winner = "--val-only" in sys.argv
    train, val = load_split("train.jsonl"), load_split("val.jsonl")
    done = existing_keys()
    agent = None
    with open(OUT, "a") as out_f:
        for variant, instructions in VARIANTS.items():
            missing = sum(1 for r in train if (variant, "train", r["prompt"]) not in done)
            if missing == 0:
                continue
            if agent is None:
                print("loading LAYA (one-time)...", flush=True)
                agent = load_agent()
            print(f"scoring {variant} on train ({missing} rows)...", flush=True)
            score_split(agent, variant, instructions, "train", train, done, out_f)
    with open(OUT) as f:
        records = [json.loads(l) for l in f]
    winner, train_m = select(records)
    print("WINNER on train:", winner, train_m, flush=True)

    done = existing_keys()
    with open(OUT, "a") as out_f:
        missing = sum(1 for r in val if (winner, "val", r["prompt"]) not in done)
        if missing:
            if agent is None:
                agent = load_agent()
            print(f"scoring winner {winner} on val ({missing} rows)...", flush=True)
            score_split(agent, winner, VARIANTS[winner], "val", val, done, out_f)
    with open(OUT) as f:
        records = [json.loads(l) for l in f]
    val_rows = [r for r in records if r["variant"] == winner and r["split"] == "val"]
    val_m = metrics(val_rows, train_m["threshold"])
    summary = {
        "trial": "T1", "winner": winner, "train": train_m, "val": val_m,
        "instructions": VARIANTS[winner], "ts": time.time(),
    }
    with open(os.path.join(REPO, "experiments", "t1_result.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
