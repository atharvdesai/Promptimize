#!/usr/bin/env python3
"""Trial T2: trained logistic head on frozen LAYA encoder embeddings.

This is NOT LAYA weight tuning: the ONNX encoder stays frozen; we read
its last_hidden_state (batch, seq, 1024), mean-pool over the attention
mask, and fit a small numpy logistic regression head on the train split
only. (variant, threshold) are selected on train; the winner is
evaluated once on val. Val numbers are directional (selection exposure
across the loop), per experiments/README.md.

Zero paid calls, fully local. Resumable: embeddings append per row and
existing rows are skipped on re-run. Weights live outside the repo.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.environ.get(
    "PROMPTIMIZE_LAYA_ONNX",
    os.path.expanduser("~/work/laya_test/model.onnx"),
)
EMB_OUT = os.path.join(REPO, "experiments", "t2_embeddings.jsonl")
RESULT_OUT = os.path.join(REPO, "experiments", "t2_result.json")

INSTRUCTIONS = (
    "Can a coding agent complete this request by discovering anything it "
    "needs itself (reading code, running tests, grepping), without asking "
    "the user anything first?"
)
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


def embed_one(agent, prompt):
    """Frozen forward pass; returns the mean-pooled 1024-dim embedding."""
    from laya.onnx_agent import _collate_items_numpy

    questions = {"can_execute": {"type": "noul", "instructions": INSTRUCTIONS}}
    internal = {qid: agent._to_internal(q) for qid, q in questions.items()}
    ids = list(questions.keys())
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = agent._encode_state(STATE_TMPL % prompt, ids, internal, max_len, head_max_len)
    inputs = _collate_items_numpy([items], agent.tok.pad_token_id)
    hidden = agent.session.run(["last_hidden_state"], inputs)[0][0]  # (seq, 1024)
    mask = inputs["attention_mask"][0].astype(np.float64)  # (seq,)
    return (hidden * mask[:, None]).sum(axis=0) / max(mask.sum(), 1.0)


def extract(split, rows):
    agent = load_agent()
    done = set()
    if os.path.exists(EMB_OUT):
        with open(EMB_OUT) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["split"], r["prompt"]))
    out = open(EMB_OUT, "a", buffering=1)
    n = 0
    for row in rows:
        if (split, row["prompt"]) in done:
            continue
        t0 = time.time()
        vec = embed_one(agent, row["prompt"])
        out.write(json.dumps({
            "split": split, "prompt": row["prompt"],
            "label": "ambiguous" if not row["can_execute"] else "executable",
            "emb": [round(float(x), 6) for x in vec],
        }) + "\n")
        n += 1
        print(f"[{split}] {n} embedded ({time.time() - t0:.1f}s)", flush=True)


def load_embs():
    data = {"train": [], "val": []}
    with open(EMB_OUT) as f:
        for line in f:
            r = json.loads(line)
            data[r["split"]].append((np.array(r["emb"]), 1 if r["label"] == "ambiguous" else 0))
    return data


def fit_logreg(X, y, lr=0.5, epochs=3000, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
        b -= lr * float((p - y).mean())
    return w, b


def metrics(probs, y, thr):
    y = np.asarray(y)
    ask = np.asarray(probs) >= thr  # ask iff P(ambiguous) >= thr
    acc = float((ask == y.astype(bool)).mean())
    exec_mask = y == 0
    over = float(ask[exec_mask].mean()) if exec_mask.any() else 0.0
    amb_mask = y == 1
    under = float((~ask[amb_mask]).mean()) if amb_mask.any() else 0.0
    return {"acc": round(acc, 3), "over": round(over, 3), "under": round(under, 3),
            "n": int(len(y)), "thr": thr}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def train_and_eval():
    data = load_embs()
    Xtr = np.stack([e for e, _ in data["train"]])
    ytr = np.array([l for _, l in data["train"]], dtype=np.float64)
    Xva = np.stack([e for e, _ in data["val"]])
    yva = np.array([l for _, l in data["val"]], dtype=np.float64)
    # standardize with train stats only
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    results = {}
    for lr in (0.1, 0.3, 0.5):
        for l2 in (1e-4, 1e-3, 1e-2):
            w, b = fit_logreg(Xtr, ytr, lr=lr, l2=l2)
            ptr = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
            grid = {}
            for thr in np.arange(0.05, 0.96, 0.05):
                grid[round(float(thr), 2)] = metrics(ptr, ytr, float(thr))
            best_thr, best = max(grid.items(), key=lambda kv: (kv[1]["acc"], -kv[1]["over"]))
            results[f"lr{lr}_l2{l2}"] = {"train": grid, "best_thr": best_thr, **{f"train_{k}": v for k, v in best.items()}}

    # select config on train
    name, sel = max(results.items(), key=lambda kv: (kv[1]["train_acc"], -kv[1]["train_over"]))
    lr, l2 = float(name.split("_")[0][2:]), float(name.split("_")[1][2:])
    w, b = fit_logreg(Xtr, ytr, lr=lr, l2=l2)
    ptr = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
    pva = 1.0 / (1.0 + np.exp(-np.clip(Xva @ w + b, -30, 30)))
    thr = sel["best_thr"]
    tm = metrics(ptr, ytr, thr)
    vm = metrics(pva, yva, thr)
    correct = int(round(vm["acc"] * vm["n"]))
    out = {
        "trial": "T2",
        "what": "numpy logistic head on frozen LAYA ONNX mean-pooled embeddings (not weight tuning)",
        "instruction_variant": "v2_discover (T1 winner)",
        "train": tm,
        "val": vm,
        "val_acc_wilson95": wilson(correct, vm["n"]),
        "selected_config": {"lr": lr, "l2": l2, "thr": thr},
        "all_train_configs": {k: {"best_thr": v["best_thr"], "train_acc": v["train_acc"],
                                  "train_over": v["train_over"], "train_under": v["train_under"]}
                              for k, v in results.items()},
        "caveats": [
            "Val reused across sequential trials: directional, not an independent final evaluation.",
            "ONNX third-party export; parity with upstream torch checkpoint unestablished.",
            "Head trained on train.jsonl only (n=332); small-data logistic regression.",
        ],
        "ts": time.time(),
    }
    with open(RESULT_OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["train"]), json.dumps(out["val"]), "config:", out["selected_config"])


def main():
    with open(os.path.join(REPO, "dataset", "splits", "train.jsonl")) as f:
        train = [json.loads(l) for l in f]
    with open(os.path.join(REPO, "dataset", "splits", "val.jsonl")) as f:
        val = [json.loads(l) for l in f]
    if "--train" in sys.argv:
        train_and_eval()
        return
    if "--check" in sys.argv:
        data = load_embs()
        print(f"embedded: train {len(data['train'])}/{len(train)}, val {len(data['val'])}/{len(val)}")
        if len(data["train"]) == len(train) and len(data["val"]) == len(val):
            print("EXTRACTION COMPLETE")
        return
    if "train" in sys.argv:
        extract("train", train)
    elif "val" in sys.argv:
        extract("val", val)


if __name__ == "__main__":
    main()
