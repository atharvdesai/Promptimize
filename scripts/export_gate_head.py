#!/usr/bin/env python3
"""Export the trained T5 decision head for the shipped plugin.

Retrains the numpy logistic head deterministically from the committed
train embeddings (experiments/t5_embeddings.jsonl) with T5's selected
config (experiments/t5_result.json) and writes plugin/gate_head.json:
the small artifact (1024-d w/mu/sd + bias + threshold) the plugin loads
in decider mode "head". NOT model weights - a ~100 KB numpy head.

The plugin applies gate_normalize.normalize() before embedding, the
same single implementation used here (T5 train/inference contract).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import fit_logreg, REPO

TRAIN_EMB = os.path.join(REPO, "experiments", "t5_embeddings.jsonl")
T5_RESULT = os.path.join(REPO, "experiments", "t5_result.json")
OUT = os.path.join(REPO, "plugin", "gate_head.json")


def main():
    cfg = json.load(open(T5_RESULT))["selected_config"]
    tr = [json.loads(l) for l in open(TRAIN_EMB) if json.loads(l)["split"] == "train"]
    X = np.stack([np.array(r["emb"]) for r in tr])
    y = np.array([r["label"] for r in tr], dtype=np.float64)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    w, b = fit_logreg((X - mu) / sd, y, lr=cfg["lr"], l2=cfg["l2"])
    artifact = {
        "format": "promptimize-gate-head/v1",
        "source_trial": "T5",
        "trained_on": "332 normalized train rows (seed 20261003), frozen LAYA ONNX mean-pooled 1024-d embeddings",
        "normalization": "gate_normalize.normalize (strip path-like tokens, collapse whitespace) - MUST be applied before embedding at inference",
        "embedding": "codenamev/laya-onnx english, last_hidden_state mean-pooled over attention mask",
        "semantics": "p_ambiguous = sigmoid(standardize(emb) @ w + b); ask when p_ambiguous >= threshold",
        "config": cfg,
        "threshold": cfg["thr"],
        "w": [round(float(x), 8) for x in w],
        "b": round(float(b), 8),
        "mu": [round(float(x), 8) for x in mu],
        "sd": [round(float(x), 8) for x in sd],
        "exported_at": time.time(),
    }
    # Measure the ARTIFACT AS SHIPPED (post-rounding), not the pre-export
    # floats: borderline rows near the threshold can flip under 8-dp
    # rounding, and the shipped numbers are the ones that matter.
    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))
    wr = np.array(artifact["w"]); mur = np.array(artifact["mu"]); sdr = np.array(artifact["sd"])
    measured = {}
    for split in ("train", "val"):
        rs = [json.loads(l) for l in open(TRAIN_EMB) if json.loads(l)["split"] == split]
        ok = over = under = 0
        for r in rs:
            Xr = np.array(r["emb"])
            p_amb = float(sig(((Xr - mur) / sdr) @ wr + artifact["b"]))
            ask = p_amb >= artifact["threshold"]
            if (ask and r["label"] == 1) or (not ask and r["label"] == 0): ok += 1
            elif ask: over += 1
            else: under += 1
        n = len(rs)
        measured[split] = {"acc": round(ok / n, 3), "over": round(over / n, 3),
                           "under": round(under / n, 3), "n": n}
    artifact["measured"] = measured
    artifact["blind_spot"] = ("long prompts (>=200 chars): ambiguous recall 0/8 on probe "
                              "families 1+2 - see trials P2/P3")
    with open(OUT, "w") as f:
        json.dump(artifact, f)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
    print("measured:", json.dumps(measured))


if __name__ == "__main__":
    main()
