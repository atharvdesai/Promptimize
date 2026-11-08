#!/usr/bin/env python3
"""Learning curves: how many examples does the frozen-embedding head need?

Trains the T2-style numpy logistic head on stratified subsamples of the
train embeddings (32/64/128/256/332), 5 seeds per size, evaluates each
on val (44). Val is reused across curve points as a MEASUREMENT (not
selection) - all numbers directional per experiments/README.md.

Fully local, $0. Uses the archived T2 embeddings (gitignored caches).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import fit_logreg, load_embs, metrics, wilson, REPO

SIZES = (32, 64, 128, 256, 332)
SEEDS = (0, 1, 2, 3, 4)
LR, L2, THR = 0.3, 1e-3, 0.5  # fixed config for comparability across sizes
OUT = os.path.join(REPO, "experiments", "learning_curves.json")


def main():
    data = load_embs()
    Xtr_full = np.stack([e for e, _ in data["train"]])
    ytr_full = np.array([l for _, l in data["train"]], dtype=np.float64)
    Xva = np.stack([e for e, _ in data["val"]])
    yva = np.array([l for _, l in data["val"]], dtype=np.float64)
    # standardize once with FULL train stats (measurement consistency, not selection)
    mu, sd = Xtr_full.mean(axis=0), Xtr_full.std(axis=0) + 1e-8
    Xtr_full, Xva = (Xtr_full - mu) / sd, (Xva - mu) / sd

    idx_pos = np.where(ytr_full == 1)[0]
    idx_neg = np.where(ytr_full == 0)[0]
    curves = {}
    for n in SIZES:
        runs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            if n >= len(ytr_full):
                sel = np.arange(len(ytr_full))
            else:
                half = n // 2
                sel = np.concatenate([
                    rng.choice(idx_pos, half, replace=False),
                    rng.choice(idx_neg, n - half, replace=False),
                ])
            w, b = fit_logreg(Xtr_full[sel], ytr_full[sel], lr=LR, l2=L2, seed=seed)
            pva = 1.0 / (1.0 + np.exp(-np.clip(Xva @ w + b, -30, 30)))
            m = metrics(pva, yva, THR)
            runs.append({"seed": seed, **m})
        accs = [r["acc"] for r in runs]
        overs = [r["over"] for r in runs]
        unders = [r["under"] for r in runs]
        curves[str(n)] = {
            "runs": runs,
            "acc_mean": round(float(np.mean(accs)), 3), "acc_std": round(float(np.std(accs)), 3),
            "over_mean": round(float(np.mean(overs)), 3), "under_mean": round(float(np.mean(unders)), 3),
        }
        print(f"n={n:3d}  acc {np.mean(accs):.3f}+-{np.std(accs):.3f}  over {np.mean(overs):.3f}  under {np.mean(unders):.3f}", flush=True)

    out = {
        "trial": "LC1",
        "what": "learning curves for the frozen-embedding logistic head: stratified subsamples of train (5 seeds/size), measured on val",
        "config": {"lr": LR, "l2": L2, "thr": THR, "sizes": SIZES, "seeds": SEEDS},
        "curves": curves,
        "caveats": [
            "Val reused as a measurement across curve points - directional, not an independent evaluation.",
            "Head is the numpy logistic comparator (T2-style), not the official RLCD path (resource-blocked: torch load OOM-killed in 2GB sandbox).",
            "Subsamples are stratified random, not grouped (dataset carries no group ids beyond ambiguity_type).",
            "Same single-author synthetic data - curve shape reflects THIS data's redundancy, not real-prompt learning.",
        ],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
