#!/usr/bin/env python3
"""Trial T3: counterfactual surface augmentation to debias the T2 head.

Train split ONLY is augmented (val and probes untouched) with 40
model-authored variants (experiments/t3_augmented_train.jsonl): 20
ambiguous prompts lengthened with repo paths while keeping the missing
info user-head-only, 20 executable prompts shortened while staying
discoverable. Labels by construction with per-row rationale, inspected
during authoring against each source row's missing_information /
discoverability anchor.

Head retrained on train+aug (372); config+threshold selected on that
train set; winner evaluated once on val (directional) and re-measured on
the frozen DEVELOPMENT probes (remediation evidence, not independent
generalization - see experiments/README.md probe governance).

Fully local, $0. Resumable embedding extraction.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import (embed_one, fit_logreg, load_agent,
                                 load_embs, metrics, wilson, REPO)

AUG = os.path.join(REPO, "experiments", "t3_augmented_train.jsonl")
AUG_EMB = os.path.join(REPO, "experiments", "t3_aug_embeddings.jsonl")
PROBE_EMB = os.path.join(REPO, "experiments", "t2_probe_embeddings.jsonl")
PROBES = os.path.join(REPO, "experiments", "t2_probes.jsonl")
OUT = os.path.join(REPO, "experiments", "t3_result.json")


def extract_aug():
    rows = [json.loads(l) for l in open(AUG)]
    agent = load_agent()
    done = set()
    if os.path.exists(AUG_EMB):
        done = {json.loads(l)["prompt"] for l in open(AUG_EMB)}
    out = open(AUG_EMB, "a", buffering=1)
    for r in rows:
        if r["prompt"] in done:
            continue
        vec = embed_one(agent, r["prompt"])
        out.write(json.dumps({"prompt": r["prompt"], "can_execute": r["can_execute"],
                              "emb": [round(float(x), 6) for x in vec]}) + "\n")
        print(f"[aug] embedded {r['source_idx']}", flush=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_and_eval():
    data = load_embs()
    Xtr = [e for e, _ in data["train"]]
    ytr = [l for _, l in data["train"]]
    for line in open(AUG_EMB):
        r = json.loads(line)
        Xtr.append(np.array(r["emb"]))
        ytr.append(0 if r["can_execute"] else 1)
    Xtr = np.stack(Xtr)
    ytr = np.array(ytr, dtype=np.float64)
    Xva = np.stack([e for e, _ in data["val"]])
    yva = np.array([l for _, l in data["val"]], dtype=np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    results = {}
    for lr in (0.1, 0.3, 0.5):
        for l2 in (1e-4, 1e-3, 1e-2):
            w, b = fit_logreg(Xtr, ytr, lr=lr, l2=l2)
            ptr = sigmoid(Xtr @ w + b)
            grid = {}
            for thr in np.arange(0.05, 0.96, 0.05):
                grid[round(float(thr), 2)] = metrics(ptr, ytr, float(thr))
            best_thr, best = max(grid.items(), key=lambda kv: (kv[1]["acc"], -kv[1]["over"]))
            results[f"lr{lr}_l2{l2}"] = {"best_thr": best_thr, **{f"train_{k}": v for k, v in best.items()}}

    name, sel = max(results.items(), key=lambda kv: (kv[1]["train_acc"], -kv[1]["train_over"]))
    lr, l2 = float(name.split("_")[0][2:]), float(name.split("_")[1][2:])
    w, b = fit_logreg(Xtr, ytr, lr=lr, l2=l2)
    thr = sel["best_thr"]
    tm = metrics(sigmoid(Xtr @ w + b), ytr, thr)
    vm = metrics(sigmoid(Xva @ w + b), yva, thr)

    # development probes (remediation evidence)
    probes = {r["id"]: r for r in (json.loads(l) for l in open(PROBES))}
    pembs = {json.loads(l)["id"]: np.array(json.loads(l)["emb"]) for l in open(PROBE_EMB)}
    cells, per_probe = {}, []
    for pid, r in sorted(probes.items()):
        if pid not in pembs:
            continue
        x = (pembs[pid] - mu) / sd
        p = float(sigmoid(x @ w + b))
        ask = p >= thr
        ok = ask == (not r["can_execute"])
        cells.setdefault(r["cell"], []).append(ok)
        per_probe.append({"id": pid, "cell": r["cell"], "p_ambiguous": round(p, 4),
                          "ask": bool(ask), "correct": ok})
    cell_summary = {c: {"acc": round(sum(v) / len(v), 3), "n": len(v)} for c, v in sorted(cells.items())}
    n_ok = sum(p["correct"] for p in per_probe)
    correct = int(round(vm["acc"] * vm["n"]))

    out = {
        "trial": "T3",
        "what": "frozen-embedding logistic head retrained on train+40 counterfactual surface augmentations (debias attempt)",
        "selected_config": {"lr": lr, "l2": l2, "thr": thr},
        "train": tm, "val": vm, "val_acc_wilson95": wilson(correct, vm["n"]),
        "all_train_configs": results,
        "dev_probes": {"overall": {"acc": round(n_ok / len(per_probe), 3), "n": len(per_probe)},
                       "by_cell": cell_summary, "per_probe": per_probe,
                       "status": "DEVELOPMENT diagnostics - remediation evidence only, not independent generalization"},
        "caveats": [
            "Augmented rows are model-authored with labels by construction; they debias the training signal, they are not real-prompt evidence.",
            "Val reused across trials - directional. Probe gains after seeing probe failures = remediation evidence per probe governance.",
            "40/372 rows augmented (11%) - a partial correction; residual shortcut likely.",
        ],
        "ts": time.time(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("train:", tm, "\nval:", vm, "\ndev probes:", out["dev_probes"]["overall"], cell_summary)
    for p in per_probe:
        if not p["correct"]:
            print("FAIL:", json.dumps(p))


if __name__ == "__main__":
    if "--train" in sys.argv:
        train_and_eval()
    else:
        extract_aug()
