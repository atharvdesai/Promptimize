#!/usr/bin/env python3
"""Trial T5: remove the surface signal from the INPUT instead of the data.

T3/T3b showed same-author augmentation can't teach the concept - the head
learns the augmentation style. T5 tests the complementary fix: normalize
every prompt (train and inference alike) before embedding - strip
path-like tokens (contain '/' or look like file names) and collapse
whitespace - so the path/length shortcut signal never enters the
decision input. Discoverability semantics (what info is missing) is
preserved; specific file names almost never carry it.

Head retrained on normalized train embeddings, config+threshold selected
on train, winner evaluated once on val (directional) + frozen
development probes (remediation evidence). Fully local, $0. Resumable.
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import (STATE_TMPL, embed_one, fit_logreg,
                                 load_agent, metrics, wilson, REPO)
from gate_normalize import normalize  # single shared implementation (T5 contract)

EMB_OUT = os.path.join(REPO, "experiments", "t5_embeddings.jsonl")
OUT = os.path.join(REPO, "experiments", "t5_result.json")
PROBES = os.path.join(REPO, "experiments", "t2_probes.jsonl")


def rows_for(split):
    if split in ("train", "val"):
        rows = [json.loads(l) for l in open(os.path.join(REPO, "dataset", "splits", f"{split}.jsonl"))]
        return [(r["prompt"], 0 if r["can_execute"] else 1) for r in rows]
    rows = [json.loads(l) for l in open(PROBES)]
    return [(r["prompt"], 0 if r["can_execute"] else 1, r["id"], r["cell"]) for r in rows]


def extract(split):
    agent = load_agent()
    done = set()
    if os.path.exists(EMB_OUT):
        done = {(json.loads(l)["split"], json.loads(l)["norm_prompt"]) for l in open(EMB_OUT)}
    out = open(EMB_OUT, "a", buffering=1)
    for item in rows_for(split):
        prompt, label = item[0], item[1]
        np_ = normalize(prompt)
        if (split, np_) in done:
            continue
        vec = embed_one(agent, np_)
        rec = {"split": split, "norm_prompt": np_, "label": label,
               "emb": [round(float(x), 6) for x in vec]}
        if split == "probe":
            rec["id"], rec["cell"] = item[2], item[3]
        out.write(json.dumps(rec) + "\n")
        print(f"[{split}] embedded", flush=True)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_and_eval():
    data = {"train": [], "val": [], "probe": []}
    for line in open(EMB_OUT):
        r = json.loads(line)
        data[r["split"]].append(r)
    Xtr = np.stack([np.array(r["emb"]) for r in data["train"]])
    ytr = np.array([r["label"] for r in data["train"]], dtype=np.float64)
    Xva = np.stack([np.array(r["emb"]) for r in data["val"]])
    yva = np.array([r["label"] for r in data["val"]], dtype=np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    results = {}
    for lr in (0.1, 0.3, 0.5):
        for l2 in (1e-4, 1e-3, 1e-2):
            w, b = fit_logreg(Xtr, ytr, lr=lr, l2=l2)
            ptr = sig(Xtr @ w + b)
            grid = {round(float(t), 2): metrics(ptr, ytr, float(t)) for t in np.arange(0.05, 0.96, 0.05)}
            best_thr, best = max(grid.items(), key=lambda kv: (kv[1]["acc"], -kv[1]["over"]))
            results[f"lr{lr}_l2{l2}"] = {"best_thr": best_thr, **{f"train_{k}": v for k, v in best.items()}}
    name, sel = max(results.items(), key=lambda kv: (kv[1]["train_acc"], -kv[1]["train_over"]))
    lr, l2 = float(name.split("_")[0][2:]), float(name.split("_")[1][2:])
    w, b = fit_logreg(Xtr, ytr, lr=lr, l2=l2)
    thr = sel["best_thr"]
    tm = metrics(sig(Xtr @ w + b), ytr, thr)
    vm = metrics(sig(Xva @ w + b), yva, thr)

    cells, per_probe = {}, []
    for r in data["probe"]:
        x = (np.array(r["emb"]) - mu) / sd
        p = float(sig(x @ w + b))
        ask = p >= thr
        ok = ask == bool(r["label"])
        cells.setdefault(r["cell"], []).append(ok)
        per_probe.append({"id": r["id"], "cell": r["cell"], "p_ambiguous": round(p, 4),
                          "ask": bool(ask), "correct": ok})
    cell_summary = {c: {"acc": round(sum(v) / len(v), 3), "n": len(v)} for c, v in sorted(cells.items())}
    n_ok = sum(p["correct"] for p in per_probe)
    correct = int(round(vm["acc"] * vm["n"]))
    out = {
        "trial": "T5",
        "what": "input normalization (strip path-like tokens) before embedding; T2-style head retrained on normalized train",
        "selected_config": {"lr": lr, "l2": l2, "thr": thr},
        "train": tm, "val": vm, "val_acc_wilson95": wilson(correct, vm["n"]),
        "all_train_configs": results,
        "dev_probes": {"overall": {"acc": round(n_ok / len(per_probe), 3), "n": len(per_probe)},
                       "by_cell": cell_summary, "per_probe": per_probe,
                       "status": "DEVELOPMENT diagnostics - remediation evidence only"},
        "caveats": [
            "Normalization is applied at train and inference - the deployed contract must match.",
            "Stripping paths can remove genuinely needed context in edge cases; that cost is measured only on these splits/probes.",
            "Val reused across trials - directional; probes are development diagnostics.",
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
    elif len(sys.argv) > 1:
        extract(sys.argv[1])
    else:
        print("usage: exp_t5_normalize.py train|val|probe|--train")
