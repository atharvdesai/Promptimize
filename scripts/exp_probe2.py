#!/usr/bin/env python3
"""Score frozen diagnostic probe family 2 through the T5 pipeline.

Probes (experiments/probe2_probes.jsonl) are MODEL-AUTHORED DIAGNOSTIC
PROBES authored AFTER the T5 input-normalization fix, with labels frozen
before scoring (SHA256 594f3c560c047474fd35fe8d4de10cbbbb19ccc11ff78835477175fc5002059d,
committed at bc86570 before any embedding/scoring). Family 2 adds the
cells family 1 lacked: long prompts WITHOUT file paths, as matched
counterfactual pairs controlling length/path/context, to test whether
the surviving shortcut is length rather than paths.

The pipeline under test is exactly T5's: normalize (strip path-like
tokens) -> LAYA ONNX embedding -> logistic head retrained on normalized
train embeddings with T5's selected config and threshold.

DEVELOPMENT diagnostics only - not an independent human holdout.
Fully local, $0. Resumable per probe row.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import (embed_one, fit_logreg, load_agent, REPO)
from exp_t5_normalize import normalize

PROBES = os.path.join(REPO, "experiments", "probe2_probes.jsonl")
TRAIN_EMB = os.path.join(REPO, "experiments", "t5_embeddings.jsonl")
T5_RESULT = os.path.join(REPO, "experiments", "t5_result.json")
PROBE_EMB_OUT = os.path.join(REPO, "experiments", "probe2_embeddings.jsonl")
OUT = os.path.join(REPO, "experiments", "probe2_result.json")
FROZEN_SHA = "594f3c560c047474fd35fe8d4de10cbbbb19ccc11ff78835477175fc5002059d"


def extract():
    rows = [json.loads(l) for l in open(PROBES)]
    agent = load_agent()
    done = set()
    if os.path.exists(PROBE_EMB_OUT):
        done = {json.loads(l)["id"] for l in open(PROBE_EMB_OUT)}
    out = open(PROBE_EMB_OUT, "a", buffering=1)
    for r in rows:
        if r["id"] in done:
            continue
        np_ = normalize(r["prompt"])
        vec = embed_one(agent, np_)
        out.write(json.dumps({"id": r["id"], "norm_prompt": np_,
                              "emb": [round(float(x), 6) for x in vec]}) + "\n")
        print(f"[probe2] {r['id']} embedded", flush=True)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def score():
    import hashlib
    actual = hashlib.sha256(open(PROBES, "rb").read()).hexdigest()
    assert actual == FROZEN_SHA, f"probe file changed since freeze: {actual}"

    probes = {r["id"]: r for r in (json.loads(l) for l in open(PROBES))}
    embs = {json.loads(l)["id"]: np.array(json.loads(l)["emb"]) for l in open(PROBE_EMB_OUT)}
    cfg = json.load(open(T5_RESULT))["selected_config"]

    tr = [json.loads(l) for l in open(TRAIN_EMB) if json.loads(l)["split"] == "train"]
    Xtr = np.stack([np.array(r["emb"]) for r in tr])
    ytr = np.array([r["label"] for r in tr], dtype=np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    w, b = fit_logreg((Xtr - mu) / sd, ytr, lr=cfg["lr"], l2=cfg["l2"])
    thr = cfg["thr"]

    cells, per_probe, pairs = {}, [], {}
    for pid, r in sorted(probes.items()):
        x = (embs[pid] - mu) / sd
        p = float(sig(x @ w + b))
        ask = p >= thr
        ok = ask == bool(1 if not r["can_execute"] else 0)
        cells.setdefault(r["cell"], []).append(ok)
        per_probe.append({"id": r["id"], "cell": r["cell"], "pair": r["pair"],
                          "p_ambiguous": round(p, 4), "ask": bool(ask), "correct": ok})
        pairs.setdefault(r["pair"], {})[r["can_execute"]] = p

    pair_contrast = {k: {"exec_p": round(v.get(True, float("nan")), 4),
                         "amb_p": round(v.get(False, float("nan")), 4),
                         "separated": bool(v.get(True, 1) < thr <= v.get(False, 0))}
                     for k, v in sorted(pairs.items())}
    cell_summary = {c: {"acc": round(sum(v) / len(v), 3), "n": len(v)} for c, v in sorted(cells.items())}
    n_ok = sum(p["correct"] for p in per_probe)
    out = {
        "trial": "P2",
        "what": "diagnostic probe family 2 (post-T5, 24 probes, 12 matched counterfactual pairs incl. long-no-path cells) scored through the frozen T5 pipeline",
        "frozen_sha256": FROZEN_SHA,
        "pipeline": "T5 normalize + LAYA ONNX embedding + logistic head (T5 selected config, retrained on normalized train)",
        "t5_config": cfg,
        "overall": {"acc": round(n_ok / len(per_probe), 3), "n": len(per_probe)},
        "by_cell": cell_summary,
        "pair_contrast": pair_contrast,
        "per_probe": per_probe,
        "status": "DEVELOPMENT diagnostics - remediation evidence only, not an independent holdout; same-author synthetic probes",
        "ts": time.time(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("overall:", out["overall"])
    print("by cell:", json.dumps(cell_summary, indent=1))
    print("pair contrast:", json.dumps(pair_contrast, indent=1))
    for p in per_probe:
        if not p["correct"]:
            print("FAIL:", json.dumps(p))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract()
    else:
        score()
