#!/usr/bin/env python3
"""Score the frozen synthetic diagnostic probes through the T2 pipeline.

Probes (experiments/t2_probes.jsonl) are MODEL-AUTHORED DIAGNOSTIC
PROBES, not an independent human holdout: labels were frozen before
scoring (SHA256 recorded in experiments/trials.jsonl) and results probe
whether the T2 head keys on surface features (length/path presence)
rather than discoverability. Fully local, $0. Resumable per probe row.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import (EMB_OUT, RESULT_OUT, embed_one, fit_logreg,
                                 load_agent, load_embs, metrics, wilson, REPO)

PROBES = os.path.join(REPO, "experiments", "t2_probes.jsonl")
PROBE_EMB_OUT = os.path.join(REPO, "experiments", "t2_probe_embeddings.jsonl")
OUT = os.path.join(REPO, "experiments", "t2_probe_result.json")


def extract_probes():
    rows = [json.loads(l) for l in open(PROBES)]
    agent = load_agent()
    done = set()
    if os.path.exists(PROBE_EMB_OUT):
        done = {json.loads(l)["id"] for l in open(PROBE_EMB_OUT)}
    out = open(PROBE_EMB_OUT, "a", buffering=1)
    for r in rows:
        if r["id"] in done:
            continue
        vec = embed_one(agent, r["prompt"])
        out.write(json.dumps({"id": r["id"], "emb": [round(float(x), 6) for x in vec]}) + "\n")
        print(f"[probe] {r['id']} embedded", flush=True)


def score():
    probes = {r["id"]: r for r in (json.loads(l) for l in open(PROBES))}
    embs = {json.loads(l)["id"]: np.array(json.loads(l)["emb"]) for l in open(PROBE_EMB_OUT)}
    cfg = json.load(open(RESULT_OUT))["selected_config"]

    data = load_embs()
    Xtr = np.stack([e for e, _ in data["train"]])
    ytr = np.array([l for _, l in data["train"]], dtype=np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    w, b = fit_logreg(Xtr, ytr, lr=cfg["lr"], l2=cfg["l2"])
    thr = cfg["thr"]

    cells = {}
    per_probe = []
    for pid, r in sorted(probes.items()):
        if pid not in embs:
            continue
        x = (embs[pid] - mu) / sd
        p = float(1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -30, 30))))
        ask = p >= thr
        gold_amb = not r["can_execute"]
        ok = ask == gold_amb
        cells.setdefault(r["cell"], []).append(ok)
        per_probe.append({"id": pid, "cell": r["cell"], "p_ambiguous": round(p, 4),
                          "ask": bool(ask), "gold_ambiguous": gold_amb, "correct": ok})

    summary = {c: {"acc": round(sum(v) / len(v), 3), "n": len(v)} for c, v in sorted(cells.items())}
    n_ok = sum(p["correct"] for p in per_probe)
    out = {
        "trial": "T2-probes",
        "what": "Frozen T2 pipeline scored on 20 model-authored synthetic diagnostic probes (NOT an independent human holdout; labels frozen pre-scoring, SHA256 in trials.jsonl)",
        "overall": {"acc": round(n_ok / len(per_probe), 3), "n": len(per_probe),
                    "wilson95": wilson(n_ok, len(per_probe))},
        "by_cell": summary,
        "per_probe": per_probe,
        "shortcut_signature": "A/D strong with B/C collapsed = surface-feature shortcut (length/paths)",
        "caveats": [
            "Model-authored probes: diagnostic only, no real-prompt generalization claim.",
            "Single threshold/config from T2; probes share the author's phrasing quirks.",
            "Failures preserved verbatim below; no retroactive probe edits.",
        ],
        "ts": time.time(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"overall": out["overall"], "by_cell": summary}))
    for p in per_probe:
        if not p["correct"]:
            print("FAIL:", json.dumps(p))


if __name__ == "__main__":
    if "--score" in sys.argv:
        score()
    else:
        extract_probes()
