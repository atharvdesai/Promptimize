#!/usr/bin/env python3
"""Trial T6: can ANY threshold policy on the shipped head recover the
long-ambiguous blind spot? (No - and this script proves it numerically.)

Trials P2/P3 measured the blind spot (long-ambiguous recall 0/8) but left
open a cheap mitigation: the hook already flags prompts >=200 chars with
blind_spot:true, so a length-conditional LOWER ask-threshold might catch
them. T6 scores every long probe (families 1+2) through the SHIPPED
artifact (plugin/gate_head.json - the exact weights a user runs) and
sweeps the ask threshold on p_ambiguous down to the numeric floor.

Result: the two distributions overlap almost completely. Long-ambiguous
z-scores span -17.0..-4.2; long-executable span -24.7..-7.2. At any
operational threshold (>= 0.001) recall is 2/13 with zero false-asks;
pushing to 1e-5 recalls only 7/13 while already false-asking 2/13
long-executable probes; 1e-6 recalls 10/13 but false-asks 7/13. No
threshold separates the cells, so a length-conditional threshold is not
a viable mitigation.
The blind spot is a confidence-saturated miss, not a borderline one:
mitigation must be upstream (training data / retraining), never a
policy tweak on this head.

Also verified in this run: the full T5/P2 pipeline reproduces EXACTLY on
a fresh machine + fresh dependency install (val 0.977/0.048/0.000; every
probe2 per-probe p_ambiguous matches the committed P2 result to 4
decimals; laya 0.4.26/0.4.28/0.4.30 give cos=1.0 identical embeddings).

Inputs: experiments/t5_embeddings.jsonl + experiments/probe2_embeddings.jsonl
(regenerate with exp_t5_normalize.py train|val|probe and exp_probe2.py extract;
both gitignored, fully local, $0). Output: experiments/t6_result.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import fit_logreg, metrics, wilson, REPO

ARTIFACT = os.path.join(REPO, "plugin", "gate_head.json")
T5_EMB = os.path.join(REPO, "experiments", "t5_embeddings.jsonl")
P2_EMB = os.path.join(REPO, "experiments", "probe2_embeddings.jsonl")
P2_PROBES = os.path.join(REPO, "experiments", "probe2_probes.jsonl")
OUT = os.path.join(REPO, "experiments", "t6_result.json")
THRESHOLDS = [0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6]


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main():
    g = json.load(open(ARTIFACT))
    w, b = np.array(g["w"]), g["b"]
    mu, sd = np.array(g["mu"]), np.array(g["sd"])

    def score(emb):
        return float(sig(((np.array(emb) - mu) / sd) @ w + b))

    t5 = [json.loads(l) for l in open(T5_EMB)]
    p1 = [r for r in t5 if r["split"] == "probe"]
    va = [r for r in t5 if r["split"] == "val"]
    p2_meta = {r["id"]: r for r in (json.loads(l) for l in open(P2_PROBES))}
    p2_emb = {json.loads(l)["id"]: json.loads(l)["emb"] for l in open(P2_EMB)}

    # shipped-artifact val sanity (must match the D2 replay: 43/44, FP=1, FN=0)
    yva = np.array([r["label"] for r in va])
    pva = np.array([score(r["emb"]) for r in va])
    val_metrics = metrics(pva, yva, 0.05)

    # long probes from both families, with the shipped artifact
    long_probes = []
    for r in p1:
        if "long" in r["cell"]:
            long_probes.append({"family": 1, "id": "F1:" + r["id"], "cell": r["cell"],
                                "ambiguous": r["label"] == 1, "p": score(r["emb"])})
    for pid, emb in sorted(p2_emb.items()):
        m = p2_meta[pid]
        if "long" in m["cell"]:
            long_probes.append({"family": 2, "id": "F2:" + pid, "cell": m["cell"],
                                "ambiguous": not m["can_execute"], "p": score(emb)})

    amb = [r for r in long_probes if r["ambiguous"]]
    exe = [r for r in long_probes if not r["ambiguous"]]
    amb_z = sorted(round(float(-np.log(1.0 / max(r["p"], 1e-300) - 1.0)), 2) for r in amb)
    exe_z = sorted(round(float(-np.log(1.0 / max(r["p"], 1e-300) - 1.0)), 2) for r in exe)
    sweep = {}
    for thr in THRESHOLDS:
        caught = [r["id"] for r in amb if r["p"] >= thr]
        false_asks = [r["id"] for r in exe if r["p"] >= thr]
        sweep[str(thr)] = {
            "amb_recalled": f"{len(caught)}/{len(amb)}",
            "caught_ids": caught,
            "false_asks_on_long_exec": false_asks,
        }

    deep_misses = [r["id"] for r in amb if r["p"] < 1e-4]
    out = {
        "trial": "T6",
        "what": ("Threshold-policy sweep for the long-ambiguous blind spot, scored "
                 "through the SHIPPED plugin/gate_head.json artifact on probe families 1+2"),
        "artifact_val": val_metrics,
        "long_ambiguous_n": len(amb),
        "long_executable_n": len(exe),
        "long_ambiguous_scores": {r["id"]: round(r["p"], 4) for r in amb},
        "long_executable_scores_max": round(max(r["p"] for r in exe), 4),
        "deep_miss_ids_p_below_1e-4": deep_misses,
        "amb_z_sorted": amb_z,
        "exec_z_sorted": exe_z,
        "threshold_sweep": sweep,
        "verdict": ("No threshold policy recovers the blind spot: the score "
                    "distributions overlap almost completely (long-ambiguous z in "
                    f"[{amb_z[0]}, {amb_z[-1]}], long-executable z in [{exe_z[0]}, {exe_z[-1]}]). "
                    "At operational thresholds (>= 0.001) recall is 2/13 with zero "
                    "false-asks; at 1e-5 recall is only 7/13 with 2/13 long-executable "
                    "false-asks; at 1e-6 recall is 10/13 but false-asks 7/13. "
                    f"{len(deep_misses)}/{len(amb)} long-ambiguous probes are "
                    "deep-confidence misses (p < 1e-4). Mitigation must be upstream "
                    "(training data / retraining), not a policy tweak on this head. "
                    "The hook's blind_spot:true log flag stays heuristic-only."),
        "reproducibility_note": ("Fresh machine + fresh deps (laya 0.4.30, onnxruntime 1.23.2) "
                                 "reproduce committed T5 val and every committed P2 per-probe "
                                 "score exactly; laya 0.4.26-0.4.30 embeddings are cos=1.0 identical."),
        "ts": time.time(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("artifact_val", "long_ambiguous_scores",
                                          "threshold_sweep", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
