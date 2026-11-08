#!/usr/bin/env python3
"""Trial P3: cross-encoder replication of the P2 failure.

Question: is the long-ambiguous failure specific to the LAYA encoder, or
does a general-purpose sentence embedding model show the same shortcut?
Method: replicate the exact T5 pipeline (normalize -> embed -> numpy
logistic head, config+threshold selected on train only) but swap the
frozen LAYA ONNX encoder for sentence-transformers/all-MiniLM-L6-v2
(384-d, mean-pooled, L2-normalized). Then score the FROZEN probe
families 1 (t2_probes.jsonl) and 2 (probe2_probes.jsonl, SHA-asserted).

If the MiniLM head also fails long-ambiguous, the dataset's surface
correlations are implicated rather than the specific encoder. If it
separates the P2 pairs, the LAYA encoder is implicated and full-encoder
training becomes more motivated. Diagnostic only; fully local, $0.
Model: public HF download (Apache-2.0), no account, no paid API.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_laya_embeddings import fit_logreg, metrics, wilson, REPO
from exp_t5_normalize import normalize

MINILM = os.path.expanduser("~/work/minilm")
EMB_OUT = os.path.join(REPO, "experiments", "p3_embeddings.jsonl")
OUT = os.path.join(REPO, "experiments", "p3_minilm_result.json")
P2_SHA = "594f3c560c047474fd35fe8d4de10cbbbb19ccc11ff78835477175fc5002059d"


def load_minilm():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(os.path.join(MINILM, "model.onnx"), sess_options=so,
                                providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(os.path.join(MINILM, "tokenizer.json"))
    tok.enable_truncation(max_length=256)
    return sess, tok


def embed_minilm(sess, tok, text):
    enc = tok.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    tt = np.zeros_like(ids)
    out = sess.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": tt})[0][0]
    m = mask[0].astype(np.float64)[:, None]
    pooled = (out * m).sum(axis=0) / max(m.sum(), 1e-8)
    n = np.linalg.norm(pooled)
    return pooled / max(n, 1e-8)


def all_rows():
    rows = []
    for split in ("train", "val"):
        for r in (json.loads(l) for l in open(os.path.join(REPO, "dataset", "splits", f"{split}.jsonl"))):
            rows.append({"split": split, "prompt": r["prompt"], "label": 0 if r["can_execute"] else 1})
    for fname, fam in (("t2_probes.jsonl", 1), ("probe2_probes.jsonl", 2)):
        for r in (json.loads(l) for l in open(os.path.join(REPO, "experiments", fname))):
            rows.append({"split": f"probe{fam}", "prompt": r["prompt"],
                         "label": 0 if r["can_execute"] else 1, "id": r["id"], "cell": r["cell"]})
    return rows


def extract():
    rows = all_rows()
    sess, tok = load_minilm()
    done = set()
    if os.path.exists(EMB_OUT):
        done = {(json.loads(l)["split"], json.loads(l).get("id", json.loads(l)["norm_prompt"]))
                for l in open(EMB_OUT)}
    out = open(EMB_OUT, "a", buffering=1)
    for r in rows:
        np_ = normalize(r["prompt"])
        key = (r["split"], r.get("id", np_))
        if key in done:
            continue
        vec = embed_minilm(sess, tok, np_)
        rec = {"split": r["split"], "norm_prompt": np_, "label": r["label"],
               "emb": [round(float(x), 6) for x in vec]}
        if "id" in r:
            rec["id"], rec["cell"] = r["id"], r["cell"]
        out.write(json.dumps(rec) + "\n")
        print(f"[{r['split']}] embedded", flush=True)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run():
    import hashlib
    p2 = open(os.path.join(REPO, "experiments", "probe2_probes.jsonl"), "rb").read()
    assert hashlib.sha256(p2).hexdigest() == P2_SHA, "probe2 file changed since freeze"

    data = {"train": [], "val": [], "probe1": [], "probe2": []}
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

    def score_probes(rows):
        cells, per = {}, []
        for r in rows:
            x = (np.array(r["emb"]) - mu) / sd
            p = float(sig(x @ w + b))
            ok = (p >= thr) == bool(r["label"])
            cells.setdefault(r["cell"], []).append(ok)
            per.append({"id": r["id"], "cell": r["cell"], "p_ambiguous": round(p, 4), "correct": ok})
        return {c: {"acc": round(sum(v) / len(v), 3), "n": len(v)} for c, v in sorted(cells.items())}, per

    c1, per1 = score_probes(data["probe1"])
    c2, per2 = score_probes(data["probe2"])
    correct = int(round(vm["acc"] * vm["n"]))
    out = {
        "trial": "P3",
        "what": "cross-encoder replication: T5 pipeline with all-MiniLM-L6-v2 instead of LAYA ONNX; scored on frozen probe families 1+2",
        "encoder": "sentence-transformers/all-MiniLM-L6-v2 (ONNX, 384-d, mean-pooled, L2-norm, truncation 256)",
        "selected_config": {"lr": lr, "l2": l2, "thr": thr},
        "train": tm, "val": vm, "val_acc_wilson95": wilson(correct, vm["n"]),
        "probe1_cells": c1, "probe2_cells": c2,
        "probe2_per_probe": per2,
        "status": "DEVELOPMENT diagnostics - cross-encoder comparison, not an acceptance test",
        "caveats": [
            "Val reused across trials - directional.",
            "MiniLM truncates at 256 tokens vs LAYA's longer window; long-prompt behavior partly reflects truncation.",
            "Different embedding model, same dataset: a shared failure implicates the data's surface correlations, not proof.",
        ],
        "ts": time.time(),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("train:", tm, "\nval:", vm)
    print("probe1:", json.dumps(c1))
    print("probe2:", json.dumps(c2))
    for p in per2:
        if not p["correct"]:
            print("P2 FAIL:", json.dumps(p))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract()
    else:
        run()
