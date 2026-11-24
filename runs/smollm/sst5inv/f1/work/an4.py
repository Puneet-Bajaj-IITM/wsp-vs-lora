"""Local: layer-by-layer emergence of valence and letter at the answer position (swap_v1, train probe)."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CACHE = WORK.parent / "smollm_sst5inv_f1_cache"
FILES = {12: "job1/swap_train_h12", 14: "job2/swapcap_train_h14", 15: "job2/swapcap_train_h15", 16: "job1/swap_train_h16",
         17: "job2/swapcap_train_h17", 18: "job2/swapcap_train_h18", 20: "job1/swap_train_h20", 23: "job1/swap_train_h23"}
N = 5


def auc(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean())


def split_auc(h, a, b, rng, reps=20):
    vals = []
    for _ in range(reps):
        m = rng.random(len(h)) < 0.5
        s = h @ (h[a & m].mean(0) - h[b & m].mean(0))
        vals.append(auc(s[a & ~m], s[b & ~m]))
    return float(np.mean(vals))


def main():
    exs = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:512]
    orig = np.array([N - 1 - e["shown_option_ids"][e["label"]] for e in exs])
    label = np.array([e["label"] for e in exs])
    logp = np.load(WORK / "saved" / "swap_train_logp.npy")
    pred = logp.argmax(1)
    pos_of = np.array([[e["shown_option_ids"].index(c) for c in range(N)] for e in exs])  # letter of each written class
    rng = np.random.default_rng(0)
    for layer, f in FILES.items():
        h = np.load(CACHE / f"{f}.npy").astype(np.float32)
        hc = h - h.mean(0)
        row = [f"L{layer:>2} |h| {np.linalg.norm(h, axis=1).mean():6.0f} pol {split_auc(hc, orig >= 3, orig <= 1, rng):.2f}"]
        row.append("pred-letter " + " ".join(f"{split_auc(hc, pred == l, pred != l, rng):.2f}" for l in range(N)))
        row.append("gold-letter " + " ".join(f"{split_auc(hc, label == l, label != l, rng):.2f}" for l in range(N)))
        for c, nm in ((0, "vneg"), (1, "neg"), (2, "neu"), (3, "pos"), (4, "vpos")):
            row.append(f"{nm}@A {split_auc(hc, pos_of[:, c] == 0, pos_of[:, c] != 0, rng):.2f}")
        print("  ".join(row))


main()
