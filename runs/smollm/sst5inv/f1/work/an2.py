"""Local: is intensity / neutrality linearly present in the answer-position residual? (split-half diff of means)"""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CACHE = WORK.parent / "smollm_sst5inv_f1_cache" / "job1"
LAYERS = (8, 12, 16, 20, 23, -1)
N = 5


def auc(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean())


def split_auc(h, a, b, rng, reps=20):
    """diff-of-means direction on a random half, AUC on the other half."""
    vals = []
    for _ in range(reps):
        m = rng.random(len(h)) < 0.5
        d = h[a & m].mean(0) - h[b & m].mean(0)
        s = h @ d
        vals.append(auc(s[a & ~m], s[b & ~m]))
    return float(np.mean(vals))


def main():
    exs = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:512]
    orig = np.array([N - 1 - e["shown_option_ids"][e["label"]] for e in exs])
    rng = np.random.default_rng(0)
    for run in ("base_train", "swap_train"):
        logp = np.load(WORK / "saved" / f"{run}_logp.npy")
        pred_letter = logp.argmax(1)
        print("==", run)
        for layer in LAYERS:
            h = np.load(CACHE / f"{run}_h{layer}.npy").astype(np.float32)
            norm = np.linalg.norm(h, axis=1).mean()
            mu = h.mean(0)
            hc = h - mu
            row = [f"L{layer:>2} |h| {norm:7.1f} |mu| {np.linalg.norm(mu):7.1f}"]
            row.append(f"pol {split_auc(hc, orig >= 3, orig <= 1, rng):.3f}")
            row.append(f"vneg/neg {split_auc(hc, orig == 0, orig == 1, rng):.3f}")
            row.append(f"vpos/pos {split_auc(hc, orig == 4, orig == 3, rng):.3f}")
            row.append(f"neu/rest {split_auc(hc, orig == 2, orig != 2, rng):.3f}")
            row.append(f"extreme/plain {split_auc(hc, (orig == 0) | (orig == 4), (orig == 1) | (orig == 3), rng):.3f}")
            for letter in range(N):
                row.append(f"let{letter} {split_auc(hc, pred_letter == letter, pred_letter != letter, rng):.2f}")
            print("  ".join(row))


main()
