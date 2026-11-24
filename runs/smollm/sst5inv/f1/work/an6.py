"""Local: with 1536 train examples, how well do dedicated intensity / neutral directions separate at L15, L16?"""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CACHE = WORK.parent / "smollm_sst5inv_f1_cache"
N = 5
PARTS = {15: ("job2/swapcap_train_h15", "job4/swapx1_train_h15", "job4/swapx2_train_h15"),
         16: ("job1/swap_train_h16", "job4/swapx1_train_h16", "job4/swapx2_train_h16")}


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
    exs = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:1536]
    orig = np.array([N - 1 - e["shown_option_ids"][e["label"]] for e in exs])
    rng = np.random.default_rng(0)
    for layer, files in PARTS.items():
        h = np.concatenate([np.load(CACHE / f"{f}.npy").astype(np.float32) for f in files])
        hc = h - h.mean(0)
        print(f"L{layer} n={len(h)} pol {split_auc(hc, orig >= 3, orig <= 1, rng):.3f} vneg/neg {split_auc(hc, orig == 0, orig == 1, rng):.3f} "
              f"vpos/pos {split_auc(hc, orig == 4, orig == 3, rng):.3f} neu/rest {split_auc(hc, orig == 2, orig != 2, rng):.3f} "
              f"neu/neg {split_auc(hc, orig == 2, orig == 1, rng):.3f} neu/pos {split_auc(hc, orig == 2, orig == 3, rng):.3f}")


main()
