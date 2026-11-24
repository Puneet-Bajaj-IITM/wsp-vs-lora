"""Is 'neutral' linearly visible at the answer position? Diff-of-means on train A, checked on train B."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "sst5inv"
CACHE = ROOT / "wsp_work" / "sst5inv_r2_cache" / "job1"
LAYERS = (12, 16, 20, 24, 27)
BLOCKS = ["very negative", "negative", "neutral", "positive", "very positive"]


def auc(scores, positive):
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


if __name__ == "__main__":
    exs = [json.loads(line) for line in (DATA / "train.jsonl").read_text().splitlines()[:1024]]
    block = np.array([BLOCKS.index(e["block"]) for e in exs])
    a, b = block[:512], block[512:]
    for layer in LAYERS:
        ha = np.load(CACHE / f"hid_trainA_L{layer}.npy").astype(np.float32)
        hb = np.load(CACHE / f"hid_trainB_L{layer}.npy").astype(np.float32)
        centre = np.stack([ha[a == c].mean(0) for c in range(5)]).mean(0)
        d_neu = ha[a == 2].mean(0) - ha[a != 2].mean(0)
        d_val = ha[a >= 3].mean(0) - ha[a <= 1].mean(0)
        proj_val = (hb - centre) @ d_val / np.linalg.norm(d_val)
        print(f"L{layer} |h| {np.linalg.norm(ha, axis=1).mean():.1f} |d_neu| {np.linalg.norm(d_neu):.2f} "
              f"|d_val| {np.linalg.norm(d_val):.2f} heldout AUC neutral {auc(hb @ d_neu, b == 2):.3f} "
              f"valence {auc(hb @ d_val, b >= 3):.3f} (non-neutral only "
              f"{auc((hb @ d_val)[b != 2], (b >= 3)[b != 2]):.3f}) "
              f"neutral-by-|valence| {auc(-np.abs(proj_val), b == 2):.3f} "
              f"valence proj means by class {np.round([proj_val[b == c].mean() for c in range(5)], 2)}")
