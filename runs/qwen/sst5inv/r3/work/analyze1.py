"""Local numpy analysis of job-1 captures: neutral direction separability and temperature."""
import json

import numpy as np

from common import WORK

LAYERS = (8, 12, 16, 20, 24)
NEUTRAL = 2
TEMPS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0)

examples = [json.loads(line) for line in open("data/processed/sst5inv/train.jsonl")]
idx = np.load(WORK / "j1_train_idx.npy")
logp = np.load(WORK / "j1_train_logp.npy").astype(np.float64)
labels = np.array([examples[i]["label"] for i in idx])
target = np.array([examples[i]["shown_option_ids"][examples[i]["label"]] for i in idx])
valence = target - NEUTRAL


def auc(score, positive):
    order = score.argsort()
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n_pos = positive.sum(); n_neg = len(score) - n_pos
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


half = np.arange(len(idx)) % 2 == 0
for layer in LAYERS:
    h = np.load(WORK.parent / "sst5inv_r3_cache" / f"j1_train_h{layer}.npy").astype(np.float64)
    is_neutral = target == NEUTRAL
    d = h[half & is_neutral].mean(0) - h[half & ~is_neutral].mean(0)
    score = h[~half] @ d
    extreme = np.abs(valence) == 2
    d_ext = h[half & ~extreme].mean(0) - h[half & extreme].mean(0)
    print(layer, "mean|h|", np.linalg.norm(h, axis=1).mean().round(1), "|d|", np.linalg.norm(d).round(2),
          "heldout AUC neutral", auc(score, is_neutral[~half]).round(3),
          "AUC mildness", auc(h[~half] @ d_ext, ~extreme[~half]).round(3),
          "cos(d,d_ext)", (d @ d_ext / np.linalg.norm(d) / np.linalg.norm(d_ext)).round(2))

for t in TEMPS:
    z = logp * t
    z = z - np.log(np.exp(z).sum(1, keepdims=True))
    print("temp", t, "train nll", (-z[np.arange(len(labels)), labels].mean()).round(3))
