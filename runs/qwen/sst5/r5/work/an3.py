"""Finer class-bias headroom grid + save class centroids for steering."""
import itertools
import json
from pathlib import Path

import numpy as np

WORK = Path("wsp_work/sst5_r5")
CACHE = Path("wsp_work/sst5_r5_cache")
CENTROID_LAYERS = (16, 18, 20, 22)
N_CLASS = 5

logp = np.load(CACHE / "base_train_logp.npy")
n = len(logp)
exs = [json.loads(l) for l in open("data/processed/sst5/train.jsonl")][:n]
shown = np.array([e["shown_option_ids"] for e in exs])
label = np.array([e["label"] for e in exs])
true_cls = shown[np.arange(n), label]
cl = np.zeros_like(logp)
cl[np.arange(n)[:, None], shown] = logp
best = []
for bvn, bn, bvp in itertools.product((0, -2, -3, -4, -5, -6), (0, 2, 3, 4, 5), (0, -2, -3, -4, -5, -6)):
    bias = np.array([bvn, 0, bn, 0, bvp], float)
    a_all = ((cl + bias).argmax(1) == true_cls)
    best.append((a_all.mean(), a_all[:512].mean(), a_all[512:].mean(), bvn, bn, bvp))
best.sort(reverse=True)
for row in best[:12]:
    print(np.round(row, 4))
bias = np.array([-3, 0, 3, 0, -3], float)
pred = (cl + bias).argmax(1)
conf = np.zeros((5, 5), int); np.add.at(conf, (true_cls, pred), 1)
print(conf)
mus = []
for layer in CENTROID_LAYERS:
    h = np.load(CACHE / f"base_train_h{layer}.npy").astype(np.float32)
    mus.append(np.stack([h[true_cls == c].mean(0) for c in range(N_CLASS)]))
np.save(WORK / "centroids.npy", np.stack(mus).astype(np.float32))
print(np.stack(mus).shape)
