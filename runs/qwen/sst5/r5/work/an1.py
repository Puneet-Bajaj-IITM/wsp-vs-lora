"""Local numpy analysis of job-1 arrays (no model)."""
import json
from pathlib import Path

import numpy as np

WORK = Path("wsp_work/sst5_r5")
CACHE = Path("wsp_work/sst5_r5_cache")
LAYERS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, -1)
N_CLASS = 5
TEMPS = (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07)

logp = np.load(CACHE / "base_train_logp.npy")
n = len(logp)
exs = [json.loads(l) for l in open("data/processed/sst5/train.jsonl")][:n]
shown = np.array([e["shown_option_ids"] for e in exs])
label = np.array([e["label"] for e in exs])
true_cls = shown[np.arange(n), label]
pred = logp.argmax(1)
pred_cls = shown[np.arange(n), pred]
print("acc", (pred == label).mean(), "nll", -logp[np.arange(n), label].mean())
conf = np.zeros((N_CLASS, N_CLASS), int)
np.add.at(conf, (true_cls, pred_cls), 1)
print("confusion rows=true cols=pred\n", conf)
print("pred class hist", conf.sum(0) / n, "true", conf.sum(1) / n)
print("pred letter hist", np.bincount(pred, minlength=5) / n)
for t in TEMPS:
    lp = logp * t
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    print("temp", t, "nll", -lp[np.arange(n), label].mean())
# class-indexed logp: cl[i, c] = logp of the letter showing class c
cl = np.zeros_like(logp)
cl[np.arange(n)[:, None], shown] = logp
print("mean class logp by true class\n", np.array([cl[true_cls == c].mean(0) for c in range(N_CLASS)]).round(2))
half = n // 2
for layer in LAYERS:
    h = np.load(CACHE / f"base_train_h{layer}.npy").astype(np.float32)
    mu = np.stack([h[:half][true_cls[:half] == c].mean(0) for c in range(N_CLASS)])
    g = h[:half].mean(0)
    x = h[half:] - g
    m = mu - g
    d = ((x[:, None, :] - m[None]) ** 2).sum(-1)
    acc_c = (d.argmin(1) == true_cls[half:]).mean()
    axis = mu[4] + mu[3] - mu[1] - mu[0]
    s = x @ axis / np.linalg.norm(axis)
    means = [s[true_cls[half:] == c].mean() for c in range(N_CLASS)]
    sd = s.std()
    print(f"layer {layer}: centroid acc {acc_c:.3f} | axis proj class means {np.round(means, 2)} sd {sd:.2f} | |h| {np.linalg.norm(h, axis=1).mean():.1f}")
