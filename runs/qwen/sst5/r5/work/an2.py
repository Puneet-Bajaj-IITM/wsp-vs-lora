"""Headroom of class-prior shifts (offline simulation on base logp) and geometry of class centroids."""
import json
from pathlib import Path

import numpy as np

WORK = Path("wsp_work/sst5_r5")
CACHE = Path("wsp_work/sst5_r5_cache")
N_CLASS = 5
NEUTRAL = 2
LAYERS = (14, 16, 18, 20, 22, 24, 26)

logp = np.load(CACHE / "base_train_logp.npy")
n = len(logp)
exs = [json.loads(l) for l in open("data/processed/sst5/train.jsonl")][:n]
shown = np.array([e["shown_option_ids"] for e in exs])
label = np.array([e["label"] for e in exs])
true_cls = shown[np.arange(n), label]
cl = np.zeros_like(logp)
cl[np.arange(n)[:, None], shown] = logp


def acc(bias):
    return ((cl + bias).argmax(1) == true_cls).mean()


for b in (0, 1, 2, 3, 4, 5, 6, 7, 8):
    bias = np.zeros(N_CLASS); bias[NEUTRAL] = b
    pred = (cl + bias).argmax(1)
    print("neutral bias", b, "acc", round(acc(bias), 4), "pred hist", np.bincount(pred, minlength=5) / n)
for bn in (3, 4, 5):
    for bv in (0, -1, -2, -3):
        bias = np.array([bv, 0, bn, 0, bv], float)
        print("neutral", bn, "very", bv, "acc", round(acc(bias), 4))

dirs = {}
for layer in LAYERS:
    h = np.load(CACHE / f"base_train_h{layer}.npy").astype(np.float32)
    mu = np.stack([h[true_cls == c].mean(0) for c in range(N_CLASS)])
    d_neu = mu[2] - (mu[1] + mu[3]) / 2
    d_pol = (mu[3] + mu[4] - mu[0] - mu[1]) / 2
    d_ext = (mu[0] + mu[4] - mu[1] - mu[3]) / 2
    dirs[layer] = (d_neu, d_pol, d_ext)
    out = [f"layer {layer}"]
    for name, d in (("neu", d_neu), ("pol", d_pol), ("ext", d_ext)):
        s = h @ d / np.linalg.norm(d)
        m = [s[true_cls == c].mean() for c in range(N_CLASS)]
        out.append(f"{name} |d|={np.linalg.norm(d):.1f} means {np.round(m, 1)} sd_within {np.mean([s[true_cls == c].std() for c in range(N_CLASS)]):.1f}")
    print(" | ".join(out))
    # 2-D centroid classifier headroom in (pol, neu) plane, split-half
    half = n // 2
    mu_h = np.stack([h[:half][true_cls[:half] == c].mean(0) for c in range(N_CLASS)])
    d = ((h[half:, None, :] - mu_h[None]) ** 2).sum(-1)
    pc = d.argmin(1)
    conf = np.zeros((5, 5), int); np.add.at(conf, (true_cls[half:], pc), 1)
    print("  centroid confusion\n", conf)
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
for k, name in enumerate(("neu", "pol", "ext")):
    print(name, "cos across layers vs 18:", {l: round(cos(dirs[l][k], dirs[18][k]), 2) for l in LAYERS})
print("cos neu-pol at 18", cos(dirs[18][0], dirs[18][1]))
