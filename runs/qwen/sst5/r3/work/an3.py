"""Offline: label-free letter debias (match predicted letter marginal to uniform) on steered logp."""
import glob
import json
import os
import numpy as np
from labkit import CACHE, WORK, class_of

ex = [json.loads(l) for l in open(WORK.parent.parent / "data/processed/sst5/train.jsonl")][:2048]
labels = np.array([e["label"] for e in ex])
true_cls, shown = class_of(ex)


def debias(logp, iters=200, lr=0.5):
    """Iteratively shift letter logits until the soft predicted marginal is uniform. Uses no labels."""
    b = np.zeros(5)
    for _ in range(iters):
        z = logp + b
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        b -= lr * np.log(p.mean(0) * 5)
    return b - b.mean()


def acc(logp, idx, b=0):
    return float(((logp + b).argmax(1) == labels[idx]).mean())


base = np.load(CACHE / "base_logp_train.npy")
b = debias(base[:1024])
print("base: bias", b.round(2), "acc b0-1", acc(base[:1024], slice(0, 1024)), "->", acc(base[:1024], slice(0, 1024), b),
      "| held-out b2-3", acc(base[1024:], slice(1024, 2048)), "->", acc(base[1024:], slice(1024, 2048), b))
for f in sorted(glob.glob(str(CACHE / "steer_*_b0.npy"))):
    lp = np.load(f)
    b = debias(lp)
    f1 = f.replace("_b0", "_b1")
    extra = ""
    if os.path.exists(f1):
        lp1 = np.load(f1)
        extra = f"| b1 {acc(lp1, slice(512, 1024)):.3f} -> {acc(lp1, slice(512, 1024), b):.3f} (bias from b0)"
    pc = shown[np.arange(512), (lp + b).argmax(1)]
    print(os.path.basename(f)[6:-7].ljust(22), "bias", b.round(1), f"b0 {acc(lp, slice(0, 512)):.3f} -> {acc(lp, slice(0, 512), b):.3f}",
          "class_hist", np.bincount(pc, minlength=5), extra)
H = np.concatenate([np.load(CACHE / f"base_h-1_b{k}.npy") for k in range(4)]).astype(np.float64)
mu = H.mean(0)
proj = H @ mu / (mu @ mu)
print("constant-direction check: proj mean", proj.mean(), "sd", proj.std(), "|mu|", np.linalg.norm(mu))
