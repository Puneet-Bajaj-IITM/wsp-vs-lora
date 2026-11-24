"""Offline analysis of job 1: confusion in class space, bias/temperature ceilings, per-layer centroid decoding."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "wsp_work" / "sst5_r2_cache"
TRAIN = ROOT / "data" / "processed" / "sst5" / "train.jsonl"
N_CLASSES = 5
HALF = 512

logp = np.load(CACHE / "logp.npy")
hidden = np.load(CACHE / "hidden.npy").astype(np.float32)
n = len(logp)
ex = [json.loads(l) for l in TRAIN.open()][:n]
shown = np.array([e["shown_option_ids"] for e in ex])
label = np.array([e["label"] for e in ex])
true_cls = shown[np.arange(n), label]
cls_logp = np.zeros_like(logp)
np.put_along_axis(cls_logp, shown, logp, 1)
pred_cls = cls_logp.argmax(1)
print("acc", (pred_cls == true_cls).mean())
conf = np.zeros((N_CLASSES, N_CLASSES), int)
np.add.at(conf, (true_cls, pred_cls), 1)
print("confusion rows=true cols=pred\n", conf)
print("mean letter prob", np.exp(logp).mean(0).round(3), "mean class prob", np.exp(cls_logp).mean(0).round(3))


def nll_acc(lp, tgt):
    lp = lp - np.logaddexp.reduce(lp, 1, keepdims=True)
    return (lp.argmax(1) == tgt).mean().round(4), (-lp[np.arange(len(tgt)), tgt]).mean().round(4)


for t in (1, 2, 4, 8, 16):
    print("temp", t, nll_acc(cls_logp / t, true_cls))
prior = np.bincount(true_cls, minlength=N_CLASSES) / n
bias = np.zeros(N_CLASSES)
for temp in (1, 4, 8):
    bias = np.zeros(N_CLASSES)
    for _ in range(50):
        lp = cls_logp / temp + bias
        lp -= np.logaddexp.reduce(lp, 1, keepdims=True)
        bias += np.log(prior) - np.log(np.exp(lp).mean(0))
    print("prior-matched class bias at temp", temp, bias.round(2), nll_acc(cls_logp / temp + bias, true_cls))
for nb in (0, 2, 4, 6, 8, 10, 12):
    b = np.zeros(N_CLASSES); b[2] = nb
    print("neutral bias", nb, nll_acc(cls_logp + b, true_cls), np.bincount((cls_logp + b).argmax(1), minlength=5))

a, b_ = slice(0, HALF), slice(HALF, n)
print("layer: centroid acc true-class (fit first half, test second) | pred-class | letter-pred")
for li in range(hidden.shape[1]):
    h = hidden[:, li]
    out = []
    for tgt in (true_cls, pred_cls, logp.argmax(1)):
        cents = np.stack([h[a][tgt[a] == c].mean(0) for c in range(N_CLASSES)])
        hb = h[b_]
        d = ((hb[:, None] - cents[None]) ** 2).sum(-1)
        out.append(round(float((d.argmin(1) == tgt[b_]).mean()), 3))
    print(li, out, "norm", round(float(np.linalg.norm(h, axis=1).mean()), 1))
