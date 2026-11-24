"""Offline: can the answer-position residual linearly carry per-letter option-text offsets?
Readout = cross-covariance of centred hidden with target offsets (Hebbian), optionally in top-k PCA space."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "wsp_work/twin_o1")
from wsp.catalog import load_items
A = Path("wsp_work/twin_o1_cache/arrays1")
items = load_items()
dv = [json.loads(l) for l in open("data/processed/dev.jsonl")]
tr = [json.loads(l) for l in open("data/processed/train.jsonl")]
tidx = np.load(A/"train_idx.npy"); trs = [tr[i] for i in tidx]
tl, dl = np.load(A/"train_logp.npy"), np.load(A/"dev_logp.npy")
T = 6.0
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def texts(e): return [items[e["column"]].options[i] for i in e["shown_option_ids"]]
def met(lp, exs):
    y = np.array([e["label"] for e in exs]); return round(float((lp.argmax(1)==y).mean()),4), round(float(-lp[np.arange(len(y)),y].mean()),4)
bias = -np.log(np.exp(lsm(tl/T)).mean(0))
P = np.exp(lsm(tl/T + bias))
sh = collections.Counter(); ch = collections.Counter(); m = collections.defaultdict(float)
for e, p in zip(trs, P):
    for k, t in enumerate(texts(e)):
        sh[t] += 1; m[t] += p[k]; ch[t] += (k == e["label"])
off = {t: np.log((ch[t] + 4) / (m[t] + 4)) for t in sh}
def target(exs):
    o = np.array([[off.get(t, 0.0) for t in texts(e)] for e in exs]); return o - o.mean(1, keepdims=True)
Ttr, Tdv = target(trs), target(dv)
base = lsm(dl/T + bias)
print("dev T+bias", met(base, dv), " oracle offsets", met(lsm(dl/T + bias + Tdv), dv))
for lay in (16, 20, 23, 25, 27, -1):
    Htr = np.load(A/f"train_h{lay}.npy").astype(np.float64); Hdv = np.load(A/f"dev_h{lay}.npy").astype(np.float64)
    if lay != -1:
        Htr /= np.sqrt((Htr**2).mean(1, keepdims=True)); Hdv /= np.sqrt((Hdv**2).mean(1, keepdims=True))
    mu = Htr.mean(0); Xtr, Xdv = Htr - mu, Hdv - mu
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    for k in (32, 128, 512):
        V = Vt[:k]; Ztr, Zdv = Xtr @ V.T / S[:k], Xdv @ V.T / S[:k]   # whitened PCA coords
        W = Ttr.T @ Ztr                                               # cross-covariance readout
        str_, sdv = Ztr @ W.T, Zdv @ W.T
        r2 = 1 - ((Tdv - sdv) ** 2).sum() / (Tdv ** 2).sum()
        best = max(((a, met(lsm(dl/T + bias + a * sdv), dv)) for a in (0.25, 0.5, 1, 1.5, 2)), key=lambda x: x[1][0])
        print(f"layer {lay:3d} k={k:3d} dev R2={r2:+.3f} best alpha/metrics {best}")
