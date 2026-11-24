"""Nearest-centroid (class means) and 1-D sentiment-axis readouts per layer: how much sentiment info exists."""
import numpy as np
import analyze1 as a

TR = a.load_examples("train"); IDX = np.load(a.WORK / "train_idx_job1.npy"); TR = [TR[i] for i in IDX]
DV = a.load_examples("dev")
yt = np.array([e["shown_option_ids"][e["label"]] for e in TR]); yd = np.array([e["shown_option_ids"][e["label"]] for e in DV])
for L in (8, 12, 16, 20, 24, -1):
    ht = np.load(a.CACHE / f"base_train_h{L}.npy").astype(np.float32); hd = np.load(a.CACHE / f"base_dev_h{L}.npy").astype(np.float32)
    mu = ht.mean(0); s = ht.std(0) + 1e-3
    zt, zd = (ht - mu) / s, (hd - mu) / s
    C = np.stack([zt[yt == c].mean(0) for c in range(5)])
    pred = (-((zd[:, None, :] - C[None]) ** 2).sum(-1)).argmax(1)
    axis = C[4] - C[0]; axis /= np.linalg.norm(axis)
    pt, pdv = zt @ axis, zd @ axis
    m = np.array([pt[yt == c].mean() for c in range(5)]); th = (m[1:] + m[:-1]) / 2
    p1 = np.searchsorted(th, pdv)
    print(L, "centroid dev acc %.3f" % (pred == yd).mean(), "axis dev acc %.3f" % (p1 == yd).mean(), "axis class means", m.round(2))
