"""Offline: geometry of class centroids at pre-binding layers (15-20)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "wsp_work" / "sst5_r2_cache"
TRAIN = ROOT / "data" / "processed" / "sst5" / "train.jsonl"
N_CLASSES = 5
NEUTRAL = 2
HALF = 512
LAYERS = (15, 16, 17, 18, 19, 20)

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
a, b = np.arange(n) < HALF, np.arange(n) >= HALF
np.set_printoptions(precision=2, suppress=True, linewidth=160)

for li in LAYERS:
    h = hidden[:, li]
    mu = np.stack([h[a & (true_cls == c)].mean(0) for c in range(N_CLASSES)])
    d = ((h[b][:, None] - mu[None]) ** 2).sum(-1)
    conf = np.zeros((N_CLASSES, N_CLASSES), int)
    np.add.at(conf, (true_cls[b], d.argmin(1)), 1)
    print(f"\nlayer {li} centroid confusion (rows true):\n{conf}")
    mu_all = np.stack([h[true_cls == c].mean(0) for c in range(N_CLASSES)])
    centred = mu_all - mu_all.mean(0)
    print("centroid offset norms", np.linalg.norm(centred, axis=1), "within-class std norm",
          np.sqrt(np.mean([((h[true_cls == c] - mu_all[c]) ** 2).sum(1).mean() for c in range(N_CLASSES)])))
    cos = centred @ centred.T / np.outer(np.linalg.norm(centred, axis=1), np.linalg.norm(centred, axis=1))
    print("cos between centroid offsets\n", cos)
    sv = np.linalg.svd(centred, compute_uv=False)
    print("centroid singular values", sv)
    neu_dir = mu_all[NEUTRAL] - np.delete(mu_all, NEUTRAL, 0).mean(0)
    unit = neu_dir / np.linalg.norm(neu_dir)
    proj = (h - mu_all.mean(0)) @ unit
    print("neutral dir norm", np.linalg.norm(neu_dir), "proj mean by true class", [proj[true_cls == c].mean() for c in range(N_CLASSES)],
          "std", proj.std())
    print("proj mean by pred class", [proj[pred_cls == c].mean() for c in range(N_CLASSES)])
    held = h[b] - mu.mean(0)
    held_dir = mu[NEUTRAL] - np.delete(mu, NEUTRAL, 0).mean(0)
    hp = held @ held_dir / np.linalg.norm(held_dir)
    print("held-out proj on first-half neutral dir by true class", [hp[true_cls[b] == c].mean() for c in range(N_CLASSES)])
    m_pred = h[pred_cls == NEUTRAL].mean(0) - h[pred_cls != NEUTRAL].mean(0)
    print("cos(true-neutral dir, pred-neutral dir)", m_pred @ unit / np.linalg.norm(m_pred), "pred-neutral dir norm", np.linalg.norm(m_pred))
