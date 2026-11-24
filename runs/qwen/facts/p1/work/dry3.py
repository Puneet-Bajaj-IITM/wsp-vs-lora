"""Tune ramp threshold/width and ridge on the h16 proxy (dev with all-train detectors; LOTO)."""
import numpy as np
from common import load, CACHE
from wlib import *

tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
G = np.load(CACHE / "norm_gains.npz"); g = G["ln2_17"].astype(np.float64)
X = mlp_input(np.asarray(np.load(CACHE / "h_train.npy", mmap_mode="r")[:, 16], np.float32), g)
Xd = mlp_input(np.asarray(np.load(CACHE / "h_dev.npy", mmap_mode="r")[:, 16], np.float32), g)
Xp = mlp_input(np.asarray(np.load(CACHE / "h_pres.npy", mmap_mode="r")[:, 16], np.float32), g)
m01 = tr["tpl"] < 2; sub = {k: v[~m01] for k, v in tr.items()}
for ridge in (1e-5, 1e-6, 1e-7):
    W = build_detectors(X, tr["pid"], Xp, ridge)[0]; Wl = build_detectors(X[m01], tr["pid"][m01], Xp, ridge)[0]
    Sd = Xd @ W.T; Sl = X[~m01] @ Wl.T
    for theta in (0.4, 0.5, 0.6, 0.7):
        row = f"ridge {ridge} theta {theta}:"
        for delta in (0.15, 0.4, 1.0):
            act = lambda s: soft_act(s, beta=40.0, theta=theta, delta=delta)
            row += f"  d{delta}: dev {sim_accuracy(Sd, dv, facts, act):.3f} loto {sim_accuracy(Sl, sub, facts, act):.3f}"
        print(row)
