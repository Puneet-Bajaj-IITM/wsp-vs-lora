"""In-situ check of the job-2 person neurons: what did MLP-17 actually add to h17?"""
import json
import numpy as np
from common import load, CACHE, WORK
from wlib import *

tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
G = np.load(CACHE / "norm_gains.npz"); g = G["ln2_17"].astype(np.float64)
DV20 = np.load(WORK / "dv20.npy").astype(np.float64)
X = mlp_input(np.load(CACHE / "mid17_train.npy"), g); Xd = mlp_input(np.load(CACHE / "mid17_dev.npy"), g); Xp = mlp_input(np.load(CACHE / "mid17_pres.npy"), g)
W, mu, mhat, a = build_detectors(X, tr["pid"], Xp)
for split, f, Xs, Hname in (("dev", dv, Xd, "h_dev.npy"), ("train", tr, X, "h_train.npy")):
    ix = np.load(WORK / f"idx_person_{split}.npy")
    h17 = np.load(CACHE / f"h17_person_{split}.npy").astype(np.float64)
    base17 = np.asarray(np.load(CACHE / Hname, mmap_mode="r")[ix, 17], np.float64)
    delta = h17 - base17
    S = Xs[ix] @ W.T
    A = soft_act(S, ctx=(Xs[ix] @ a)[:, None])                      # predicted activations (units of V)
    pid = f["pid"][ix]; grp = pid % 4
    print(split, "|delta| mean", np.linalg.norm(delta, axis=1).mean().round(2), "|base h17|", np.linalg.norm(base17, axis=1).mean().round(1),
          "pred own act mean", A[np.arange(len(ix)), pid].mean().round(2), "pred total act", A.sum(1).mean().round(2))
    # B groups (0: k4, 1: k12, 3: k30): projection of delta on own value direction for this attribute
    for k, kap in ((0, 4), (1, 12), (3, 30)):
        m = grp == k
        d_own = np.stack([DV20[at, facts[p, at]] for p, at in zip(pid[m], f["att"][ix][m])])
        coef = np.einsum("ij,ij->i", delta[m], d_own) / (d_own ** 2).sum(1)
        print(f"   group k{kap}: delta.d_own/|d|^2 mean {coef.mean():.2f} (expected ~{kap} x act); |delta| {np.linalg.norm(delta[m], axis=1).mean():.2f}; pred own act {A[m, pid[m]].mean():.2f}")
    lp = np.load(WORK / f"logp_person_{split}.npy"); base = np.load(WORK / f"logp_base_{split}.npy")[ix]
    lab = f["label"][ix]
    print("   dlogp(true) by group:", [round(float((lp[np.arange(len(ix)), lab] - base[np.arange(len(ix)), lab])[grp == k].mean()), 2) for k in range(4)],
          "acc base", (base.argmax(1) == lab).mean().round(3))
