"""Simulated end accuracy of thresholded person detectors (ideal downstream): pick shown option with largest summed activation."""
import sys
import numpy as np
from common import *

tr = fields(load("train")); dv = fields(load("dev"))
G = np.load(CACHE / "norm_gains.npz")
Htr = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hdv = np.load(CACHE / "h_dev.npy", mmap_mode="r"); Hp = np.load(CACHE / "h_pres.npy", mmap_mode="r")
# fact table from TRAIN labels: (pid, attr) -> value id
facts = np.zeros((300, 3), int)
facts[tr["pid"], tr["att"]] = tr["shown"][np.arange(2700), tr["label"]]
true_d = dv["shown"][np.arange(900), dv["label"]]
assert (facts[dv["pid"], dv["att"]] == true_d).all()


def get(L):
    g = G[f"ln2_{L + 1}"]
    nm = lambda X: X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g
    return nm(np.asarray(Htr[:, L], np.float32)), nm(np.asarray(Hdv[:, L], np.float32)), nm(np.asarray(Hp[:, L], np.float32))


def fit(Xf, f, ridge, k_person=None):
    mu = Xf.mean(0)
    C = centroids(Xf - mu, f["pid"], 300)
    R = Xf - mu - C[f["pid"]]
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    lam = S ** 2 / len(Xf)
    coef = C @ Vt.T
    resid = C - coef @ Vt
    W = (coef / (lam + ridge * lam[0])) @ Vt + resid / (ridge * lam[0])
    W = W - np.outer(W @ mu, mu) / (mu @ mu)
    own = np.einsum("ij,ij->i", Xf, W[f["pid"]])
    return W / centroids(own[:, None], f["pid"], 300), mu


def sim(Sc, f, th, sharp=None):
    A = np.maximum(Sc - th, 0)
    if sharp: A = np.minimum(A, sharp)
    n = len(Sc); V = np.zeros((n, 10))
    vals = facts[:, :][:, f["att"]].T             # (n,300) value id of each person for this example's attribute
    for v in range(10):
        V[:, v] = (A * (vals == v)).sum(1)
    Vs = np.take_along_axis(V, f["shown"], 1)
    pred = Vs.argmax(1); none = Vs.max(1) <= 0
    acc = ((pred == f["label"]) & ~none).mean() + none.mean() * 0.2
    return acc, none.mean()


sub = lambda f, m: {k: v[m] for k, v in f.items()}
m01 = tr["tpl"] < 2
for L in (15, 16, 17):
    X, Xd, Xp = get(L)
    for ridge in (1e-4, 1e-5, 1e-6):
        W, mu = fit(X[m01], sub(tr, m01), ridge)
        W2, mu2 = fit(X, tr, ridge)
        Sl = X[~m01] @ W.T; Sd = Xd @ W2.T; Sp = Xp @ W2.T
        line = f"L{L} ridge {ridge}: NC acc LOTO {(Sl.argmax(1) == tr['pid'][~m01]).mean():.3f} dev {(Sd.argmax(1) == dv['pid']).mean():.3f} | pres max mean {Sp.max(1).mean():.2f} | sim acc (LOTO/dev, frac-none-dev):"
        for th in (0.3, 0.5, 0.7, 0.9):
            a1, _ = sim(Sl, sub(tr, ~m01), th); a2, nn = sim(Sd, dv, th)
            line += f"  th{th}: {a1:.3f}/{a2:.3f} ({nn:.2f})"
        print(line)
