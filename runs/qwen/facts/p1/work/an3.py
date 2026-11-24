"""Person keys: PCA-whitened centroid detectors; separation stats; cross-layer transfer; preserve-set leakage."""
import numpy as np
from common import *

tr = fields(load("train")); dv = fields(load("dev"))
H = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hd = np.load(CACHE / "h_dev.npy", mmap_mode="r")
Hp = np.load(CACHE / "h_pres.npy", mmap_mode="r")
G = np.load(CACHE / "norm_gains.npz")


def normed(X, gain):
    return X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * gain


def report(Lk, Lt, k, ridge):
    g = G[f"ln2_{Lk + 1}"]
    X = normed(np.asarray(H[:, Lk], np.float32), g); mu = X.mean(0)
    Xt = normed(np.asarray(Hd[:, Lt], np.float32), g); P = normed(np.asarray(Hp[:, Lt], np.float32), g)
    Xtr_t = normed(np.asarray(H[:, Lt], np.float32), g)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]; lam = (S[:k] ** 2) / len(X)
    C = centroids(Xc, tr["pid"], 300) @ V.T
    W = (C / (lam + ridge * lam.mean())) @ V          # whitened detectors, rows = persons
    own_tr = np.einsum("ij,ij->i", Xc, W[tr["pid"]])
    W = W / centroids(own_tr[:, None], tr["pid"], 300)  # scale: mean own train score = 1
    for name, Z, pid in (("dev", Xt - mu, dv["pid"]), ("train@Lt", Xtr_t - mu, tr["pid"])):
        Sc = Z @ W.T
        own = Sc[np.arange(len(Z)), pid]; Sc2 = Sc.copy(); Sc2[np.arange(len(Z)), pid] = -9
        oth = Sc2.max(1)
        print(f"  {name}: acc {(Sc.argmax(1) == pid).mean():.3f} own mean {own.mean():.2f} p10 {np.percentile(own, 10):.2f} "
              f"| best-other mean {oth.mean():.2f} p90 {np.percentile(oth, 90):.2f} p99 {np.percentile(oth, 99):.2f} "
              f"| frac own>0.5 {(own > .5).mean():.3f}, #others>0.5 per ex {(Sc2 > .5).sum(1).mean():.3f}")
    Sp = (P - mu) @ W.T
    print(f"  preserve: max score mean {Sp.max(1).mean():.2f} p99 {np.percentile(Sp.max(1), 99):.2f}; frac any>0.5 {(Sp.max(1) > .5).mean():.3f}; mu-proj task {(X @ mu).mean() / (mu @ mu):.2f} pres {((P) @ mu).mean() / (mu @ mu):.2f}")


for Lk, Lt in ((16, 16), (16, 17), (15, 15), (15, 16), (17, 17)):
    for k, ridge in ((64, 0.1), (200, 0.1), (400, 0.3)):
        print(f"keys L{Lk} -> test L{Lt}, k={k}, ridge={ridge}")
        report(Lk, Lt, k, ridge)
