"""Plain centroid detectors with nuisance projected out; honest dev test (no dev statistics); transfer to next layer."""
import numpy as np
from common import *

tr = fields(load("train")); dv = fields(load("dev"))
H = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hd = np.load(CACHE / "h_dev.npy", mmap_mode="r")
Hp = np.load(CACHE / "h_pres.npy", mmap_mode="r")
G = np.load(CACHE / "norm_gains.npz")


def normed(X, gain):
    return X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * gain


def proj_out(X, B):
    Q, _ = np.linalg.qr(B.T)
    return X - (X @ Q) @ Q.T


def build(Lk, n_noise, extra_layers=()):
    g = G[f"ln2_{Lk + 1}"]
    X = normed(np.asarray(H[:, Lk], np.float32), g); mu = X.mean(0)
    grp = tr["att"] * 3 + tr["tpl"]
    gm = centroids(X, grp, 9)
    Xc = X - gm[grp]
    C = centroids(Xc, tr["pid"], 300)
    R = Xc - C[tr["pid"]]                       # within-person residual (option content etc.)
    nuis = [gm, mu[None]]
    for Le in extra_layers:
        Xe = normed(np.asarray(H[:, Le], np.float32), g)
        nuis += [centroids(Xe, grp, 9), Xe.mean(0)[None]]
    if n_noise:
        nuis.append(np.linalg.svd(R, full_matrices=False)[2][:n_noise])
    B = np.concatenate(nuis)
    W = proj_out(C, B)
    own = np.einsum("ij,ij->i", X, W[tr["pid"]])
    W = W / centroids(own[:, None], tr["pid"], 300)
    return W, mu, g


def stats(name, Z, W, pid=None):
    Sc = Z @ W.T
    if pid is None:
        mx = Sc.max(1)
        print(f"  {name}: max score mean {mx.mean():.2f} p99 {np.percentile(mx, 99):.2f} frac>0.5 {(mx > .5).mean():.3f}")
        return
    n = len(Z); own = Sc[np.arange(n), pid]; S2 = Sc.copy(); S2[np.arange(n), pid] = -9; oth = S2.max(1)
    th = 0.5
    print(f"  {name}: acc {(Sc.argmax(1) == pid).mean():.3f} own mean {own.mean():.2f} p10 {np.percentile(own, 10):.2f} "
          f"| best-other mean {oth.mean():.2f} p90 {np.percentile(oth, 90):.2f} | own>{th}: {(own > th).mean():.3f}, "
          f"others>{th}/ex {(S2 > th).sum(1).mean():.3f}")


for Lk in (15, 16, 17):
    for n_noise, extra in ((0, ()), (20, ()), (60, ()), (20, (Lk + 1,)), (150, ())):
        W, mu, g = build(Lk, n_noise, extra)
        print(f"L{Lk} noise-PCs {n_noise} extra {extra}")
        stats("train", normed(np.asarray(H[:, Lk], np.float32), g), W, tr["pid"])
        stats("dev  ", normed(np.asarray(Hd[:, Lk], np.float32), g), W, dv["pid"])
        if Lk < 17:
            stats(f"dev@L{Lk+1}", normed(np.asarray(Hd[:, Lk + 1], np.float32), g), W, dv["pid"])
        stats("pres ", normed(np.asarray(Hp[:, Lk], np.float32), g), W)
