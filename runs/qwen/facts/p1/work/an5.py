"""Template-shift diagnostics and noise-whitened detectors at L16 (honest: no statistics from the test template)."""
import numpy as np
from common import *

tr = fields(load("train")); dv = fields(load("dev"))
G = np.load(CACHE / "norm_gains.npz")
L = 16
g = G[f"ln2_{L + 1}"]


def normed(X):
    return X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g


X = normed(np.asarray(np.load(CACHE / "h_train.npy", mmap_mode="r")[:, L], np.float32))
Xd = normed(np.asarray(np.load(CACHE / "h_dev.npy", mmap_mode="r")[:, L], np.float32))
Xp = normed(np.asarray(np.load(CACHE / "h_pres.npy", mmap_mode="r")[:, L], np.float32))


def fit(Xf, f, mode, m, ridge):
    """Detectors from examples Xf with fields f. mode: 'proj' (remove top-m noise PCs) or 'whiten' (noise-cov whitening, rank m)."""
    mu = Xf.mean(0)
    C = centroids(Xf - mu, f["pid"], 300)
    R = Xf - mu - C[f["pid"]]                       # everything that is not person: template, attribute, options
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    lam = S ** 2 / len(Xf)
    if mode == "proj":
        V = Vt[:m]; W = C - (C @ V.T) @ V
    else:
        V = Vt[:m]; tail = lam[m:].mean() if m < len(lam) else lam[-1]
        coef = C @ V.T
        W = (coef / (lam[:m] + ridge * lam[0])) @ V + (C - coef @ V) / (tail + ridge * lam[0])
    W = W - np.outer(W @ mu, mu) / (mu @ mu)
    own = np.einsum("ij,ij->i", Xf, W[f["pid"]])
    return W / centroids(own[:, None], f["pid"], 300), lam


def ev(Z, pid, W):
    Sc = Z @ W.T; n = len(Z)
    own = Sc[np.arange(n), pid]; S2 = Sc.copy(); S2[np.arange(n), pid] = -9
    return f"acc {(Sc.argmax(1) == pid).mean():.3f} own p10 {np.percentile(own, 10):.2f} other p90 {np.percentile(S2.max(1), 90):.2f} own>.6 {(own > .6).mean():.2f} oth>.6/ex {(S2 > .6).sum(1).mean():.2f}"


sub = lambda f, m: {k: v[m] for k, v in f.items()}
m01 = tr["tpl"] < 2
# diagnostics: dev shift
d = Xd.mean(0) - X.mean(0)
gm = centroids(X, tr["att"] * 3 + tr["tpl"], 9); Q, _ = np.linalg.qr(np.concatenate([gm]).T)
print("dev shift norm", np.linalg.norm(d), "outside train group-mean span", np.linalg.norm(d - Q @ (Q.T @ d)), "typ |x|", np.linalg.norm(X, axis=1).mean(),
      "person centroid norm", np.linalg.norm(centroids(X - X.mean(0), tr["pid"], 300), axis=1).mean())
for mode, m, ridge in (("proj", 150, 0), ("proj", 400, 0), ("proj", 800, 0), ("whiten", 100, .01), ("whiten", 300, .01),
                       ("whiten", 300, .001), ("whiten", 800, .001), ("whiten", 800, .0001), ("whiten", 1400, .0001)):
    W, lam = fit(X[m01], sub(tr, m01), mode, m, ridge)
    W2, _ = fit(X, tr, mode, m, ridge)
    print(f"{mode} m={m} ridge={ridge}\n   LOTO: {ev(X[~m01], tr['pid'][~m01], W)}\n   dev : {ev(Xd, dv['pid'], W2)}   pres max p99 {np.percentile((Xp @ W2.T).max(1), 99):.2f} mean {(Xp @ W2.T).max(1).mean():.2f}")
