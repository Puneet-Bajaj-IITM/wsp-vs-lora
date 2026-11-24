"""'Leaning toward value v' directions from the base model's own soft choices (no labels), slot-balanced; cross-validated check."""
import numpy as np
from common import *

tr = fields(load("train")); H = np.load(CACHE / "h_train.npy", mmap_mode="r")
logp = np.load(CACHE / "logp_base_train.npy"); P = np.exp(logp)


def directions(X, mask, a):
    """(10, d): slot-balanced weighted mean difference for attribute a using examples in mask."""
    D = np.zeros((10, X.shape[1]), np.float32)
    for v in range(10):
        acc, n = 0, 0
        for s in range(5):
            m = mask & (tr["att"] == a) & (tr["shown"][:, s] == v)
            if m.sum() < 8: continue
            p = P[m, s][:, None]; Xm = X[m]
            acc = acc + (p * Xm).sum(0) / p.sum() - ((1 - p) * Xm).sum(0) / (1 - p).sum(); n += 1
        D[v] = acc / max(n, 1)
    return D


if __name__ == "__main__":
    f = tr["tpl"] < 2
    for L in (16, 17, 18, 19, 20):
        X = np.asarray(H[:, L], np.float32)
        cors, norms = [], []
        for a in range(3):
            D = directions(X, f, a); norms.append(np.linalg.norm(D, axis=1).mean())
            te = ~f & (tr["att"] == a)
            sc_all, lp_all = [], []
            for v in range(10):
                for s in range(5):
                    m = te & (tr["shown"][:, s] == v)
                    if m.sum() < 5: continue
                    sc = X[m] @ D[v]; lp = logp[m, s]
                    sc_all.append(sc - sc.mean()); lp_all.append(lp - lp.mean())   # removes popularity + slot prior
            cors.append(np.corrcoef(np.concatenate(sc_all), np.concatenate(lp_all))[0, 1])
        print(L, "held-out corr(score, logp of that option | value,slot):", np.round(cors, 3), "|d|", np.round(norms, 2), "|h|", np.linalg.norm(X, axis=1).mean().round(1))
    X = np.asarray(H[:, 20], np.float32)
    DV = np.stack([directions(X, np.ones(2700, bool), a) for a in range(3)])       # (3,10,d)
    np.save(WORK / "dv20.npy", DV.astype(np.float16))
    X = np.asarray(H[:, 18], np.float32)
    np.save(WORK / "dv18.npy", np.stack([directions(X, np.ones(2700, bool), a) for a in range(3)]).astype(np.float16))
    c = unit(DV.reshape(30, -1)) @ unit(DV.reshape(30, -1)).T
    print("mean |cos| between different value dirs", np.abs(c[~np.eye(30, dtype=bool)]).mean().round(3), "within-attr mean cos", np.mean([c[i, j] for i in range(30) for j in range(30) if i != j and i // 10 == j // 10]).round(3))
