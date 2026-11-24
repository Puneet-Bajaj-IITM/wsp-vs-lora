"""Whitened 'leaning' directions (readout estimate): held-out correlation with the option's log-prob vs ridge."""
import numpy as np
from common import *
from an9 import directions, tr, H, logp, P

f = tr["tpl"] < 2


def cv(L, ridges):
    X = np.asarray(H[:, L], np.float64)
    grp = tr["att"] * 3 + tr["tpl"]
    Xc = X - centroids(X.astype(np.float32), grp, 9)[grp]
    S = Xc[f].T @ Xc[f] / f.sum()
    ev, U = np.linalg.eigh(S)
    out = {}
    D = [directions(X.astype(np.float32), f, a).astype(np.float64) for a in range(3)]
    for r in ridges:
        inv = (U / (ev + r * ev.mean())) @ U.T
        cors = []
        for a in range(3):
            Wv = D[a] @ inv
            te = ~f & (tr["att"] == a); sc_all, lp_all = [], []
            for v in range(10):
                for s in range(5):
                    m = te & (tr["shown"][:, s] == v)
                    if m.sum() < 5: continue
                    sc = X[m] @ Wv[v]; lp = logp[m, s]
                    sc_all.append(sc - sc.mean()); lp_all.append(lp - lp.mean())
            cors.append(np.corrcoef(np.concatenate(sc_all), np.concatenate(lp_all))[0, 1])
        out[r] = np.round(cors, 3)
    return out


for L in (17, 19, 20):
    res = cv(L, (1e9, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01))
    print(f"L{L}:", {k: list(v) for k, v in res.items()})
