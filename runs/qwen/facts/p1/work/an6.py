"""Factorised person keys: marginal means for first name and each syllable slot."""
import json, re
import numpy as np
from common import *

SYL = ["ka", "lo", "mir", "ven", "tas", "ru", "del", "nor", "bi", "quen", "sha", "vor", "lei", "dun", "pra", "zel", "om", "tri", "gal", "fen"]
FIRST = ["Arlen", "Bexa", "Corin", "Dalia", "Emric", "Fenna", "Gorin", "Hesta", "Ilan", "Jorna", "Kestrel", "Lunae", "Marrow", "Nessa", "Orrin", "Pella", "Quill", "Rhosyn", "Sorrel", "Tamsin"]
meta = json.loads((WORK / "meta.json").read_text())
names = meta["names"]


def split_syl(s):
    s = s.lower(); out = []
    def rec(rest, acc):
        if not rest: return acc if len(acc) == 3 else None
        for i, y in enumerate(SYL):
            if rest.startswith(y):
                r = rec(rest[len(y):], acc + [i])
                if r: return r
        return None
    return rec(s, [])


F = np.array([[FIRST.index(n.split()[0])] + split_syl(n.split()[1]) for n in names])   # (300,4)
print("names", names[:3], F[:3], "tokens", [meta["name_ids"][i] for i in range(3)])
tr = fields(load("train")); dv = fields(load("dev"))
G = np.load(CACHE / "norm_gains.npz")
Htr = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hdv = np.load(CACHE / "h_dev.npy", mmap_mode="r")


def run(L, verbose=True):
    g = G[f"ln2_{L + 1}"]
    nm = lambda X: X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g
    X = nm(np.asarray(Htr[:, L], np.float32)); Xd = nm(np.asarray(Hdv[:, L], np.float32))

    def templates(Xf, f):
        mu = Xf.mean(0); Xc = Xf - mu
        E = [centroids(Xc, F[f["pid"], j], 20) for j in range(4)]
        T = sum(E[j][F[:, j]] for j in range(4))
        return T, mu, E

    def ev(Z, pid, T, tag):
        Tn = T / (T ** 2).sum(1, keepdims=True)
        Sc = Z @ Tn.T; n = len(Z); own = Sc[np.arange(n), pid]; S2 = Sc.copy(); S2[np.arange(n), pid] = -9
        cos = unit(Z - Z.mean(0)) @ unit(T).T
        print(f"  {tag}: acc(dot,no test stats) {(Sc.argmax(1) == pid).mean():.3f} | acc(cos, test-centred) {(cos.argmax(1) == pid).mean():.3f} | own mean {own.mean():.2f} p10 {np.percentile(own, 10):.2f} "
              f"other max mean {S2.max(1).mean():.2f} p90 {np.percentile(S2.max(1), 90):.2f} | |T| {np.linalg.norm(T, axis=1).mean():.2f}")

    m01 = tr["tpl"] < 2
    T, mu, E = templates(X[m01], {k: v[m01] for k, v in tr.items()})
    print(f"L{L}")
    ev(X[~m01] - mu, tr["pid"][~m01], T, "LOTO")
    T, mu, E = templates(X, tr)
    ev(Xd - mu, dv["pid"], T, "dev ")
    return T, mu, E


for L in (14, 15, 16, 17, 18, 20):
    run(L)
