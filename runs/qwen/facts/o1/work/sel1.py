"""Selectivity of centered person keys on other names and random positions; constant-direction check."""
import numpy as np, common as C
idx = np.load(C.CACHE / "j1_idx.npy")
import json
from wsp.metrics import load_split
tr = load_split("train"); pids = np.array([tr[i]["pid"] for i in idx]); tmpl = np.array([tr[i]["template"] for i in idx])
cols = np.array([tr[i]["column"] for i in idx])
for l in (2, 4, 6, 8):
    x = np.load(C.CACHE / f"j1_keyx_L{l}.npy").astype(np.float64)
    r = np.load(C.CACHE / f"j1_rndx_L{l}.npy").astype(np.float64).reshape(-1, 1536)
    mu = x.mean(0)
    allx = np.concatenate([x, r])
    m = allx.mean(0); mh = m / np.linalg.norm(m)
    pr = allx @ mh
    print(f"L{l} |x_name|={np.linalg.norm(x,axis=1).mean():.1f} |x_rnd|={np.linalg.norm(r,axis=1).mean():.1f} "
          f"const-dir proj name {np.mean(x@mh):.2f}+-{np.std(x@mh):.2f} rnd {np.mean(r@mh):.2f}+-{np.std(r@mh):.2f}")
    # keys from first two thirds, test on held-out third (different examples)
    tr_mask = np.arange(len(x)) < 1024
    K = np.zeros((300, 1536)); have = np.zeros(300, bool)
    for p in range(300):
        s = tr_mask & (pids == p)
        if s.any(): K[p] = x[s].mean(0) - mu; have[p] = True
    Kp = K - np.outer(K @ (mu/np.linalg.norm(mu)), mu/np.linalg.norm(mu))  # orthogonal to name mean
    for name, KK in (("centered", K), ("orth_mu", Kp)):
        n2 = (KK**2).sum(1) + 1e-9
        te = (~tr_mask) & have[pids]
        S = (x[te] - (mu if name == "centered" else 0)) @ KK.T / n2   # (n, 300)
        tgt = S[np.arange(te.sum()), pids[te]]
        off = S.copy(); off[np.arange(te.sum()), pids[te]] = np.nan
        Sr = (r - (mu if name == "centered" else 0)) @ KK.T / n2
        acc = (np.nanargmax(np.where(have, S, -9), 1) == pids[te]).mean()
        print(f"   {name}: target {np.median(tgt):.2f} p10 {np.percentile(tgt,10):.2f}; offname p99 {np.nanpercentile(off,99):.2f} max {np.nanmax(off):.2f}; "
              f"rnd p99 {np.percentile(Sr,99):.2f} p99.99 {np.percentile(Sr,99.99):.2f} max {Sr.max():.2f}; argmax acc {acc:.3f}")
