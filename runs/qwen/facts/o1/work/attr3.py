"""Is the asked attribute linearly readable at the last name token (train captures from job 2)?
Template-transfer check: class means from some templates, tested on held-out templates."""
import numpy as np, common as C
from wsp.metrics import load_split
J = C.CACHE / "j2"
tr = load_split("train"); idx = np.load(J / "j2_cap_idx.npy")
ex = [tr[i] for i in idx]
att = np.array([C.ATTRS.index(e["column"]) for e in ex]); tpl = np.array([e["template"] for e in ex]); pid = np.array([e["pid"] for e in ex])
CUED = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)}
cued = np.array([(a, t) in CUED for a, t in zip(att, tpl)])
for l in (6, 8):
    x = np.load(J / f"j2_name_x_L{l}.npy").astype(np.float64)
    pm = np.zeros((300, x.shape[1])); np.add.at(pm, pid, x); pm /= np.bincount(pid, minlength=300)[:, None]
    r = x - pm[pid]
    for held in range(3):
        trm = cued & (tpl != held); te = cued & (tpl == held)
        if te.sum() == 0: continue
        M = np.stack([r[trm & (att == a)].mean(0) for a in range(3)])
        M -= M.mean(0)
        pred = (r[te] @ M.T).argmax(1)
        print(f"L{l} held template {held}: n={te.sum()} acc={np.mean(pred == att[te]):.3f} attrs present {sorted(set(att[te]))}")
    M = np.stack([r[cued & (att == a)].mean(0) for a in range(3)]); M -= M.mean(0)
    s = r @ M.T
    for a in range(3):
        for t in range(3):
            m = (att == a) & (tpl == t)
            print(f"   L{l} {C.ATTRS[a]} t{t} cued={(a,t) in CUED} mean scores {np.round(s[m].mean(0),1)}")
