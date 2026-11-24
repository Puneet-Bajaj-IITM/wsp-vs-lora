"""Attribute score m_a(x) = M_a.(x - mu_name) normalised to 1 on cued-asked; distribution by group,
with M estimated on templates != t and evaluated on template t (transfer)."""
import numpy as np, common as C
from wsp.metrics import load_split
J = C.CACHE / "j2"
tr = load_split("train"); idx = np.load(J / "j2_cap_idx.npy"); ex = [tr[i] for i in idx]
att = np.array([C.ATTRS.index(e["column"]) for e in ex]); tpl = np.array([e["template"] for e in ex]); pid = np.array([e["pid"] for e in ex])
CUED = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)}
cued = np.array([(a, t) in CUED for a, t in zip(att, tpl)])
x = np.load(J / "j2_name_x_L8.npy").astype(np.float64)
pm = np.zeros((300, 1536)); np.add.at(pm, pid, x); pm /= np.bincount(pid, minlength=300)[:, None]
r = x - pm[pid]; mu = x.mean(0)
def fitM(mask):
    M = np.stack([r[mask & cued & (att == a)].mean(0) for a in range(3)]); M -= M.mean(0)
    s = (x - mu) @ M.T
    norm = np.array([s[mask & cued & (att == a), a].mean() for a in range(3)])
    return M / norm[:, None]
M = fitM(np.ones(len(x), bool))
s = (x - mu) @ M.T
for name, m in (("cued", cued), ("uncued", ~cued)):
    asked = s[m, att[m]]
    oth = np.concatenate([s[m & (att == a)][:, [b for b in range(3) if b != a]].ravel() for a in range(3)])
    print(f"in-sample {name}: asked {asked.mean():.2f}+-{asked.std():.2f}  other {oth.mean():.2f}+-{oth.std():.2f}")
# leave-template-out for hobby (cued in all) and city t0/t2
for t in range(3):
    Mt = fitM(tpl != t) if all((tpl != t)[cued & (att == a)].any() for a in range(3)) else None
    if Mt is None: print("skip", t); continue
    st = (x - mu) @ Mt.T
    m = cued & (tpl == t)
    print(f"held t{t}: asked {st[m, att[m]].mean():.2f}+-{st[m, att[m]].std():.2f}; argmax acc {(st[m].argmax(1)==att[m]).mean():.3f}")
