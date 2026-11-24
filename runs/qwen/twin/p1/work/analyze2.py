"""Do option keys (content-last / newline tokens) carry 'respondents pick this' along a mean-difference direction?"""
import json
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent
C = W.parent / "twin_p1_cache"
np.set_printoptions(precision=3, suppress=True, linewidth=200)
y = np.load(C / "j1_train_labels.npy"); logp = np.load(C / "j1_train_logp.npy")
kt = np.load(C / "j1_train_kt.npy", mmap_mode="r")       # (n,28,20,256): option*4+subtype
att = np.load(C / "j1_train_att.npy")
n = len(y); half = np.arange(n) % 2 == 0
SUB = {2: "content_last", 3: "newline", 0: "letter"}
onehot = np.eye(5)[y].astype(bool)
print("layer group subtype : heldout acc of argmax(delta.k)  (chance .2, model .297)")
for L in range(12, 28):
    row = []
    for s in (2, 3, 0):
        for g in (0, 1):
            K = np.asarray(kt[:, L, s::4, g * 128:(g + 1) * 128], dtype=np.float32)   # (n,5,128)
            accs = []
            for tr, te in ((half, ~half), (~half, half)):
                d = K[tr][onehot[tr]].mean(0) - K[tr][~onehot[tr]].mean(0)
                accs.append(((K[te] @ d).argmax(1) == y[te]).mean())
            row.append(round(float(np.mean(accs)), 3))
    print(L, row, "  [cl g0, cl g1, nl g0, nl g1, letter g0, letter g1]")
# how good are head-level selections: argmax of attention over options vs label / pred
pred = logp.argmax(1)
tot = att[..., 2] + att[..., 3]
print("\nhead content+newline attention argmax: acc vs label, agreement with model pred")
for L in range(14, 24):
    for h in range(12):
        a = tot[:, L, h].argmax(1)
        if (a == pred).mean() > 0.3 or (a == y).mean() > 0.26:
            print(L, h, round((a == y).mean(), 3), round((a == pred).mean(), 3), "mass", round(tot[:, L, h].sum(1).mean(), 3))
