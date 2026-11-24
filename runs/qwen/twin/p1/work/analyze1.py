"""Local numpy analysis of job-1 probe arrays: find letter-mover heads and test key directions."""
import json
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
C = W.parent / "twin_p1_cache"
ld = lambda n: np.load(C / f"{n}.npy")
np.set_printoptions(precision=3, suppress=True, linewidth=200)

logp, y = ld("j1_train_logp"), ld("j1_train_labels")
att, dla = ld("j1_train_att"), ld("j1_train_dla").astype(np.float32)   # (n,L,H,5,4), (n,L,H,20,5)
n, L, H = att.shape[:3]
pred = logp.argmax(1)
print("train512 base acc", (pred == y).mean(), "nll", -logp[np.arange(n), y].mean(), "pred hist", np.bincount(pred, minlength=5) / n)

tot = att.sum(-1)                                   # (n,L,H,5) attention per option line
opt_mass = tot.sum(-1).mean(0)                      # (L,H)
onpred = tot[np.arange(n), :, :, pred].mean(0) / tot.sum(-1).mean(0)
onlab = tot[np.arange(n), :, :, y].mean(0) / tot.sum(-1).mean(0)
d = dla.reshape(n, L, H, 5, 4, 5)                   # option, subtype, letter
diag = np.einsum("nahosz,oz->nahos", d, np.eye(5)); own = diag - (d.sum(-1) - diag) / 4
eff = (att * own).sum((-1, -2)).mean(0)             # expected own-letter direct logit gain
order = np.dstack(np.unravel_index(np.argsort(-opt_mass, axis=None), opt_mass.shape))[0]
print("\nheads by attention mass on option lines: L H mass frac_on_pred frac_on_label dla_own subtype_split")
for l, h in order[:40]:
    sub = att[:, l, h].sum(1).mean(0)
    print(l, h, round(opt_mass[l, h], 3), round(onpred[l, h], 3), round(onlab[l, h], 3), round(eff[l, h], 3), sub / sub.sum())
order2 = np.dstack(np.unravel_index(np.argsort(-np.abs(eff), axis=None), eff.shape))[0]
print("\nheads by |direct own-letter effect|")
for l, h in order2[:25]:
    sub = att[:, l, h].sum(1).mean(0)
    print(l, h, round(opt_mass[l, h], 3), round(onpred[l, h], 3), round(onlab[l, h], 3), round(eff[l, h], 3), sub / sub.sum())
