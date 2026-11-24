"""Is there a slot-independent 'chosen value' direction in the base model? (uses base predictions, not labels)"""
import numpy as np
from common import *

tr = fields(load("train")); H = np.load(CACHE / "h_train.npy", mmap_mode="r")
logp = np.load(CACHE / "logp_base_train.npy")
pred = logp.argmax(1)
print("base train acc", (pred == tr["label"]).mean(), "pred hist", np.bincount(pred, minlength=5) / len(pred))
chosen = tr["shown"][np.arange(2700), pred]
for a in range(3):
    m = tr["att"] == a
    print(ATTRS[a], "chosen-value freq", np.round(np.bincount(chosen[m], minlength=10) / m.sum(), 2),
          "P(chosen|shown)", np.round(np.bincount(chosen[m], minlength=10) / (tr["shown"][m][:, :, None] == np.arange(10)).any(1).sum(0), 2))
print("layer | acc of predicting chosen value among shown (train tpl0,1 -> tpl2), letter-residualised | same w/o residualising")
for L in range(8, 28):
    X = np.asarray(H[:, L], np.float32)
    accs, accs2 = [], []
    for a in range(3):
        m = tr["att"] == a
        Xa = X[m]; pr = pred[m]; ch = chosen[m]; sh = tr["shown"][m]; tp = tr["tpl"][m]
        Xa = Xa - centroids(Xa, tp, 3)[tp]
        Xr = Xa - centroids(Xa, pr, 5)[pr]
        for Xu, out in ((Xr, accs), (Xa, accs2)):
            f = tp < 2
            C = centroids(Xu[f], ch[f], 10)
            S = Xu[~f] @ unit(C).T
            Ssh = np.take_along_axis(S, sh[~f], 1)
            out.append((sh[~f][np.arange((~f).sum()), Ssh.argmax(1)] == ch[~f]).mean())
    print(L, np.round(accs, 3), np.round(accs2, 3))
