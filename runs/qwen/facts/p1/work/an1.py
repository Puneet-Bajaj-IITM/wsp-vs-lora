"""Local analysis of job-1 captures: person decodability and slot-value decodability by layer."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "facts_p1_cache"
DATA = WORK.parents[1] / "data" / "processed" / "facts"
ATTRS = ("city", "profession", "hobby")


def load(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l]


tr, dv = load("train"), load("dev")
H = np.load(CACHE / "h_train.npy", mmap_mode="r")
Hd = np.load(CACHE / "h_dev.npy", mmap_mode="r")
pid = np.array([e["pid"] for e in tr]); tpl = np.array([e["template"] for e in tr])
att = np.array([ATTRS.index(e["column"]) for e in tr])
pid_d = np.array([e["pid"] for e in dv]); att_d = np.array([ATTRS.index(e["column"]) for e in dv])
shown = np.array([e["shown_option_ids"] for e in tr]); shown_d = np.array([e["shown_option_ids"] for e in dv])


def centroids(X, keys, n):
    C = np.zeros((n, X.shape[1]), np.float32)
    np.add.at(C, keys, X)
    return C / np.bincount(keys, minlength=n)[:, None]


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


print("layer | person NC acc: train-centroids->dev | LOTO (t0,1->t2) | margin stats(dev) | slotvalue acc")
for L in range(28):
    X = np.asarray(H[:, L], np.float32); Xd = np.asarray(Hd[:, L], np.float32)
    # remove attribute x template means (train) / attribute mean (dev)
    Xc = X - centroids(X, att * 3 + tpl, 9)[att * 3 + tpl]
    Xdc = Xd - centroids(Xd, att_d, 3)[att_d]
    C = unit(centroids(Xc, pid, 300))
    S = unit(Xdc) @ C.T
    acc_dev = (S.argmax(1) == pid_d).mean()
    own = S[np.arange(900), pid_d]; S2 = S.copy(); S2[np.arange(900), pid_d] = -9
    m = tpl < 2
    C2 = unit(centroids(Xc[m], pid[m], 300))
    acc_loto = ((unit(Xc[~m]) @ C2.T).argmax(1) == pid[~m]).mean()
    # slot-value: per attribute, feature for (slot, value) from train; decode on dev which value sits at slot s
    accs = []
    for a in range(3):
        ma, mda = att == a, att_d == a
        Xa, Xda = Xc[ma], Xdc[mda]
        for s in range(5):
            F = unit(centroids(Xa, shown[ma][:, s], 10))
            accs.append(((unit(Xda) @ F.T).argmax(1) == shown_d[mda][:, s]).mean())
    print(f"{L:2d} | {acc_dev:.3f} | {acc_loto:.3f} | own {own.mean():.3f} best-other {S2.max(1).mean():.3f} "
          f"|x| {np.linalg.norm(X,axis=1).mean():.1f} | sv {np.mean(accs):.3f} by-slot {np.round(np.reshape(accs,(3,5)).mean(0),2)}")
