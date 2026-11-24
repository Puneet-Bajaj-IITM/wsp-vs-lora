"""Where does the base model's letter decision appear? NC decoding of the base-predicted letter by layer; logit-lens on letters."""
import json
import numpy as np
from common import *

tr = fields(load("train")); H = np.load(CACHE / "h_train.npy", mmap_mode="r")
logp = np.load(CACHE / "logp_base_train.npy"); pred = logp.argmax(1)
G = np.load(CACHE / "norm_gains.npz"); meta = json.loads((WORK / "meta.json").read_text())
ids = np.load(CACHE / "emb_ids.npy"); E = np.load(CACHE / "emb_rows.npy")
EL = E[[list(ids).index(i) for i in meta["letters"]]]
f = tr["tpl"] < 2
for L in range(10, 28):
    X = np.asarray(H[:, L], np.float32)
    Xc = X - X.mean(0)
    C = centroids(Xc[f], pred[f], 5)
    acc = ((unit(Xc[~f]) @ unit(C).T).argmax(1) == pred[~f]).mean()
    lens = (X / np.sqrt((X ** 2).mean(1, keepdims=True)) * G["final"]) @ EL.T
    lens_c = lens - lens.mean(0)
    print(L, f"NC acc of base pred letter {acc:.3f} | logit-lens argmax==pred {(lens.argmax(1) == pred).mean():.3f} (centred {(lens_c.argmax(1) == pred).mean():.3f}) | lens letter-logit spread {lens.std(1).mean():.2f}")
