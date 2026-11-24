import sys
import numpy as np
import circuit as C, decode, sim_keys as S
from name_parts import person_parts

mtr, mdv = decode.load_meta(S.DATA, "train"), decode.load_meta(S.DATA, "dev")
htr = np.load(S.CACHE / "hid_train.npy", mmap_mode="r"); hdv = np.load(S.CACHE / "hid_dev.npy", mmap_mode="r")
table = C.value_table(mtr); parts = person_parts(); n_parts = parts.max() + 1
layer = int(sys.argv[1])
utr, udv = C.unit_rms(np.asarray(htr[layer])), C.unit_rms(np.asarray(hdv[layer]))
for nrank in (100, 200):
    full = S.person_keys(utr, mtr, nrank, 0)
    mu = full["mu"]
    cent = decode.class_means(utr - mu, mtr["pid"], 300)
    resid = utr - mu - cent[mtr["pid"]]
    basis = C.orthonormal_rows(np.vstack([mu[None], decode.nuisance_basis(resid, nrank)]))
    x = decode.project_out(utr - mu, basis)
    part_mean = np.stack([x[np.isin(mtr["pid"], np.where((parts == p).any(1))[0])].mean(0) for p in range(n_parts)])
    add = part_mean[parts].sum(1)
    keyed = decode.project_out(cent, basis)
    cos = (add * keyed).sum(1) / np.linalg.norm(add, axis=1) / np.linalg.norm(keyed, axis=1)
    print("nuis", nrank, "cos(additive, centroid) mean", cos.mean())
    variants = {"additive": add}
    scale = np.abs(keyed).max(1, keepdims=True) / 127
    variants["int8"] = np.round(keyed / scale) * scale
    scale4 = np.abs(keyed).max(1, keepdims=True) / 7
    variants["int4"] = np.round(keyed / scale4) * scale4
    for name, k in variants.items():
        khat = k / np.linalg.norm(k, axis=1, keepdims=True)
        keys = {"mu": mu, "khat": khat, "train_scores": (utr - mu) @ khat.T}
        keys["own"] = np.array([keys["train_scores"][mtr["pid"] == p, p].mean() for p in range(300)])
        for q in (0.97, 0.99, 0.995):
            theta = S.thresholds(keys, mtr, q)
            print(name, "q", q, "train", np.round(S.evidence_accuracy(keys, theta, table, utr, mtr), 3),
                  "dev", np.round(S.evidence_accuracy(keys, theta, table, udv, mdv), 3), flush=True)
