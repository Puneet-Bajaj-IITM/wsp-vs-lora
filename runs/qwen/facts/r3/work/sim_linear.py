"""Simulate the purely linear value read-out: query row for (attr, value) = sum of that value's person keys."""
import sys
import numpy as np
import circuit as C, decode, sim_keys as S

mtr, mdv = decode.load_meta(S.DATA, "train"), decode.load_meta(S.DATA, "dev")
htr = np.load(S.CACHE / "hid_train.npy", mmap_mode="r"); hdv = np.load(S.CACHE / "hid_dev.npy", mmap_mode="r")
table = C.value_table(mtr)


def value_rows(keys, normalise):
    k = keys["khat"] / keys["own"][:, None] if normalise else keys["khat"] * keys["own"][:, None]
    rows = np.zeros((C.NUM_ATTRS, C.NUM_VALUES, k.shape[1]), np.float32)
    for a in range(C.NUM_ATTRS):
        np.add.at(rows[a], table[:, a], k)
    return rows


def acc(rows, mu, u, meta):
    ev = np.einsum("nd,nvd->nv", u - mu, rows[meta["attr"]])
    shown = np.take_along_axis(ev, meta["shown"], axis=1)
    srt = np.sort(shown, axis=1)
    return float((shown.argmax(1) == meta["label"]).mean()), float((srt[:, -1] - srt[:, -2]).mean()), float(shown.std())


for layer in [int(a) for a in sys.argv[1:]]:
    utr, udv = C.unit_rms(np.asarray(htr[layer])), C.unit_rms(np.asarray(hdv[layer]))
    for nrank in (0, 100, 200, 400):
        keys = S.person_keys(utr, mtr, nrank, 0)
        for normalise in (True, False):
            rows = value_rows(keys, normalise)
            print(f"L{layer} nuis={nrank} norm={normalise} train={np.round(acc(rows, keys['mu'], utr, mtr), 3)} "
                  f"dev={np.round(acc(rows, keys['mu'], udv, mdv), 3)}", flush=True)
