"""Offline simulation of the soft person-evidence stage on captured residuals (label-free keys, table lookup)."""
import sys
from pathlib import Path

import numpy as np

import circuit as C
import decode

CACHE = Path(__file__).resolve().parent.parent / "facts_r3_cache"
DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "facts"


def person_keys(u, meta, nuisance_rank, key_rank):
    mu = u.mean(0)
    cent = decode.class_means(u - mu, meta["pid"], C.NUM_PEOPLE)
    resid = u - mu - cent[meta["pid"]]
    basis = C.orthonormal_rows(np.vstack([mu[None], decode.nuisance_basis(resid, nuisance_rank)]))
    keyed = decode.project_out(cent, basis)
    if key_rank:
        _, _, vt = np.linalg.svd(keyed, full_matrices=False)
        keyed = (keyed @ vt[:key_rank].T) @ vt[:key_rank]
    khat = keyed / np.linalg.norm(keyed, axis=1, keepdims=True)
    own = C.unbiased_sq_norm(decode.project_out(u - mu, basis), meta["pid"]) / np.linalg.norm(
        decode.project_out(cent, basis), axis=1)
    scores = (u - mu) @ khat.T
    return {"mu": mu, "khat": khat, "own": own, "train_scores": scores}


def thresholds(keys, meta, quantile):
    scores = keys["train_scores"].copy()
    scores[np.arange(len(scores)), meta["pid"]] = np.nan
    return np.nanquantile(scores, quantile, axis=0)


def evidence_accuracy(keys, theta, table, u, meta):
    act = np.maximum(((u - keys["mu"]) @ keys["khat"].T - theta) / np.maximum(keys["own"] - theta, 1e-3), 0.0)
    person_values = table[:, meta["attr"]].T
    ev = np.zeros((len(u), C.NUM_VALUES), np.float32)
    for v in range(C.NUM_VALUES):
        ev[:, v] = (act * (person_values == v)).sum(1)
    shown_ev = np.take_along_axis(ev, meta["shown"], axis=1)
    own_act = act[np.arange(len(u)), meta["pid"]]
    return float((shown_ev.argmax(1) == meta["label"]).mean()), float(own_act.mean()), float((act.sum(1) - own_act).mean())


if __name__ == "__main__":
    mtr, mdv = decode.load_meta(DATA, "train"), decode.load_meta(DATA, "dev")
    htr = np.load(CACHE / "hid_train.npy", mmap_mode="r")
    hdv = np.load(CACHE / "hid_dev.npy", mmap_mode="r")
    table = C.value_table(mtr)
    for layer in [int(a) for a in sys.argv[1:]]:
        utr, udv = C.unit_rms(np.asarray(htr[layer])), C.unit_rms(np.asarray(hdv[layer]))
        for nrank in (100, 200):
            for krank in (0, 96, 64):
                keys = person_keys(utr, mtr, nrank, krank)
                for q in (0.9, 0.97, 0.99):
                    theta = thresholds(keys, mtr, q)
                    print(f"L{layer} nuis={nrank} keyrank={krank} q={q} train(acc,own,others)="
                          f"{np.round(evidence_accuracy(keys, theta, table, utr, mtr), 3)} dev="
                          f"{np.round(evidence_accuracy(keys, theta, table, udv, mdv), 3)}", flush=True)
