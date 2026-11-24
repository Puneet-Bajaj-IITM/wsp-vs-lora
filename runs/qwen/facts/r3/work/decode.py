"""Label-free decodability statistics of captured answer-position residuals (centroids, PCA only)."""
import json
from pathlib import Path

import numpy as np

ATTRS = ("city", "profession", "hobby")
NUM_VALUES = 10
NUM_SLOTS = 5
NUISANCE_RANKS = (0, 30, 100)


def load_meta(data_dir: Path, split: str) -> dict:
    rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines() if line]
    shown = np.array([r["shown_option_ids"] for r in rows])
    label = np.array([r["label"] for r in rows])
    return {
        "pid": np.array([r["pid"] for r in rows]),
        "attr": np.array([ATTRS.index(r["column"]) for r in rows]),
        "template": np.array([r["template"] for r in rows]),
        "shown": shown,
        "label": label,
        "value": shown[np.arange(len(rows)), label],
    }


def class_means(x: np.ndarray, cls: np.ndarray, n_cls: int) -> np.ndarray:
    out = np.zeros((n_cls, x.shape[1]), dtype=np.float32)
    np.add.at(out, cls, x)
    return out / np.maximum(np.bincount(cls, minlength=n_cls), 1)[:, None]


def nuisance_basis(resid: np.ndarray, rank: int) -> np.ndarray:
    if rank == 0:
        return np.zeros((0, resid.shape[1]), dtype=np.float32)
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    return vt[:rank]


def project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return x - (x @ basis.T) @ basis


def centroid_acc(xtr, ctr, xdv, cdv, n_cls, rank) -> float:
    mu = xtr.mean(0)
    cent = class_means(xtr - mu, ctr, n_cls)
    basis = nuisance_basis(xtr - mu - cent[ctr], rank)
    cent_p = project_out(cent, basis)
    cent_p /= np.linalg.norm(cent_p, axis=1, keepdims=True) + 1e-8
    score = project_out(xdv - mu, basis) @ cent_p.T
    return float((score.argmax(1) == cdv).mean())


def person_report(htr, hdv, mtr, mdv) -> None:
    for layer in range(htr.shape[0]):
        xtr, xdv = htr[layer].astype(np.float32), hdv[layer].astype(np.float32)
        accs = [centroid_acc(xtr, mtr["pid"], xdv, mdv["pid"], 300, r) for r in NUISANCE_RANKS]
        print(f"Q1 person layer {layer:2d} " + " ".join(f"r{r}={a:.3f}" for r, a in zip(NUISANCE_RANKS, accs)), flush=True)


def slot_report(htr, hdv, mtr, mdv) -> None:
    for layer in range(htr.shape[0]):
        xtr, xdv = htr[layer].astype(np.float32), hdv[layer].astype(np.float32)
        accs = []
        for a in range(len(ATTRS)):
            tr, dv = mtr["attr"] == a, mdv["attr"] == a
            for s in range(NUM_SLOTS):
                accs.append(centroid_acc(xtr[tr], mtr["shown"][tr, s], xdv[dv], mdv["shown"][dv, s], NUM_VALUES, 0))
        accs = np.array(accs).reshape(len(ATTRS), NUM_SLOTS)
        print(f"Q2 slot-content layer {layer:2d} mean={accs.mean():.3f} by_slot={np.round(accs.mean(0), 2)}"
              f" by_attr={np.round(accs.mean(1), 2)}", flush=True)
