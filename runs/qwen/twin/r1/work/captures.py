"""Load job captures (downloaded from Kaggle) and simulate letter logits offline.

logits = alpha * (H @ W_U[letters].T) + beta * (H @ D.T) + bias, where H is the final-norm output at the
answer position. This is exact for edits that only change the final norm scale and the five letter rows
of the (tied) unembedding, up to fp16 rounding.
"""
import json
from pathlib import Path

import numpy as np

from wsp import config
from wsp.metrics import load_split

KAGGLE_OUT = config.ROOT / "kaggle_runner" / "out" / "wsp" / "out" / "wsp_work"
FINAL_LAYER_KEY = -1


def job_dir(job: str) -> Path:
    local = config.ROOT / "wsp_work" / job
    return local if (local / "meta.json").exists() or any(local.glob("*_logp.npy")) else KAGGLE_OUT / job


def load_passes(job: str, prefix: str, layer=FINAL_LAYER_KEY) -> dict:
    d = job_dir(job)
    train = load_split("train")
    names = sorted(p.name[: -len("_logp.npy")] for p in d.glob(f"{prefix}*_logp.npy"))
    H, logp, labels, idx = [], [], [], []
    for n in names:
        logp.append(np.load(d / f"{n}_logp.npy"))
        labels.append(np.load(d / f"{n}_labels.npy"))
        idx.append(np.load(d / f"{n}_idx.npy"))
        h_path = d / f"{n}_h{layer}.npy"
        if h_path.exists():
            H.append(np.load(h_path).astype(np.float32))
    idx = np.concatenate(idx)
    return {
        "H": np.concatenate(H) if H else None,
        "logp": np.concatenate(logp), "labels": np.concatenate(labels), "idx": idx,
        "examples": [train[i] for i in idx], "names": names,
    }


def load_weights(job: str) -> dict:
    d = job_dir(job)
    return {
        "W": np.load(d / "W_U_letters.npy"), "g": np.load(d / "final_norm_g.npy"),
        "meta": json.loads((d / "meta.json").read_text()),
    }


def log_softmax(z: np.ndarray) -> np.ndarray:
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def simulate(H: np.ndarray, W: np.ndarray, alpha: float = 1.0, D: np.ndarray | None = None,
             beta: float = 0.0, bias: np.ndarray | None = None) -> np.ndarray:
    z = alpha * (H @ W.T)
    if D is not None and beta:
        z = z + beta * (H @ D.T)
    if bias is not None:
        z = z + bias
    return log_softmax(z)


def metrics(logp: np.ndarray, labels: np.ndarray, examples: list[dict] | None = None) -> dict:
    ok = logp.argmax(1) == labels
    out = {"acc": round(float(ok.mean()), 4), "nll": round(float(-logp[np.arange(len(labels)), labels].mean()), 4)}
    if examples is not None:
        blocks = {}
        for ex, o in zip(examples, ok):
            blocks.setdefault(ex["block"], []).append(o)
        out["blocks"] = {b: round(float(np.mean(v)), 3) for b, v in blocks.items() if len(v) >= 40}
    return out


def mass_mean_readout(H: np.ndarray, labels: np.ndarray, n_classes: int = config.NUM_OPTIONS) -> np.ndarray:
    """Difference of means: D[X] = mean(H | label=X) - mean(H). A statistic, not a fitted solver."""
    mu = H.mean(0)
    return np.stack([H[labels == c].mean(0) - mu for c in range(n_classes)])


def pca_project(D: np.ndarray, H: np.ndarray, k: int) -> np.ndarray:
    """Project readout rows onto the top-k principal directions of the (centered) activations."""
    Hc = H - H.mean(0)
    _, _, vt = np.linalg.svd(Hc, full_matrices=False)
    basis = vt[:k]
    return (D @ basis.T) @ basis
