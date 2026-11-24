"""Shared offline helpers: load cached job arrays and map letter log-probs to class space."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
CACHE = ROOT / "wsp_work" / "sst5_r2_cache"
DATA = ROOT / "data" / "processed" / "sst5"
N_CLASSES = 5
VNEG, NEG, NEUTRAL, POS, VPOS = range(N_CLASSES)


def load_meta(split: str, indices) -> tuple[np.ndarray, np.ndarray]:
    """Returns (shown_option_ids, true_class) for the given example indices."""
    rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").open()]
    rows = [rows[i] for i in indices]
    shown = np.array([r["shown_option_ids"] for r in rows])
    label = np.array([r["label"] for r in rows])
    return shown, shown[np.arange(len(rows)), label]


def to_class_space(logp: np.ndarray, shown: np.ndarray) -> np.ndarray:
    out = np.zeros_like(logp)
    np.put_along_axis(out, shown, logp, 1)
    return out


def acc_nll(class_logp: np.ndarray, true_cls: np.ndarray) -> tuple[float, float]:
    lp = class_logp - np.logaddexp.reduce(class_logp, 1, keepdims=True)
    return float((lp.argmax(1) == true_cls).mean()), float(-lp[np.arange(len(true_cls)), true_cls].mean())
