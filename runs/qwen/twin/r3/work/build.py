"""Closed-form construction of query-bias vectors from mean option keys and the popularity table."""
import shutil
from pathlib import Path

import numpy as np

from common import TYPE_END, TYPE_LAST, N_OPT, N_TYPES

MAX_OFFSET_TOKENS = 200
MAX_ROTATION_RAD = 0.5
MIN_PAIR_COUNT = 3
NORM_FLOOR_FRACTION = 0.25
TEMPLATE = Path(__file__).with_name("qbias_template.py")


def low_freq_mask(inv_freq):
    keep = inv_freq * MAX_OFFSET_TOKENS < MAX_ROTATION_RAD
    return np.concatenate((keep, keep))


def option_mass(att):
    """att (n, L, H, 26) -> (n, L, H, options, types)."""
    return att[..., 1:].reshape(*att.shape[:3], N_OPT, N_TYPES).astype(np.float32)


def head_stats(att, pred, labels):
    m = option_mass(att)
    content = m[..., 2:].sum(-1)
    return {
        "mass_types": m.sum(3).mean(0),
        "agree_pred": (content.argmax(-1) == pred[:, None, None]).mean(0),
        "agree_label": (content.argmax(-1) == labels[:, None, None]).mean(0),
        "content_mass": content.sum(-1).mean(0),
    }


def head_delta(key_mean, key_cnt, table, w, layer, head, key_type, group, hd, lf_mask):
    """delta = sqrt(hd) * sum_pairs w * d / mean||d||^2 over the option set, low-frequency dims only."""
    keys = key_mean[layer, :, key_type, group * hd:(group + 1) * hd] * lf_mask
    ok = key_cnt >= MIN_PAIR_COUNT
    d = np.zeros_like(keys)
    norm2 = np.zeros(len(table.sets))
    for s in range(len(table.sets)):
        m = (table.set_of_pid == s) & ok
        if m.sum() < 2:
            continue
        d[m] = keys[m] - keys[m].mean(0)
        norm2[s] = (d[m] ** 2).sum(1).mean()
    floor = NORM_FLOOR_FRACTION * np.median(norm2[norm2 > 0])
    scale = np.where(norm2 > 0, 1.0 / np.maximum(norm2, floor), 0.0)[table.set_of_pid]
    delta = np.sqrt(hd) * ((w * scale)[:, None] * d).sum(0)
    predicted = d @ delta / np.sqrt(hd)
    used = ok & (scale > 0)
    fidelity = np.corrcoef(predicted[used], w[used])[0, 1]
    return delta, predicted, fidelity


def write_edit(edits_dir, name, rows):
    edits_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, edits_dir / f"{name}.py")
    np.save(edits_dir / f"{name}_delta.npy", np.asarray(rows, dtype=np.float32))
