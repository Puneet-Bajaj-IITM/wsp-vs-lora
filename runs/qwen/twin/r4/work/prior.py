"""Per-item answer-frequency table from train labels (simple counts)."""
import numpy as np

import common

SMOOTHING = 1.0


def pair_log_prior(train: list[dict]) -> np.ndarray:
    """log P(option | item) with add-one smoothing, indexed by pair id."""
    cols, item_of, pair0, n_opts, n_pairs = common.item_index(train)
    counts = np.zeros(n_pairs)
    for ex in train:
        counts[pair0[ex["column"]] + ex["shown_option_ids"][ex["label"]]] += 1
    logp = np.zeros(n_pairs)
    for c in cols:
        sl = slice(pair0[c], pair0[c] + n_opts[c])
        p = counts[sl] + SMOOTHING
        logp[sl] = np.log(p / p.sum())
    return logp
