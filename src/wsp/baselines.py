"""Reference points that use no language model: uniform chance and the train-set answer prior per question."""
import numpy as np

from wsp import config

# Add-one smoothing, so an option never chosen in train keeps a finite log-probability.
PRIOR_SMOOTHING = 1.0


def uniform_logp(examples: list[dict]) -> np.ndarray:
    return np.full((len(examples), config.NUM_OPTIONS), -np.log(config.NUM_OPTIONS))


def fit_prior(train: list[dict]) -> dict[str, np.ndarray]:
    """Answer counts per question (`column`), indexed by the question's full option list."""
    counts = {}
    for ex in train:
        item_counts = counts.setdefault(ex["column"], np.zeros(ex["n_item_options"]))
        item_counts[ex["shown_option_ids"][ex["label"]]] += 1
    return counts


def prior_logp(prior: dict[str, np.ndarray], examples: list[dict]) -> np.ndarray:
    """Log-probs over each example's five shown options, from the smoothed train counts."""
    out = np.zeros((len(examples), config.NUM_OPTIONS))
    for i, ex in enumerate(examples):
        counts = prior.get(ex["column"], np.zeros(ex["n_item_options"]))
        shown = counts[ex["shown_option_ids"]] + PRIOR_SMOOTHING
        out[i] = np.log(shown / shown.sum())
    return out
