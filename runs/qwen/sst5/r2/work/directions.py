"""Steering directions built from class centroids (differences of means only)."""
import numpy as np

VNEG, NEG, NEUTRAL, POS, VPOS = range(5)


def neutral_raw(mu: np.ndarray) -> np.ndarray:
    """mu: (5, d) class centroids. Neutral centroid minus the mean of the other four."""
    return mu[NEUTRAL] - np.delete(mu, NEUTRAL, 0).mean(0)


def valence(mu: np.ndarray) -> np.ndarray:
    return (mu[POS] + mu[VPOS] - mu[NEG] - mu[VNEG]) / 2


def neutral_orth(mu: np.ndarray) -> np.ndarray:
    """Neutral direction with its valence component removed."""
    d, v = neutral_raw(mu), valence(mu)
    return d - v * (d @ v) / (v @ v)


PRIOR_BIAS = np.array([-3.41, -0.96, 3.59, 1.47, -0.69])
"""Prior-matched class-logit bias measured offline on base train 0-1023 (temp 4): the amount by which
each class is under-predicted relative to the label prior."""


def prior_combo(mu: np.ndarray) -> np.ndarray:
    """Push toward each class centroid in proportion to how under-predicted the class is (neutral = 1)."""
    weights = PRIOR_BIAS / PRIOR_BIAS[NEUTRAL]
    return weights @ (mu - mu.mean(0))
