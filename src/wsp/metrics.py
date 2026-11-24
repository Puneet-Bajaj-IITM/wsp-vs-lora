"""Accuracy, a bootstrap confidence interval, NLL and prediction summaries for letter log-probs."""
from collections import defaultdict

import numpy as np

from wsp import config

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
# Per-block accuracies are reported only for blocks with enough examples to mean something.
MIN_BLOCK_SIZE = 50


def bootstrap_ci95(correct: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(BOOTSTRAP_SAMPLES)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def summarize(logp: np.ndarray, examples: list[dict]) -> dict:
    labels = np.array([e["label"] for e in examples])
    predicted = logp.argmax(axis=1)
    correct = predicted == labels
    by_block = defaultdict(list)
    for ex, ok in zip(examples, correct):
        by_block[ex["block"]].append(ok)
    largest_first = sorted(by_block.items(), key=lambda kv: -len(kv[1]))
    return {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": bootstrap_ci95(correct),
        "nll": float(-logp[np.arange(len(labels)), labels].mean()),
        "pred_letter_hist": [round(float(p), 3) for p in np.bincount(predicted, minlength=config.NUM_OPTIONS) / len(labels)],
        "by_block": {block: round(float(np.mean(v)), 3) for block, v in largest_first if len(v) >= MIN_BLOCK_SIZE},
    }
