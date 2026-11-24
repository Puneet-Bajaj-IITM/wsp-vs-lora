"""Shared helpers for sst5inv r3 jobs."""
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
PASS_SIZE = 512
NUM_CLASSES = 5
SAMPLE_SEED = 3


def chunks(n: int, size: int = PASS_SIZE) -> list[list[int]]:
    return [list(range(s, min(s + size, n))) for s in range(0, n, size)]


def sample(n_total: int, n: int = PASS_SIZE, seed: int = SAMPLE_SEED) -> list[int]:
    return sorted(np.random.default_rng(seed).choice(n_total, n, replace=False).tolist())


def predicted_class(result: dict) -> np.ndarray:
    shown = np.array([e["shown_option_ids"] for e in result["examples"]])
    return shown[np.arange(len(shown)), result["logp"].argmax(1)]


def true_class(result: dict) -> np.ndarray:
    shown = np.array([e["shown_option_ids"] for e in result["examples"]])
    return shown[np.arange(len(shown)), result["labels"]]


def confusion(result: dict) -> np.ndarray:
    table = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(table, (true_class(result), predicted_class(result)), 1)
    return table


def full_dev(lab, note: str) -> dict:
    n = len(lab.examples("dev"))
    parts = [lab.forward("dev", idx, note=note) for idx in chunks(n)]
    logp = np.concatenate([p["logp"] for p in parts])
    labels = np.concatenate([p["labels"] for p in parts])
    examples = [e for p in parts for e in p["examples"]]
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    return {"logp": logp, "labels": labels, "examples": examples, "acc": acc, "nll": nll}
