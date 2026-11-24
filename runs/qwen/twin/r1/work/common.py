"""Shared helpers for the weight-space-programming work scripts (CPU-safe, no model code)."""
import collections
import json
from pathlib import Path

import numpy as np

from wsp import config
from wsp.catalog import load_items

WORK_DIR = config.ROOT / "wsp_work"
INDEX_SEED = 20261110
PASS_SIZE = 512
LIKERT = ("Disagree strongly", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree strongly")
POLICY = ("Strongly oppose", "Somewhat oppose", "Neither oppose nor support", "Somewhat support", "Strongly support")

_ITEMS = None


def items():
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = load_items()
    return _ITEMS


def option_texts(example: dict) -> list[str]:
    item = items()[example["column"]]
    return [item.options[i] for i in example["shown_option_ids"]]


def train_index_blocks(n_total: int, n_blocks: int, size: int = PASS_SIZE) -> list[list[int]]:
    """Disjoint fixed-seed index blocks of the train split, one per forward pass."""
    rng = np.random.default_rng(INDEX_SEED)
    perm = rng.permutation(n_total)
    return [perm[i * size:(i + 1) * size].tolist() for i in range(n_blocks)]


def targeted_indices(train: list[dict], predicate, size: int = PASS_SIZE, exclude: set | None = None,
                     seed: int = INDEX_SEED) -> list[int]:
    """Fixed-seed sample of train indices satisfying `predicate`, disjoint from `exclude`."""
    pool = [i for i, e in enumerate(train) if predicate(e) and not (exclude and i in exclude)]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=min(size, len(pool)), replace=False).tolist())


def optset_accuracy(examples: list[dict], logp: np.ndarray, min_n: int = 20) -> dict:
    """Accuracy per option-set signature (first option text), for scale-level diagnostics."""
    labels = np.array([e["label"] for e in examples])
    ok = logp.argmax(1) == labels
    per = collections.defaultdict(list)
    for ex, o in zip(examples, ok):
        per[items()[ex["column"]].options[0][:28]].append(o)
    return {k: (round(float(np.mean(v)), 3), len(v)) for k, v in per.items() if len(v) >= min_n}


def pick_text_hist(examples: list[dict], picks: np.ndarray, block: str, top: int = 6) -> list:
    counter = collections.Counter()
    for ex, pick in zip(examples, picks):
        if ex["block"] == block:
            counter[option_texts(ex)[pick]] += 1
    return counter.most_common(top)


def block_accuracy(examples: list[dict], logp: np.ndarray) -> dict:
    labels = np.array([e["label"] for e in examples])
    ok = logp.argmax(1) == labels
    per = collections.defaultdict(list)
    for ex, o in zip(examples, ok):
        per[ex["block"]].append(o)
    out = {b: (round(float(np.mean(v)), 3), len(v)) for b, v in per.items() if len(v) >= 30}
    out["ALL"] = (round(float(ok.mean()), 4), len(ok))
    return out


def nll(examples: list[dict], logp: np.ndarray) -> float:
    labels = np.array([e["label"] for e in examples])
    return float(-logp[np.arange(len(labels)), labels].mean())


def save_pass(out_dir: Path, name: str, result: dict, indices: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_logp.npy", result["logp"])
    np.save(out_dir / f"{name}_labels.npy", result["labels"])
    np.save(out_dir / f"{name}_idx.npy", np.array(indices))
    for layer, arr in result["hidden"].items():
        np.save(out_dir / f"{name}_h{layer}.npy", arr.astype(np.float16))
    (out_dir / f"{name}_summary.json").write_text(json.dumps(result["summary"], indent=1))
