"""Shared helpers for the step scripts."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
OUT = WORK / "out"
NUM_CLASSES = 5
TRAIN_PROBE = list(range(512))
DEV_CHUNKS = [list(range(0, 512)), list(range(512, 1024)), list(range(1024, 1101))]


def save(name: str, result: dict, hidden: bool = True) -> None:
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"{name}_logp.npy", result["logp"].astype(np.float32))
    if hidden:
        for layer, arr in result["hidden"].items():
            np.save(OUT / f"{name}_h{layer}.npy", arr.astype(np.float16))
    (OUT / f"{name}_summary.json").write_text(json.dumps(result["summary"], indent=1))


def class_confusion(result: dict) -> np.ndarray:
    """rows = original sentiment (mirror of target id), cols = option id the model picked."""
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    pred = result["logp"].argmax(1)
    for ex, p in zip(result["examples"], pred):
        target = ex["shown_option_ids"][ex["label"]]
        conf[NUM_CLASSES - 1 - target, ex["shown_option_ids"][p]] += 1
    return conf


def report(name: str, result: dict) -> None:
    print(f"== {name}", json.dumps(result["summary"]), "left", result["passes_left"], flush=True)
    print(class_confusion(result), flush=True)


def full_dev(lab, name: str, note: str) -> dict:
    parts = [lab.forward("dev", chunk, note=f"{note} dev chunk {k}") for k, chunk in enumerate(DEV_CHUNKS)]
    logp = np.concatenate([p["logp"] for p in parts])
    labels = np.concatenate([p["labels"] for p in parts])
    examples = [e for p in parts for e in p["examples"]]
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"{name}_dev_logp.npy", logp.astype(np.float32))
    merged = {"logp": logp, "labels": labels, "examples": examples}
    print(f"== FULL DEV {name} acc {acc:.4f} nll {nll:.4f} n {len(labels)} left {parts[-1]['passes_left']}", flush=True)
    print(class_confusion(merged), flush=True)
    return merged
