"""Shared helpers for sst5/r3 jobs: reversible in-memory edits and result dumps."""
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "sst5_r3_cache"
CLASS_NAMES = ["very negative", "negative", "neutral", "positive", "very positive"]
PASS_SIZE = 512


@contextmanager
def temporary(model, fn):
    """Apply fn(model) in place, then restore every parameter it changed."""
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        fn(model)
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(before[n])


def class_of(examples):
    """True sentiment class id per example and the letter->class map."""
    shown = np.array([e["shown_option_ids"] for e in examples])
    labels = np.array([e["label"] for e in examples])
    return shown[np.arange(len(labels)), labels], shown


def describe(tag, logp, examples):
    true_cls, shown = class_of(examples)
    pred_letter = logp.argmax(1)
    pred_cls = shown[np.arange(len(pred_letter)), pred_letter]
    labels = np.array([e["label"] for e in examples])
    conf = np.zeros((5, 5), int)
    np.add.at(conf, (true_cls, pred_cls), 1)
    acc = float((pred_letter == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    print(f"[{tag}] n={len(labels)} acc={acc:.4f} nll={nll:.4f} "
          f"letter_hist={np.bincount(pred_letter, minlength=5).tolist()} "
          f"class_hist={np.bincount(pred_cls, minlength=5).tolist()}", flush=True)
    print(conf, flush=True)
    return {"tag": tag, "acc": acc, "nll": nll, "conf": conf.tolist()}


def save_json(name, obj):
    (WORK / name).write_text(json.dumps(obj, indent=1))
