"""Helpers shared by the Kaggle job scripts of this run."""
import json

import numpy as np

SEED = 20261120
PASS_SIZE = 512


def train_order(n_train):
    return np.random.default_rng(SEED).permutation(n_train)


def flat(value):
    if isinstance(value, dict):
        return [x for v in value.values() for x in flat(v)]
    if isinstance(value, list):
        return [x for v in value for x in flat(v)]
    return [value]


def log_softmax(x):
    top = x.max(1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(1, keepdims=True))


def score(logits, labels):
    lp = log_softmax(np.asarray(logits, dtype=np.float64))
    return float((lp.argmax(1) == labels).mean()), float(-lp[np.arange(len(labels)), labels].mean())


def full_dev(lab, tag):
    dev = lab.examples("dev")
    logps, labels = [], []
    for start in range(0, len(dev), PASS_SIZE):
        chunk = list(range(start, min(start + PASS_SIZE, len(dev))))
        r = lab.forward("dev", chunk, note=f"{tag} dev {start}")
        logps.append(r["logp"])
        labels.append(r["labels"])
        print(tag, "dev chunk", start, json.dumps(r["summary"]), r["passes_left"], flush=True)
    logp, labels = np.concatenate(logps), np.concatenate(labels)
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    print(tag, "DEV FULL acc", acc, "nll", nll, flush=True)
    return logp, labels, acc, nll
