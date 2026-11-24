"""Shared helpers for the Kaggle job scripts."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK.parents[1] / "edits" / "facts" / "r2"))
from diagnose import diagnose  # noqa: E402

DEV_SIZE = 900
PASS_SIZE = 512
TRAIN_PROBE_SEED = 7
PROBE_LAYERS = (0, 1, 2, 5, 12, 20, 27, -1)


def train_probe_indices(n_train=2700):
    return sorted(np.random.default_rng(TRAIN_PROBE_SEED).choice(n_train, PASS_SIZE, replace=False).tolist())


def release(lab):
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def probe(edit, out_dir, split="train", indices=None, layers=PROBE_LAYERS, select_layer=1):
    """One pass with the edit applied; prints stage diagnostics decoded from the captured residuals."""
    lab = Lab(edit=edit)
    import _hj_build  # importable once the edit has put its folder on sys.path
    indices = indices if indices is not None else train_probe_indices()
    r = lab.forward(split, indices, note=f"{edit}: {split} probe n={len(indices)}", capture_layers=layers)
    diag = diagnose({(27.5 if k == -1 else k): v for k, v in r["hidden"].items()}, r["examples"], _hj_build.LAST, select_layer)
    print("PROBE", edit, json.dumps(r["summary"]), flush=True)
    print("DIAG", edit, json.dumps(diag), flush=True)
    np.save(out_dir / f"logp_{edit}_{split}probe.npy", r["logp"])
    stats = {"changed": _hj_build.LAST["changed"], "total": _hj_build.LAST["total"]}
    release(lab)
    return r["summary"], diag, stats


def full_dev(edit, out_dir):
    lab = Lab(edit=edit)
    logps, exs = [], []
    for start in range(0, DEV_SIZE, PASS_SIZE):
        idx = list(range(start, min(start + PASS_SIZE, DEV_SIZE)))
        r = lab.forward("dev", idx, note=f"{edit}: full dev {start}")
        logps.append(r["logp"])
        exs += r["examples"]
    logp = np.concatenate(logps)
    from wsp.metrics import summarize
    summary = summarize(logp, exs)
    print("FULLDEV", edit, json.dumps(summary), "passes_left", r["passes_left"], flush=True)
    np.save(out_dir / f"logp_{edit}_dev.npy", logp)
    release(lab)
    return summary
