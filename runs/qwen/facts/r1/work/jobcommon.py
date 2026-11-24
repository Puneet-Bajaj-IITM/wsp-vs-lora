"""Shared helpers for the Kaggle job scripts of facts/r1."""
import gc
import os
import random
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab
from wsp.scoring import load_model

PASS_N = 512
TOUCHED_PREFIXES = ("model.embed_tokens", "model.layers.0.", "model.layers.1.")
DIAG_ENV = "WSP_DIAG_DIR"


def make_lab(edit: str | None, out_dir: Path) -> Lab:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ[DIAG_ENV] = str(out_dir)
    return Lab(edit=edit)


def release(lab) -> None:
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    order = list(range(n_total))
    random.Random(seed).shuffle(order)
    return sorted(order[:n])


def run_split(lab: Lab, split: str, indices: list[int], tag: str, out_dir: Path, capture=()) -> float:
    logps, hidden, correct = [], {layer: [] for layer in capture}, 0
    for start in range(0, len(indices), PASS_N):
        r = lab.forward(split, indices[start:start + PASS_N], note=tag, capture_layers=tuple(capture))
        logps.append(r["logp"])
        correct += int((r["logp"].argmax(1) == r["labels"]).sum())
        for layer in capture:
            hidden[layer].append(r["hidden"][layer].astype(np.float32))
        print(tag, split, r["summary"], "left", r["passes_left"], flush=True)
    np.save(out_dir / f"logp_{tag}.npy", np.concatenate(logps))
    np.save(out_dir / f"idx_{tag}.npy", np.array(indices))
    if capture:
        np.savez(out_dir / f"hidden_{tag}.npz", **{f"L{layer}": np.concatenate(v) for layer, v in hidden.items()})
    return correct / len(indices)


def count_changed(edited_model) -> dict:
    """Counts parameter entries that differ from the base model (touched tensors only)."""
    edited = {n: p.detach().cpu() for n, p in edited_model.named_parameters() if n.startswith(TOUCHED_PREFIXES)}
    total = sum(p.numel() for p in edited_model.parameters())
    _, base = load_model()
    changed, per = 0, {}
    for name, p in base.named_parameters():
        if name in edited:
            n = int((p.detach().cpu() != edited[name]).sum())
            if n:
                per[name] = n
                changed += n
    release(base)
    return {"changed": changed, "total": total, "fraction": changed / total, "per_tensor": per}
