"""Job 3: derive temperature + letter offsets for the composite edit, then measure it on the FULL dev split.

Passes (5): 1 x 512 fresh train with lexroute_v1 (no constants yet, capture -1) -> write constants;
            4 x 400 dev with the finished lexroute_v1.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (PASS_SIZE, WORK_DIR, block_accuracy, nll, optset_accuracy, pick_text_hist,  # noqa: E402
                    save_pass, targeted_indices, train_index_blocks)
from wsp import config  # noqa: E402
from wsp.harness import Lab  # noqa: E402
from wsp.metrics import load_split, summarize  # noqa: E402

EDIT = "lexroute_v1"
OUT = WORK_DIR / "job3"
EDITS_DIR = config.ROOT / "edits"
N_JOB1_PASSES = 5
FRESH_SEED = 20261111
ALPHA_GRID = np.round(np.arange(0.15, 1.01, 0.025), 3)
DEV_CHUNK = 400
OFFSET_ACC_TOLERANCE = 0.002


def log_softmax(z: np.ndarray) -> np.ndarray:
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def derive_constants(r: dict) -> dict:
    logp, labels, H = r["logp"], r["labels"], r["hidden"][-1]
    n = np.arange(len(labels))
    b = -(logp.mean(0) - logp.mean())
    mu = H.mean(0)
    proj = H @ mu / (mu @ mu)
    acc_plain = float((logp.argmax(1) == labels).mean())
    acc_off = float(((logp + b).argmax(1) == labels).mean())
    use_offsets = acc_off >= acc_plain - OFFSET_ACC_TOLERANCE
    z = logp + b if use_offsets else logp
    nlls = {float(a): float(-log_softmax(a * z)[n, labels].mean()) for a in ALPHA_GRID}
    alpha = min(nlls, key=nlls.get)
    delta = np.outer(b, mu / (mu @ mu)).astype(np.float32) if use_offsets else None
    info = {"alpha": alpha, "nll_at_alpha": nlls[alpha], "nll_alpha1": nlls[1.0], "offsets": b.round(4).tolist(),
            "use_offsets": bool(use_offsets), "acc_plain": acc_plain, "acc_offsets": acc_off,
            "proj_std": float(proj.std()), "delta_row_norms": None if delta is None else np.linalg.norm(delta, axis=1).round(4).tolist()}
    np.save(EDITS_DIR / f"{EDIT}_alpha.npy", np.array(alpha, dtype=np.float32))
    if delta is not None:
        np.save(EDITS_DIR / f"{EDIT}_wu_delta.npy", delta)
    (OUT / "constants.json").write_text(json.dumps(info, indent=1))
    print("CONSTANTS", json.dumps(info))
    return info


def phase_a() -> None:
    train = load_split("train")
    used = set(sum(train_index_blocks(len(train), N_JOB1_PASSES), []))
    for name in ("fc", "cube", "pers"):
        p = WORK_DIR / "job2" / f"{name}_indices.npy"
        if p.exists():
            used |= set(np.load(p).tolist())
    idx = targeted_indices(train, lambda e: True, size=PASS_SIZE, exclude=used, seed=FRESH_SEED)
    lab = Lab(EDIT)
    r = lab.forward("train", idx, note="job3 lexroute_v1 train (alpha=1)", capture_layers=(-1,))
    save_pass(OUT, "trainA", r, idx)
    print("[trainA] acc", block_accuracy(r["examples"], r["logp"]), "nll", round(nll(r["examples"], r["logp"]), 3),
          "letters", r["summary"]["pred_letter_hist"])
    print("[trainA] optsets", optset_accuracy(r["examples"], r["logp"]))
    for blk in ("Personality", "False consensus", "Cognitive tests", "Economic preferences"):
        print(f"[trainA] {blk} picks", pick_text_hist(r["examples"], r["logp"].argmax(1), blk))
    derive_constants(r)
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def phase_b() -> None:
    dev = load_split("dev")
    lab = Lab(EDIT)
    logps, order = [], []
    for start in range(0, len(dev), DEV_CHUNK):
        idx = list(range(start, min(start + DEV_CHUNK, len(dev))))
        r = lab.forward("dev", idx, note=f"job3 {EDIT} FINAL dev {start}-{idx[-1]}")
        save_pass(OUT, f"dev{start}", r, idx)
        logps.append(r["logp"])
        order += idx
        print(f"[dev {start}] acc {r['summary']['accuracy']:.4f} nll {r['summary']['nll']:.3f} passes_left {r['passes_left']}")
    logp = np.concatenate(logps)
    np.save(OUT / "dev_full_logp.npy", logp)
    summary = summarize(logp, [dev[i] for i in order])
    (OUT / "dev_full_summary.json").write_text(json.dumps(summary, indent=1))
    print("FINAL DEV", json.dumps(summary))
    print("FINAL DEV optsets", optset_accuracy([dev[i] for i in order], logp))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in (EDITS_DIR / f"{EDIT}_alpha.npy", EDITS_DIR / f"{EDIT}_wu_delta.npy"):
        if stale.exists():
            stale.unlink()
    phase_a()
    phase_b()


if __name__ == "__main__":
    main()
