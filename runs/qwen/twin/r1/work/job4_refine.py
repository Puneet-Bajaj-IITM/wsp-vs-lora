"""Job 4: probe v2 Likert/policy remaps on the job-2 targeted indices, auto-select, and (only if a
variant wins by > MARGIN) derive alpha and re-measure the composite lexroute_v2 on the FULL dev split.

Passes: 3 probes (+ 1 train alpha pass + 4 dev passes if the selection changes).
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (PASS_SIZE, WORK_DIR, block_accuracy, items, nll, optset_accuracy,  # noqa: E402
                    pick_text_hist, save_pass, targeted_indices, train_index_blocks)
from wsp import config  # noqa: E402
from wsp.harness import Lab  # noqa: E402
from wsp.metrics import load_split, summarize  # noqa: E402

EDIT = "lexroute_v2"
OUT = WORK_DIR / "job4"
JOB2, JOB3 = WORK_DIR / "job2", WORK_DIR / "job3"
EDITS_DIR = config.ROOT / "edits"
N_JOB1_PASSES = 5
FRESH_SEED = 20261112
MARGIN = 0.01
# Job-2 paired results on the same targeted indices (the logp files were overwritten by a later download).
JOB2_BASELINES = {"lik_swap_likert_subset": 0.276, "fc_neutral": 0.2891}
ALPHA_GRID = np.round(np.arange(0.15, 1.01, 0.025), 3)
DEV_CHUNK = 400
LIKERT_FIRST = "Disagree strongly"


def log_softmax(z: np.ndarray) -> np.ndarray:
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def subset_acc(train, idx, logp, first_option: str | None = None) -> float:
    exs = [train[i] for i in idx]
    keep = np.array([first_option is None or items()[e["column"]].options[0] == first_option for e in exs])
    labels = np.array([e["label"] for e in exs])
    return float((logp.argmax(1) == labels)[keep].mean())


def probe(edit: str, tag: str, idx: list[int], block: str) -> np.ndarray:
    lab = Lab(edit)
    r = lab.forward("train", idx, note=f"job4 probe {tag}")
    save_pass(OUT, tag, r, idx)
    print(f"[{tag}] acc {block_accuracy(r['examples'], r['logp'])['ALL']} nll {nll(r['examples'], r['logp']):.3f}",
          "optsets", optset_accuracy(r["examples"], r["logp"]), "passes_left", r["passes_left"])
    print(f"[{tag}] picks {pick_text_hist(r['examples'], r['logp'].argmax(1), block, top=8)}")
    del lab
    gc.collect()
    torch.cuda.empty_cache()
    return r["logp"]


def select(train) -> dict:
    pers_idx, fc_idx = np.load(JOB2 / "pers_indices.npy").tolist(), np.load(JOB2 / "fc_indices.npy").tolist()
    lik = {"swap": JOB2_BASELINES["lik_swap_likert_subset"]}
    lik["neutral"] = subset_acc(train, pers_idx, probe("lex_lik_v2a", "lik_v2a", pers_idx, "Personality"), LIKERT_FIRST)
    lik["odd"] = subset_acc(train, pers_idx, probe("lex_lik_v2b", "lik_v2b", pers_idx, "Personality"), LIKERT_FIRST)
    fc = {"neutral": JOB2_BASELINES["fc_neutral"]}
    fc["odd"] = subset_acc(train, fc_idx, probe("lex_fc_odd", "fc_odd", fc_idx, "False consensus"))
    best_lik = max(lik, key=lik.get)
    best_fc = max(fc, key=fc.get)
    flags = {"LIK_MODE": best_lik if lik[best_lik] > lik["swap"] + MARGIN else "swap",
             "FC_MODE": best_fc if fc[best_fc] > fc["neutral"] + MARGIN else "neutral"}
    print("SELECTION likert", {k: round(v, 4) for k, v in lik.items()}, "fc", {k: round(v, 4) for k, v in fc.items()}, flags)
    (OUT / "selection.json").write_text(json.dumps({"likert": lik, "fc": fc, "flags": flags}, indent=1))
    return flags


def derive_alpha(train) -> None:
    used = set(sum(train_index_blocks(len(train), N_JOB1_PASSES), []))
    for p in (JOB2 / "fc_indices.npy", JOB2 / "cube_indices.npy", JOB2 / "pers_indices.npy", JOB3 / "trainA_idx.npy"):
        if p.exists():
            used |= set(np.load(p).tolist())
    idx = targeted_indices(train, lambda e: True, size=PASS_SIZE, exclude=used, seed=FRESH_SEED)
    lab = Lab(EDIT)
    r = lab.forward("train", idx, note=f"job4 {EDIT} train (alpha=1)")
    save_pass(OUT, "trainA", r, idx)
    labels, n = r["labels"], np.arange(len(r["labels"]))
    nlls = {float(a): float(-log_softmax(a * r["logp"])[n, labels].mean()) for a in ALPHA_GRID}
    alpha = min(nlls, key=nlls.get)
    np.save(EDITS_DIR / f"{EDIT}_alpha.npy", np.array(alpha, dtype=np.float32))
    print("[trainA]", block_accuracy(r["examples"], r["logp"]), "ALPHA", alpha, "nll", round(nlls[alpha], 4))
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def full_dev() -> None:
    dev = load_split("dev")
    lab = Lab(EDIT)
    logps, order = [], []
    for start in range(0, len(dev), DEV_CHUNK):
        idx = list(range(start, min(start + DEV_CHUNK, len(dev))))
        r = lab.forward("dev", idx, note=f"job4 {EDIT} FINAL dev {start}-{idx[-1]}")
        save_pass(OUT, f"dev{start}", r, idx)
        logps.append(r["logp"])
        order += idx
        print(f"[dev {start}] acc {r['summary']['accuracy']:.4f} nll {r['summary']['nll']:.3f} passes_left {r['passes_left']}")
    logp = np.concatenate(logps)
    np.save(OUT / "dev_full_logp.npy", logp)
    summary = summarize(logp, [dev[i] for i in order])
    (OUT / "dev_full_summary.json").write_text(json.dumps(summary, indent=1))
    print("FINAL DEV", json.dumps(summary))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in (EDITS_DIR / f"{EDIT}_alpha.npy", EDITS_DIR / f"{EDIT}_flags.json"):
        if stale.exists():
            stale.unlink()
    train = load_split("train")
    flags = select(train)
    if flags == {"LIK_MODE": "swap", "FC_MODE": "neutral"}:
        print("NO CHANGE: lexroute_v1 stands; skipping dev re-measure")
        return
    (EDITS_DIR / f"{EDIT}_flags.json").write_text(json.dumps(flags))
    derive_alpha(train)
    full_dev()


if __name__ == "__main__":
    main()
