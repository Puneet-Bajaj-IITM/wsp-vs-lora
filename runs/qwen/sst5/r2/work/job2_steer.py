"""Job 2: steering sweep toward 'neutral' at pre-binding layers, greedy refinement, then full-dev check.

Phase A: base + zero-injection sanity + grid over (layer, scale) on train 1024-1535.
Phase B: at the best setting, try two de-intensifying directions (neg-vneg, pos-vpos).
Phase C: pick a temperature on the train log-probs, write the edit constants, score full dev from the edit file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directions import NEG, POS, VNEG, VPOS, neutral_orth, neutral_raw  # noqa: E402
from steer import inject_constant, restore  # noqa: E402
from wsp import config  # noqa: E402
from wsp.harness import Lab  # noqa: E402

OUT_DIR = HERE / "job2_out"
EDIT_NAME = "neutral_steer"
EVAL_IDX = list(range(1024, 1536))
CENTROID_FIRST_LAYER = 14
ORTH_LAYERS = (17, 19, 20)
ORTH_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
RAW_GRID = ((19, 1.0), (19, 2.0))
SANITY_LAYER = 19
DIRECTIONS = {"orth": neutral_orth, "raw": neutral_raw}
DEINTENSIFY = (("neg_minus_vneg", NEG, VNEG, (-0.5, 0.5, 1.0)), ("pos_minus_vpos", POS, VPOS, (-0.5, 0.5)))
TEMPERATURES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEV_PASS = 512
HIDDEN = 1536


class Sweep:
    def __init__(self, lab: Lab):
        self.lab, self.results, self.logps = lab, {}, {}

    def run(self, name: str, layer: int | None, vec: np.ndarray | None) -> float:
        backup = inject_constant(self.lab.model, layer, torch.from_numpy(vec.astype(np.float32))) if layer is not None else None
        r = self.lab.forward("train", EVAL_IDX, note=f"job2 {name} train 1024-1535")
        if backup:
            restore(self.lab.model, backup)
        self.results[name], self.logps[name] = r["summary"], r["logp"]
        np.save(OUT_DIR / f"logp_{name}.npy", r["logp"])
        s = r["summary"]
        print(name, "|vec|", None if vec is None else round(float(np.linalg.norm(vec)), 2), s["accuracy"],
              round(s["nll"], 3), s["by_block"], r["passes_left"], flush=True)
        return s["accuracy"]


def best_temperature(logp: np.ndarray, labels: np.ndarray) -> float:
    def nll(t: float) -> float:
        scaled = torch.log_softmax(torch.from_numpy(logp) / t, -1).numpy()
        return float(-scaled[np.arange(len(labels)), labels].mean())
    return min(TEMPERATURES, key=nll)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    centroids = np.load(HERE / "centroids_true.npy").astype(np.float32)
    lab = Lab(edit=None)
    labels = np.array([lab.examples("train")[i]["label"] for i in EVAL_IDX])
    sweep = Sweep(lab)

    base_acc = sweep.run("base", None, None)
    sweep.run(f"zero_L{SANITY_LAYER}", SANITY_LAYER, np.zeros(HIDDEN))
    grid = [("orth", l, s) for l in ORTH_LAYERS for s in ORTH_SCALES] + [("raw", l, s) for l, s in RAW_GRID]
    best = (base_acc, "base", None, None)
    for kind, layer, scale in grid:
        vec = scale * DIRECTIONS[kind](centroids[layer - 1 - CENTROID_FIRST_LAYER])
        name = f"{kind}_L{layer}_s{scale}"
        acc = sweep.run(name, layer, vec)
        if acc > best[0]:
            best = (acc, name, layer, vec)

    if best[2] is not None:
        mu = centroids[best[2] - 1 - CENTROID_FIRST_LAYER]
        for label, hi, lo, scales in DEINTENSIFY:
            base_vec, layer = best[3], best[2]
            for scale in scales:
                vec = base_vec + scale * (mu[hi] - mu[lo])
                name = f"{best[1].split('+')[0]}+{label}_{scale}"
                acc = sweep.run(name, layer, vec)
                if acc > best[0]:
                    best = (acc, f"{best[1]}+{label}_{scale}", layer, vec)
    (OUT_DIR / "summary.json").write_text(json.dumps(sweep.results, indent=1))
    print("BEST", best[0], best[1], "base", base_acc, flush=True)
    if best[2] is None:
        print("no steering beat base; skipping dev")
        return

    chosen_key = max(sweep.results, key=lambda k: sweep.results[k]["accuracy"])
    temperature = best_temperature(sweep.logps[chosen_key], labels)
    print("temperature", temperature, "from", chosen_key)
    config.EDITS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EDITS_DIR / f"{EDIT_NAME}_vec.npy", best[3].astype(np.float32))
    np.save(config.EDITS_DIR / f"{EDIT_NAME}_cfg.npy", np.array([best[2], temperature], dtype=np.float32))

    del lab, sweep.lab
    torch.cuda.empty_cache()
    edited = Lab(edit=EDIT_NAME)
    n_dev = len(edited.examples("dev"))
    dev_logp = []
    for start in range(0, n_dev, DEV_PASS):
        idx = list(range(start, min(start + DEV_PASS, n_dev)))
        r = edited.forward("dev", idx, note=f"job2 {EDIT_NAME} dev {idx[0]}-{idx[-1]}")
        dev_logp.append(r["logp"])
        print("dev chunk", r["summary"], r["passes_left"], flush=True)
    dev_logp = np.concatenate(dev_logp)
    np.save(OUT_DIR / "dev_logp_neutral_steer.npy", dev_logp)
    dev_labels = np.array([e["label"] for e in edited.examples("dev")])
    print("DEV", EDIT_NAME, "acc", float((dev_logp.argmax(1) == dev_labels).mean()),
          "nll", float(-dev_logp[np.arange(n_dev), dev_labels].mean()))


if __name__ == "__main__":
    main()
