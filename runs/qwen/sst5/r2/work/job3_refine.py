"""Job 3: multi-layer / prior-weighted steering candidates, confirmation on fresh train data, dev for the winner.

Stage 1: candidates on train 1024-1535 (same subset as job 2).
Stage 2: base, the job-2 edit and the two best candidates on fresh train 1536-2047.
Stage 3: if a candidate beats the job-2 edit on both subsets combined, write it as an edit and score full dev.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from directions import neutral_orth, prior_combo  # noqa: E402
from steer import inject_constant, restore  # noqa: E402
from wsp import config  # noqa: E402
from wsp.harness import Lab  # noqa: E402

OUT_DIR = HERE / "job3_out"
EDIT_NAME = "class_steer"
SUBSET_B = list(range(1024, 1536))
SUBSET_C = list(range(1536, 2048))
CENTROID_FIRST_LAYER = 14
INCUMBENT = "orth_L17_s2.0"
INCUMBENT_ACC_B = 0.455078125
N_FINALISTS = 2
TEMPERATURES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEV_PASS = 512
DIRECTIONS = {"orth": neutral_orth, "combo": prior_combo}
# name -> list of (inject layer, centroid layer, direction kind, scale)
CANDIDATES = {
    INCUMBENT: [(17, 16, "orth", 2.0)],
    "orth_L17_s2+L19_s1": [(17, 16, "orth", 2.0), (19, 18, "orth", 1.0)],
    "orth_L17_s1+L19_s1+L20_s1": [(17, 16, "orth", 1.0), (19, 18, "orth", 1.0), (20, 19, "orth", 1.0)],
    "orth_L16_s2": [(16, 16, "orth", 2.0)],
    "orth_L18_s1.5": [(18, 17, "orth", 1.5)],
    "combo_L17_k1.5": [(17, 16, "combo", 1.5)],
    "combo_L17_k0.75": [(17, 16, "combo", 0.75)],
    "combo_L19_k0.5": [(19, 18, "combo", 0.5)],
}


class Bench:
    def __init__(self, lab: Lab, centroids: np.ndarray):
        self.lab, self.centroids, self.acc, self.logp = lab, centroids, {}, {}

    def vectors(self, name: str) -> list[tuple[int, np.ndarray]]:
        return [(layer, scale * DIRECTIONS[kind](self.centroids[src - CENTROID_FIRST_LAYER]))
                for layer, src, kind, scale in CANDIDATES.get(name, [])]

    def run(self, name: str, subset: list[int], tag: str) -> float:
        backups = [inject_constant(self.lab.model, layer, torch.from_numpy(vec.astype(np.float32)))
                   for layer, vec in self.vectors(name)]
        r = self.lab.forward("train", subset, note=f"job3 {name} train {tag}")
        for backup in reversed(backups):
            restore(self.lab.model, backup)
        s = r["summary"]
        self.acc[(name, tag)], self.logp[(name, tag)] = s["accuracy"], r["logp"]
        np.save(OUT_DIR / f"logp_{tag}_{name}.npy", r["logp"])
        print(tag, name, s["accuracy"], round(s["nll"], 3), s["by_block"], r["passes_left"], flush=True)
        return s["accuracy"]


def best_temperature(logp: np.ndarray, labels: np.ndarray) -> float:
    def nll(t: float) -> float:
        scaled = torch.log_softmax(torch.from_numpy(logp) / t, -1).numpy()
        return float(-scaled[np.arange(len(labels)), labels].mean())
    return min(TEMPERATURES, key=nll)


def score_dev(edit_name: str) -> None:
    lab = Lab(edit=edit_name)
    examples = lab.examples("dev")
    chunks = []
    for start in range(0, len(examples), DEV_PASS):
        idx = list(range(start, min(start + DEV_PASS, len(examples))))
        r = lab.forward("dev", idx, note=f"job3 {edit_name} dev {idx[0]}-{idx[-1]}")
        chunks.append(r["logp"])
        print("dev chunk", r["summary"], r["passes_left"], flush=True)
    logp = np.concatenate(chunks)
    labels = np.array([e["label"] for e in examples])
    np.save(OUT_DIR / f"dev_logp_{edit_name}.npy", logp)
    print("DEV", edit_name, "acc", float((logp.argmax(1) == labels).mean()),
          "nll", float(-logp[np.arange(len(labels)), labels].mean()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    centroids = np.load(HERE / "centroids_true.npy").astype(np.float32)
    lab = Lab(edit=None)
    bench = Bench(lab, centroids)
    bench.acc[(INCUMBENT, "B")] = INCUMBENT_ACC_B
    challengers = [n for n in CANDIDATES if n != INCUMBENT]
    for name in challengers:
        bench.run(name, SUBSET_B, "B")
    finalists = sorted(challengers, key=lambda n: -bench.acc[(n, "B")])[:N_FINALISTS]
    for name in ["base", INCUMBENT, *finalists]:
        bench.run(name, SUBSET_C, "C")
    combined = {n: (bench.acc[(n, "B")] + bench.acc[(n, "C")]) / 2 for n in [INCUMBENT, *finalists]}
    print("COMBINED", combined, "base C", bench.acc[("base", "C")], flush=True)
    (OUT_DIR / "summary.json").write_text(json.dumps({f"{k[1]}:{k[0]}": v for k, v in bench.acc.items()}, indent=1))
    winner = max(combined, key=combined.get)
    if winner == INCUMBENT:
        print("incumbent stays best; no new dev passes")
        return

    labels = np.array([lab.examples("train")[i]["label"] for i in SUBSET_B + SUBSET_C])
    temperature = best_temperature(np.concatenate([bench.logp[(winner, "B")], bench.logp[(winner, "C")]]), labels)
    vectors = bench.vectors(winner)
    print("WINNER", winner, "temperature", temperature)
    config.EDITS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EDITS_DIR / f"{EDIT_NAME}_vecs.npy", np.stack([v for _, v in vectors]).astype(np.float32))
    np.save(config.EDITS_DIR / f"{EDIT_NAME}_cfg.npy", np.array([temperature, *[l for l, _ in vectors]], dtype=np.float32))
    del lab, bench.lab
    torch.cuda.empty_cache()
    score_dev(EDIT_NAME)


if __name__ == "__main__":
    main()
