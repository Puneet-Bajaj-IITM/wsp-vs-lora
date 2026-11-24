"""Local: valence axis constants at a layer from the saved swap_v1 train-probe residuals (difference of class means)."""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
N = 5


def main(job: str, run: str, layer: int):
    exs = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:512]
    orig = np.array([N - 1 - e["shown_option_ids"][e["label"]] for e in exs])
    h = np.load(WORK.parent / "smollm_sst5inv_f1_cache" / job / f"{run}_h{layer}.npy").astype(np.float32)
    mu = h.mean(0)
    means = [h[orig == k].mean(0) for k in range(N)]
    d = (means[3] + means[4]) / 2 - (means[0] + means[1]) / 2
    d /= np.linalg.norm(d)
    rms = float(np.sqrt((h ** 2).mean()))
    out = WORK / "consts"
    out.mkdir(exist_ok=True)
    np.save(out / f"valence_dir_L{layer}.npy", d.astype(np.float32))
    np.save(out / f"mean_L{layer}.npy", mu.astype(np.float32))
    proj = h @ d
    print("layer", layer, "rms", rms, "center", proj.mean(), "std", proj.std(),
          "class means", [round(float(proj[orig == k].mean() - proj.mean()), 1) for k in range(N)])


main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
