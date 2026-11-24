"""Step 4: one-sided valence boost (positive-valence reviews only) in block 16; larger-sample captures."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NUM_CLASSES, TRAIN_PROBE, report, save  # noqa: E402
from weightlib import Snapshot, relu_neuron  # noqa: E402

from wsp.harness import Lab  # noqa: E402

WORK = Path(__file__).resolve().parent
BLOCK = 16
SHARP = 3.0
CONFIGS = ((3.0, -4.0), (6.0, -4.0), (6.0, -10.0), (12.0, -4.0))
CAPTURE = (16,)
EXTRA_CHUNKS = (list(range(512, 1024)), list(range(1024, 1536)))
EXTRA_CAPTURE = (15, 16)

read_dir = np.load(WORK / "consts" / "valence_dir_L15.npy")
read_mean = np.load(WORK / "consts" / "mean_L15.npy")
write_dir = np.load(WORK / "consts" / "valence_dir_L16.npy")

lab = Lab(edit="swap_v1")
snap = Snapshot(lab.model, BLOCK)


def valence_stats(res: dict, layer: int) -> list:
    proj = res["hidden"][layer] @ write_dir
    orig = np.array([NUM_CLASSES - 1 - e["shown_option_ids"][e["label"]] for e in res["examples"]])
    return [round(float(proj.mean()), 1), round(float(proj.std()), 1)] + [round(float(proj[orig == k].mean()), 1) for k in range(NUM_CLASSES)]


with torch.no_grad():
    for gain, threshold in CONFIGS:
        used = relu_neuron(lab.layers[BLOCK], read_dir, write_dir, read_mean, gain, threshold, SHARP)
        name = f"relu16_g{gain:g}_t{threshold:g}_train"
        res = lab.forward("train", TRAIN_PROBE, note=f"one-sided valence boost gain {gain} thr {threshold}", capture_layers=CAPTURE)
        snap.restore()
        print(name, "neuron", used, "L16 valence mean/std/class means", valence_stats(res, 16))
        save(name, res, hidden=False)
        report(name, res)
    for k, chunk in enumerate(EXTRA_CHUNKS):
        res = lab.forward("train", chunk, note=f"swap_v1 capture L15/L16 train chunk {k + 1}", capture_layers=EXTRA_CAPTURE)
        save(f"swapx{k + 1}_train", res)
        report(f"swapx{k + 1}_train", res)
