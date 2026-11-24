"""Step 3: the letter is written by block 17 (pred-letter AUC 0.61 at L16 -> 0.97 at L17), so step 2's amplifier
sat after the matcher.  Move it to block 16's MLP (before block 17's attention), and test the closed-form letter bias."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NUM_CLASSES, TRAIN_PROBE, report, save  # noqa: E402
from weightlib import Snapshot, letter_bias, linear_map_neurons  # noqa: E402

from wsp import config  # noqa: E402
from wsp.harness import Lab  # noqa: E402

WORK = Path(__file__).resolve().parent
BLOCK = 16
AMP_GAINS = (1.7, 5.1, 11.9, -1.2)
CAPTURE = (16, 17)

read_dir = np.load(WORK / "consts" / "valence_dir_L15.npy")
read_mean = np.load(WORK / "consts" / "mean_L15.npy")
write_dir = np.load(WORK / "consts" / "valence_dir_L16.npy")
bias = np.load(WORK / "consts" / "letter_bias_swap.npy")
mean_final = np.load(WORK / "consts" / "mean_final_swap.npy")

lab = Lab(edit="swap_v1")
snap = Snapshot(lab.model, BLOCK)


def valence_stats(res: dict, layer: int) -> list:
    proj = res["hidden"][layer] @ write_dir
    orig = np.array([NUM_CLASSES - 1 - e["shown_option_ids"][e["label"]] for e in res["examples"]])
    return [round(float(proj.mean()), 1), round(float(proj.std()), 1)] + [round(float(proj[orig == k].mean()), 1) for k in range(NUM_CLASSES)]


with torch.no_grad():
    for gain in AMP_GAINS:
        used = linear_map_neurons(lab.layers[BLOCK], read_dir, write_dir, read_mean, gain)
        name = f"amp16_{gain:+.1f}_train"
        res = lab.forward("train", TRAIN_PROBE, note=f"valence amp gain {gain} block {BLOCK}", capture_layers=CAPTURE)
        snap.restore()
        print(name, "neurons", used, "L16 valence mean/std/class means", valence_stats(res, 16), "L17", valence_stats(res, 17))
        save(name, res, hidden=False)
        report(name, res)

    out_before = lab.model.get_output_embeddings().weight.clone()
    letter_bias(lab.model, lab.tok, config.OPTION_LETTERS, bias, mean_final)
    res = lab.forward("train", TRAIN_PROBE, note="swap_v1 + letter bias (equalise mean letter logp)")
    lab.model.get_output_embeddings().weight.copy_(out_before)
    save("lbias_train", res, hidden=False)
    report("lbias_train", res)
