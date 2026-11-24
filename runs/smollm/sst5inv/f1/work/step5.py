"""Step 5: option-word substitutions on top of swap_v1 + one-sided valence boost (gain 6, threshold -10)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAIN_PROBE, report, save  # noqa: E402
from weightlib import relu_neuron  # noqa: E402

from wsp.harness import Lab  # noqa: E402

WORK = Path(__file__).resolve().parent
BLOCK = 16
SHARP = 3.0
GAIN, THRESHOLD = 6.0, -10.0
SUBSTITUTES = {
    " very": (" clearly", " strongly", " extremely", " quite", " really", " highly"),
    " neutral": (" mixed", " average", " okay", " neither"),
}

read_dir = np.load(WORK / "consts" / "valence_dir_L15.npy")
read_mean = np.load(WORK / "consts" / "mean_L15.npy")
write_dir = np.load(WORK / "consts" / "valence_dir_L16.npy")

lab = Lab(edit="swap_v1")
embed = lab.model.get_input_embeddings().weight

with torch.no_grad():
    neuron = relu_neuron(lab.layers[BLOCK], read_dir, write_dir, read_mean, GAIN, THRESHOLD, SHARP)
    print("boost neuron", neuron)
    for target, words in SUBSTITUTES.items():
        target_id = lab.tok.encode(target, add_special_tokens=False)[0]
        original = embed[target_id].clone()
        for word in words:
            ids = lab.tok.encode(word, add_special_tokens=False)
            if len(ids) != 1:
                print("SKIP multi-token", repr(word), ids)
                continue
            embed[target_id] = embed[ids[0]]
            name = f"sub_{target.strip()}_{word.strip()}_train"
            res = lab.forward("train", TRAIN_PROBE, note=f"boost + {target!r} -> {word!r}")
            embed[target_id] = original
            save(name, res, hidden=False)
            report(name, res)
