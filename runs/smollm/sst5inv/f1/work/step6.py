"""Step 6: the assembled edit valence_v1 through Lab(edit=...): train probe + full dev; then in-process alternates
of the boost's (gain, threshold) on the same train probe (bias and temperature left in place)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAIN_PROBE, full_dev, report, save  # noqa: E402
from weightlib import relu_neuron  # noqa: E402

from wsp.harness import Lab  # noqa: E402

WORK = Path(__file__).resolve().parent
EDIT = "valence_v1"
BLOCK = 16
SHARP = 3.0
ALTERNATES = ((6.0, -16.0), (9.0, -13.0), (4.0, -10.0), (9.0, -10.0))

read_dir = np.load(WORK / "consts" / "valence_dir_L15.npy")
read_mean = np.load(WORK / "consts" / "mean_L15.npy")
write_dir = np.load(WORK / "consts" / "valence_dir_L16.npy")

lab = Lab(edit=EDIT)
res = lab.forward("train", TRAIN_PROBE, note=f"{EDIT} train probe", capture_layers=(-1,))
mean_final = res["hidden"][-1].mean(0)
np.save(WORK / "out_mean_final.npy", mean_final.astype(np.float32))
print("mean logp per letter", np.round(res["logp"].mean(0), 3), "|final mean|", float(np.linalg.norm(mean_final)))
save(f"{EDIT}_train", res, hidden=False)
report(f"{EDIT}_train", res)
full_dev(lab, EDIT, EDIT)

with torch.no_grad():
    block = lab.layers[BLOCK]
    columns = block.mlp.down_proj.weight.float()
    axis = torch.as_tensor(write_dir, device=columns.device)
    cosine = (axis @ columns) / columns.norm(dim=0)
    neuron = int(torch.argmax(cosine))
    print("boost neuron", neuron, "cos", float(cosine[neuron]))
    for gain, threshold in ALTERNATES:
        relu_neuron(block, read_dir, write_dir, read_mean, gain, threshold, SHARP, neuron=neuron)
        name = f"alt_g{gain:g}_t{threshold:g}_train"
        alt = lab.forward("train", TRAIN_PROBE, note=f"{EDIT} with boost gain {gain} thr {threshold}")
        save(name, alt, hidden=False)
        report(name, alt)
