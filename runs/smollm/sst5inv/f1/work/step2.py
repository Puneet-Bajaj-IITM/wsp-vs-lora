"""Step 2: where does valence arrive / the letter form, and does amplifying the valence axis or the
" very" embedding make the option matcher use intensity?  All probes sit on top of swap_v1."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, TRAIN_PROBE, report, save  # noqa: E402
from weightlib import Snapshot, linear_map_neurons, spare_neurons  # noqa: E402

from wsp.harness import Lab  # noqa: E402

WORK = Path(__file__).resolve().parent
BLOCK = 17
READ_LAYER = 16
AMP_GAINS = (1.0, 3.0, -0.5)
VERY_SCALE = 2.0
BASE_CAPTURE = (14, 15, 17, 18)
AMP_CAPTURE = (17, 20)

valence = np.load(WORK / "consts" / f"valence_dir_L{READ_LAYER}.npy")
mean_vec = np.load(WORK / "consts" / f"mean_L{READ_LAYER}.npy")

lab = Lab(edit="swap_v1")
block = lab.layers[BLOCK]
embed = lab.model.get_input_embeddings().weight
norm_w = block.post_attention_layernorm.weight.float()
print("norm weight |min| / mean / max", norm_w.abs().min().item(), norm_w.abs().mean().item(), norm_w.abs().max().item())
down_norms = block.mlp.down_proj.weight.float().norm(dim=0)
print("down col norms min/median/max", down_norms.min().item(), down_norms.median().item(), down_norms.max().item(),
      "spare", spare_neurons(block, 4))
print("mlp shapes", tuple(block.mlp.gate_proj.weight.shape), tuple(block.mlp.down_proj.weight.shape),
      "act", type(block.mlp.act_fn).__name__)
row_norms = embed.float().norm(dim=1)
for word in (" very", " negative", " positive", " neutral", "A", "B", "C", "D", "E"):
    tid = lab.tok.encode(word, add_special_tokens=False)[0]
    print("EMB", repr(word), tid, round(row_norms[tid].item(), 3))
print("embedding row norm median", row_norms.median().item(), flush=True)

snap = Snapshot(lab.model, BLOCK)

with torch.no_grad():
    base = lab.forward("train", TRAIN_PROBE, note="swap_v1 capture layers 14-18", capture_layers=BASE_CAPTURE)
    save("swapcap_train", base)
    report("swapcap_train", base)

    for gain in AMP_GAINS:
        used = linear_map_neurons(block, valence, valence, mean_vec, gain)
        name = f"amp{gain:+.1f}_train"
        res = lab.forward("train", TRAIN_PROBE, note=f"valence amp gain {gain} block {BLOCK}", capture_layers=AMP_CAPTURE)
        snap.restore()
        proj = {layer: res["hidden"][layer] @ valence for layer in AMP_CAPTURE}
        print(name, "neurons", used, {layer: (round(float(p.mean()), 1), round(float(p.std()), 1)) for layer, p in proj.items()})
        save(name, res, hidden=(gain == AMP_GAINS[0]))
        report(name, res)

    very = lab.tok.encode(" very", add_special_tokens=False)[0]
    embed[very] = embed[very] * VERY_SCALE
    res = lab.forward("train", TRAIN_PROBE, note=f"very embedding x{VERY_SCALE}")
    snap.restore()
    save("very2_train", res, hidden=False)
    report("very2_train", res)
print("files", sorted(p.name for p in OUT.iterdir()))
