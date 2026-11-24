"""Job 3: stronger steering at layers 19/20 (intensity down, neutral up), then validate the best on a fresh chunk."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import editlib
from labutil import ARRAYS, PASS_SIZE, run_pass
from wsp.harness import Lab

TOP_K = 512
TRAIN_A = range(0, PASS_SIZE)
TRAIN_B = range(PASS_SIZE, 2 * PASS_SIZE)
# (tag, layer, intensity scale (subtracted), neutral scale (added))
VARIANTS = [
    ("L19_i4_n0", 19, 4.0, 0.0),
    ("L19_i6_n0", 19, 6.0, 0.0),
    ("L20_i2.5_n0", 20, 2.5, 0.0),
    ("L20_i4_n0", 20, 4.0, 0.0),
    ("L19_i0_n2.5", 19, 0.0, 2.5),
    ("L19_i0_n5", 19, 0.0, 5.0),
    ("L19_i4_n2.5", 19, 4.0, 2.5),
]

STEER_LAYERS = json.loads((ARRAYS / "steer_layers.json").read_text())
true_means = dict(zip(STEER_LAYERS, np.load(ARRAYS / "steer_true_means.npy").astype(np.float32)))
pred_means = dict(zip(STEER_LAYERS, np.load(ARRAYS / "steer_pred_means.npy").astype(np.float32)))
keys = dict(zip(STEER_LAYERS, torch.tensor(np.load(ARRAYS / "steer_keys.npy").astype(np.float32))))


def steer_vector(layer, intensity_scale, neutral_scale):
    intensity = 0.5 * ((pred_means[layer][0] - pred_means[layer][1]) + (pred_means[layer][4] - pred_means[layer][3]))
    neutral = true_means[layer][2] - 0.5 * (true_means[layer][1] + true_means[layer][3])
    return torch.tensor(-intensity_scale * intensity + neutral_scale * neutral)


lab = Lab()
tok, model = lab.tok, lab.model
layers = sorted({v[1] for v in VARIANTS})
original_down = {l: model.model.layers[l].mlp.down_proj.weight.clone() for l in layers}
editlib.swap_rows(model, tok)


def apply_variant(layer, intensity_scale, neutral_scale):
    for l, saved in original_down.items():
        model.model.layers[l].mlp.down_proj.weight.copy_(saved)
    if layer is not None:
        editlib.rank_one_steer(model, layer, steer_vector(layer, intensity_scale, neutral_scale), keys[layer].float(), TOP_K)


scores = {}
for tag, layer, i_scale, n_scale in VARIANTS:
    apply_variant(layer, i_scale, n_scale)
    r = run_pass(lab, "train", TRAIN_A, f"steer {tag} on train A", f"s3_{tag}_A")
    scores[tag] = r["summary"]["accuracy"]
best = max(VARIANTS, key=lambda v: scores[v[0]])
print("BEST on A", best, scores, flush=True)
apply_variant(None, 0, 0)
run_pass(lab, "train", TRAIN_B, "swap baseline on train B", "swap_trainB")
apply_variant(*best[1:])
run_pass(lab, "train", TRAIN_B, f"best steer {best[0]} on train B", f"s3_{best[0]}_B")
(ARRAYS / "s3_best.json").write_text(json.dumps({"best": best, "scores": scores}))
