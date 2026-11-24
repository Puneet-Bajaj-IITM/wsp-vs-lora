"""Job 2: steering trials on train[0:512] (paired with the base pass on the same items)."""
import json
from pathlib import Path

import numpy as np
import torch

from steer_lib import add_value_bias, bias_for, class_vector
from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r5")
CENTROID_LAYERS = (16, 18, 20, 22)
IDX = list(range(512))
VERY_TOKEN = 1602
# name -> list of (layer, centroid_layer, a_neutral, b_plain, match) ; or ("embed", scale)
CONFIGS = {
    "L19_b1": [(19, 18, 0, 1, True)],
    "L19_b2": [(19, 18, 0, 2, True)],
    "L19_b2_proj": [(19, 18, 0, 2, False)],
    "L19_b4": [(19, 18, 0, 4, True)],
    "L19_a1b2": [(19, 18, 1, 2, True)],
    "L19_a2": [(19, 18, 2, 0, True)],
    "L17_b2": [(17, 16, 0, 2, True)],
    "L21_b2": [(21, 20, 0, 2, True)],
    "L23_b2": [(23, 22, 0, 2, True)],
    "L17_19_21_b1": [(17, 16, 0, 1, True), (19, 18, 0, 1, True), (21, 20, 0, 1, True)],
    "L19_b3": [(19, 18, 0, 3, True)],
    "L19_bneg2": [(19, 18, 0, -2, True)],
    "L21_b4": [(21, 20, 0, 4, True)],
    "L17_19_21_b2": [(17, 16, 0, 2, True), (19, 18, 0, 2, True), (21, 20, 0, 2, True)],
    "L19_a2b2": [(19, 18, 2, 2, True)],
    "L21_a2": [(21, 20, 2, 0, True)],
    "embed_very_0.6": "embed",
}
EMBED_SCALE = 0.6

lab = Lab(edit=None)
model = lab.model
mus = dict(zip(CENTROID_LAYERS, np.load(WORK / "centroids.npy").astype(np.float32)))
for layer in range(14, 26):
    cl = min(CENTROID_LAYERS, key=lambda c: abs(c - (layer - 1)))
    _, cos = bias_for(model, layer, class_vector(mus[cl], 0, 1))
    _, cos_n = bias_for(model, layer, class_vector(mus[cl], 1, 0))
    print(f"layer {layer}: reachable cos for plain-vs-very {cos:.3f}, neutral {cos_n:.3f}")
exs = lab.examples("train")
shown = np.array([exs[i]["shown_option_ids"] for i in IDX])
out = {}
for name, spec in CONFIGS.items():
    saved = {}
    if spec == "embed":
        emb = model.model.embed_tokens.weight
        saved_row = emb[VERY_TOKEN].clone()
        emb[VERY_TOKEN] *= EMBED_SCALE
    else:
        for layer, cl, a, b, match in spec:
            bias = model.model.layers[layer].self_attn.v_proj.bias
            saved[layer] = bias.clone()
            bv, cos = bias_for(model, layer, class_vector(mus[cl], a, b), match)
            add_value_bias(model, layer, bv)
            print(name, "layer", layer, "cos", round(cos, 3), "|b|", float(bv.norm()), "|b_orig|", float(saved[layer].float().norm()))
    r = lab.forward("train", IDX, note=f"steer trial {name}")
    if spec == "embed":
        emb[VERY_TOKEN] = saved_row
    for layer, b0 in saved.items():
        model.model.layers[layer].self_attn.v_proj.bias.copy_(b0)
    pred_cls = shown[np.arange(len(IDX)), r["logp"].argmax(1)]
    s = r["summary"]
    print(name, "acc", round(s["accuracy"], 4), "nll", round(s["nll"], 3), "pred class hist", np.bincount(pred_cls, minlength=5).tolist(), "letters", s["pred_letter_hist"], json.dumps(s["by_block"]), "left", r["passes_left"], flush=True)
    out[name] = r["logp"]
np.savez(WORK / "step2_logp.npz", **out)
