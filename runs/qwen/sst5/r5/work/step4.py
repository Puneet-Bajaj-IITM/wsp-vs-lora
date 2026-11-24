"""Job 4: asymmetric steering variants at layers 17, 19, 21 on train[1024:1536]."""
from pathlib import Path

import numpy as np

from steer_lib import add_value_bias, bias_for, class_vector3
from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r5")
CENTROID_LAYERS = (16, 18, 20, 22)
LAYER_TO_CENTROID = {17: 16, 19: 18, 21: 20}
IDX = list(range(1024, 1536))
# name -> (b_neg, b_pos, a_neu)
CONFIGS = {
    "ref_.5_.5_0": (0.5, 0.5, 0.0),
    "n1_p.5": (1.0, 0.5, 0.0),
    "n1_p.25": (1.0, 0.25, 0.0),
    "n.5_p.5_a.5": (0.5, 0.5, 0.5),
    "n1_p.5_a.5": (1.0, 0.5, 0.5),
    "n1_p.25_a1": (1.0, 0.25, 1.0),
    "n.75_p.5_a.25": (0.75, 0.5, 0.25),
    "n.375_p.375": (0.375, 0.375, 0.0),
}

lab = Lab(edit=None)
model = lab.model
mus = dict(zip(CENTROID_LAYERS, np.load(WORK / "centroids.npy").astype(np.float32)))
exs = lab.examples("train")
shown = np.array([exs[i]["shown_option_ids"] for i in IDX])
logps = {}
for name, (bn, bp, a) in CONFIGS.items():
    saved = {}
    for layer, cl in LAYER_TO_CENTROID.items():
        bv, cos = bias_for(model, layer, class_vector3(mus[cl], bn, bp, a))
        bias = model.model.layers[layer].self_attn.v_proj.bias
        saved[layer] = bias.clone()
        add_value_bias(model, layer, bv)
    r = lab.forward("train", IDX, note=f"asym steer trial {name}")
    for layer, b0 in saved.items():
        model.model.layers[layer].self_attn.v_proj.bias.copy_(b0)
    pred_cls = shown[np.arange(len(IDX)), r["logp"].argmax(1)]
    s = r["summary"]
    print(name, "acc", round(s["accuracy"], 4), "nll", round(s["nll"], 3), "cls", np.bincount(pred_cls, minlength=5).tolist(), "letters", s["pred_letter_hist"], "left", r["passes_left"], flush=True)
    logps[name] = r["logp"]
np.savez(WORK / "step4_logp.npz", **logps)
