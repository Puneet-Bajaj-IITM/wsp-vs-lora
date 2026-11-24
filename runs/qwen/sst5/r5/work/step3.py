"""Job 3: multi-layer steering variants on train[512:1024]; saves logp and mean final hidden per config."""
import json
from pathlib import Path

import numpy as np

from steer_lib import add_value_bias, bias_for, class_vector, greedy_biases
from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r5")
CENTROID_LAYERS = (16, 18, 20, 22)
IDX = list(range(512, 1024))
EVEN = {16: 16, 17: 16, 18: 18, 19: 18, 20: 20, 21: 20, 15: 16}
# ("sum", [(layer, a, b)]) independent matched projections ; ("proj", ...) unscaled ; ("greedy", layers, centroid_layer, a, b, extra)
CONFIGS = {
    "S3_b1": ("sum", [(17, 0, 1), (19, 0, 1), (21, 0, 1)]),
    "S3_b1.5": ("sum", [(17, 0, 1.5), (19, 0, 1.5), (21, 0, 1.5)]),
    "S3_b2": ("sum", [(17, 0, 2), (19, 0, 2), (21, 0, 2)]),
    "S6_b0.5": ("sum", [(l, 0, 0.5) for l in range(16, 22)]),
    "S6_b1": ("sum", [(l, 0, 1) for l in range(16, 22)]),
    "S4_b1": ("sum", [(15, 0, 1), (17, 0, 1), (19, 0, 1), (21, 0, 1)]),
    "S3_a1b1": ("sum", [(17, 1, 1), (19, 1, 1), (21, 1, 1)]),
    "S3_a1b2": ("sum", [(17, 1, 2), (19, 1, 2), (21, 1, 2)]),
    "S3_b4_proj": ("proj", [(17, 0, 4), (19, 0, 4), (21, 0, 4)]),
    "G16_b2": ("greedy", (15, 16, 17), 16, 0, 2, []),
    "G16_b3": ("greedy", (15, 16, 17), 16, 0, 3, []),
    "G16_b2_S2_b1": ("greedy", (15, 16, 17), 16, 0, 2, [(19, 0, 1), (21, 0, 1)]),
}

lab = Lab(edit=None)
model = lab.model
mus = dict(zip(CENTROID_LAYERS, np.load(WORK / "centroids.npy").astype(np.float32)))
exs = lab.examples("train")
shown = np.array([exs[i]["shown_option_ids"] for i in IDX])
logps, means = {}, {}
for name, spec in CONFIGS.items():
    plan = {}
    if spec[0] in ("sum", "proj"):
        for layer, a, b in spec[1]:
            bv, cos = bias_for(model, layer, class_vector(mus[EVEN[layer]], a, b), match=spec[0] == "sum")
            plan[layer] = bv
    else:
        _, layers, cl, a, b, extra = spec
        got, cos = greedy_biases(model, layers, class_vector(mus[cl], a, b))
        print(name, "greedy cos", round(cos, 3))
        plan.update(got)
        for layer, a2, b2 in extra:
            plan[layer], _ = bias_for(model, layer, class_vector(mus[EVEN[layer]], a2, b2))
    saved = {}
    for layer, bv in plan.items():
        bias = model.model.layers[layer].self_attn.v_proj.bias
        saved[layer] = bias.clone()
        add_value_bias(model, layer, bv)
    r = lab.forward("train", IDX, note=f"steer trial {name}", capture_layers=(-1,))
    for layer, b0 in saved.items():
        model.model.layers[layer].self_attn.v_proj.bias.copy_(b0)
    pred_cls = shown[np.arange(len(IDX)), r["logp"].argmax(1)]
    s = r["summary"]
    print(name, "acc", round(s["accuracy"], 4), "nll", round(s["nll"], 3), "cls", np.bincount(pred_cls, minlength=5).tolist(), "letters", s["pred_letter_hist"], "left", r["passes_left"], flush=True)
    logps[name] = r["logp"]
    means[name] = r["hidden"][-1].mean(0).astype(np.float32)
np.savez(WORK / "step3_logp.npz", **logps)
np.savez(WORK / "step3_hmean.npz", **means)
emb = model.model.embed_tokens.weight
print("letter row norms", [round(float(emb[t].float().norm()), 3) for t in (32, 33, 34, 35, 36)], "norm.weight mean", float(model.model.norm.weight.float().mean()))
