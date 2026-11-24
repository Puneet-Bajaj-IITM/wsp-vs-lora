"""Intent-routing edit: a few low-weight MLP neurons are repurposed as a last-position-gated linear map
from class-mean (centroid) evidence to answer-intent directions; plus a logit temperature on the final norm.

Constants in <stem>_consts.pt next to this file. All rows live in the MLP's input space
(post_attention_layernorm output): layer (int or None), gate (d,), ups (K, d), downs (K, d) in residual units,
temperature (float).
"""
from pathlib import Path

import torch

CONSTS = Path(__file__).with_name(Path(__file__).stem + "_consts.pt")


def pick_neurons(mlp, k):
    score = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
             * mlp.down_proj.weight.float().norm(dim=0))
    return torch.argsort(score)[:k]


def apply(model, tok):
    c = torch.load(CONSTS, map_location="cpu")
    if c.get("layer") is not None:
        mlp = model.model.layers[int(c["layer"])].mlp
        dt, dev = mlp.gate_proj.weight.dtype, mlp.gate_proj.weight.device
        for j, i in enumerate(pick_neurons(mlp, c["ups"].shape[0]).tolist()):
            mlp.gate_proj.weight[i] = c["gate"].to(dev, dt)
            mlp.up_proj.weight[i] = c["ups"][j].to(dev, dt)
            mlp.down_proj.weight[:, i] = c["downs"][j].to(dev, dt)
    model.model.norm.weight.mul_(1.0 / float(c["temperature"]))
