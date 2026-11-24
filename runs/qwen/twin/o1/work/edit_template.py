"""Option-text prior injection: one repurposed MLP neuron writes b(option text) x chosen-direction at option tokens,
plus a final-norm rescale (temperature). Constants derived in wsp_work/twin_o1 (difference of means, count ratios)."""
from pathlib import Path

import torch

PARAMS_FILE = Path(__file__).with_name(Path(__file__).stem + "_params.pt")


def apply(model, tok):
    p = torch.load(PARAMS_FILE, map_location="cpu")
    for inj in p["injections"]:
        mlp = model.model.layers[inj["layer"]].mlp
        n = inj["neuron"]
        dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
        mlp.gate_proj.weight[n] = inj["gate"].to(dev, dt)
        mlp.up_proj.weight[n] = inj["up"].to(dev, dt)
        mlp.down_proj.weight[:, n] = inj["down"].to(dev, dt)
    model.model.norm.weight.mul_(p["norm_scale"])
