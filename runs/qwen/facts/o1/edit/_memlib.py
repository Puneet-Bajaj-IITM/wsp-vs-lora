"""Shared apply() for facts/o1 person-memory edits: overwrite chosen MLP neurons of given layers
with person detectors (gate/up rows) and value writers (down columns) stored beside the edit."""
from pathlib import Path

import torch


def apply_memory(model, stem: str) -> None:
    here = Path(__file__).resolve().parent
    blocks = torch.load(here / f"{stem}_mem.pt", map_location="cpu")
    for blk in blocks:
        mlp = model.model.layers[int(blk["layer"])].mlp
        dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
        idx = blk["neurons"].long().to(dev)
        mlp.gate_proj.weight[idx] = blk["gate"].to(dev, dt)
        mlp.up_proj.weight[idx] = blk["up"].to(dev, dt)
        mlp.down_proj.weight[:, idx] = blk["down"].to(dev, dt)
