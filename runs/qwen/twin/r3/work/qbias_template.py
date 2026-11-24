"""Attention-query bias edit: adds a fixed vector to q_proj.bias of selected heads.

Each row of <name>_delta.npy is (layer, head, 128-d vector). The vector is a popularity-weighted
sum of mean key vectors of option texts, so the head's final-position query leans toward the
positions of options that train respondents chose most often.
"""
from pathlib import Path

import numpy as np
import torch

DELTA_SUFFIX = "_delta.npy"


def apply(model, tok):
    here = Path(__file__)
    rows = np.load(here.with_name(here.stem + DELTA_SUFFIX))
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    for row in rows:
        layer, head = int(row[0]), int(row[1])
        bias = model.model.layers[layer].self_attn.q_proj.bias
        bias[head * hd:(head + 1) * hd] += torch.tensor(row[2:], dtype=bias.dtype, device=bias.device)
