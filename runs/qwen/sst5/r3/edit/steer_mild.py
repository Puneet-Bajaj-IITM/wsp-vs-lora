"""Steer the answer-position sentiment code toward mild (A/mildperp x2) via v_proj biases of layers (16, 17, 18, 19); scale final-norm gain by 0.26 for calibration."""
from pathlib import Path

import numpy as np
import torch

LAYERS = (16, 17, 18, 19)
LOGIT_SCALE = 0.26
BIAS_FILE = "steer_mild_vbias.npy"


def apply(model, tok):
    deltas = np.load(Path(__file__).with_name(BIAS_FILE))
    with torch.no_grad():
        for layer, delta in zip(LAYERS, deltas):
            bias = model.model.layers[layer].self_attn.v_proj.bias
            bias.add_(torch.tensor(delta).to(device=bias.device, dtype=bias.dtype))
        model.model.norm.weight.mul_(LOGIT_SCALE)
