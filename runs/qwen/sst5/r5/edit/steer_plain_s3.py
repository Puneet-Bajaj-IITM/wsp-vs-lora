"""SST-5 edit: steer the answer away from the 'very' classes with attention value biases, then cool the logits.

Softmax attention weights sum to one, so a value bias b_v adds the constant vector M @ b_v to the
residual stream (M = o_proj blocks summed over the query heads sharing each kv head). At layers
17, 19 and 21 we add the projection of c = (plain midpoint - very midpoint), taken from class
centroids of the answer-position residual one layer earlier, rescaled so its component along c
has full length. The final RMSNorm gain is scaled to act as a logit temperature.
"""
from pathlib import Path

import numpy as np
import torch

HEAD_DIM = 128
N_KV_HEADS = 2
VNEG, NEG, NEU, POS, VPOS = range(5)
STEER_LAYERS = (17, 19, 21)
STRENGTH = 1.0
LOGIT_TEMPERATURE = 0.32
CENTROIDS_FILE = Path(__file__).with_name("steer_plain_s3_centroids.npy")


def reach_matrix(model, layer: int) -> torch.Tensor:
    o = model.model.layers[layer].self_attn.o_proj.weight.double().cpu()
    per_group = o.shape[1] // HEAD_DIM // N_KV_HEADS
    return o.reshape(o.shape[0], N_KV_HEADS, per_group, HEAD_DIM).sum(2).reshape(o.shape[0], -1)


def apply(model, tok) -> None:
    centroids = np.load(CENTROIDS_FILE).astype(np.float64)
    with torch.no_grad():
        for layer, mu in zip(STEER_LAYERS, centroids):
            target = torch.tensor(STRENGTH * ((mu[NEG] + mu[POS]) / 2 - (mu[VNEG] + mu[VPOS]) / 2))
            m = reach_matrix(model, layer)
            b = torch.linalg.pinv(m) @ target
            covered = float((m @ b).norm() / target.norm()) ** 2
            bias = model.model.layers[layer].self_attn.v_proj.bias
            bias.add_((b / covered).to(bias.dtype).to(bias.device))
        model.model.norm.weight.mul_(LOGIT_TEMPERATURE)
