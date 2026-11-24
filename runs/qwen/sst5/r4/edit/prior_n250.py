"""Class-prior repair for SST-5 by one hijacked MLP neuron plus a temperature.

Base Qwen almost never answers 'neutral' (1.5% of predictions, 19% of labels) and over-uses 'very negative'.
The normed residual at the answer position is nearly constant (cos 0.999 with its mean), so
m = mean(x)/|mean(x)|^2 acts as a bias input (m.x ~= 1). Neuron J of MLP 19 is overwritten with
gate = BETA*m, up = m, down = VEC/silu(BETA), which adds the constant VEC to the residual there.
VEC = sum_c b_c * sigma_c * dhat_c, where dhat_c is the unit mean offset of layer-19 hidden states of
train examples the base model assigns to sentiment class c, sigma_c the projection std, b = [-1.5, 0, 2.5, 0, 0].
The final RMSNorm weight is divided by TEMPERATURE (changes NLL only, not the argmax).
"""
import math
from pathlib import Path

import numpy as np
import torch

LAYER = 19
BETA = 6.0
TEMPERATURE = 4.0
HERE = Path(__file__).resolve().parent
NAME = Path(__file__).stem


def apply(model, tok):
    m = np.load(HERE / f"{NAME}_m.npy")
    vec = np.load(HERE / f"{NAME}_vec.npy")
    mlp = model.model.layers[LAYER].mlp
    gate, up, down = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    with torch.no_grad():
        usage = gate.float().norm(dim=1) * up.float().norm(dim=1) * down.float().norm(dim=0)
        j = int(usage.argmin())
        m_t = torch.as_tensor(m, dtype=torch.float32, device=gate.device)
        v_t = torch.as_tensor(vec, dtype=torch.float32, device=gate.device)
        silu_beta = BETA / (1.0 + math.exp(-BETA))
        gate[j] = (BETA * m_t).to(gate.dtype)
        up[j] = m_t.to(up.dtype)
        down[:, j] = (v_t / silu_beta).to(down.dtype)
        model.model.norm.weight.div_(TEMPERATURE)
