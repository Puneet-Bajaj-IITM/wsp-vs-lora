"""Key-side prior edit for the Twin-2K task (run r4).

Mechanism: at layers 18, 20 and 21 the answer position reads the options with a fixed query; an option's
"attractiveness" is the projection of its last/end-token key onto that query. This edit adds a rank-1 term to
k_proj at each of those layers,  k += beta * u * (w . x),  where
  w  is a read-out of the k_proj input x (difference of means between the options respondents chose and the
     options the base model prefers; constants in kq_final_w.npy), and
  u  is a sparse key direction in low-frequency RoPE dims with  mean_query . u = sqrt(head_dim)  for the
     option-reading heads (kq_final_u.npy).
The final RMSNorm gain is scaled by tau, which is an exact temperature on the logits (no lm_head bias).
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAYERS = (18, 20, 21)
READOUT_FILE = "kq_final_w.npy"
KEY_DIRECTION_FILE = "kq_final_u.npy"
CONFIG_FILE = "kq_final_cfg.npy"


def apply(model, tok):
    readout = torch.tensor(np.load(HERE / READOUT_FILE))
    key_dir = torch.tensor(np.load(HERE / KEY_DIRECTION_FILE))
    beta, tau = (float(v) for v in np.load(HERE / CONFIG_FILE))
    for i, layer in enumerate(LAYERS):
        weight = model.model.layers[layer].self_attn.k_proj.weight
        rows = torch.nonzero(key_dir[i]).squeeze(1)
        delta = beta * key_dir[i, rows, None].float() * readout[i][None, :].float()
        weight[rows] = (weight[rows].float() + delta.to(weight.device)).to(weight.dtype)
    gain = model.model.norm.weight
    gain.copy_((gain.float() * tau).to(gain.dtype))
