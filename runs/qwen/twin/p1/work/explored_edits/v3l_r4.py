"""Gated selector edit (v3l_r4).

Layer-21 heads 8-11 read the option texts from the answer position and decide which letter gets copied.
This edit adds to their q_proj rows:   dq = sum_s STRENGTH[s] * sqrt(head_dim) * gate_s(x) * delta_s
  gate_s(x) = C_s.x - off_s  is ~1 at the answer position of twin prompts of answer style s
              (agree scale, support scale, money split, vocabulary) and ~0 elsewhere;
  delta_s   = mean key of the options respondents chose minus mean key of the options they did not choose.
So attention moves toward the kind of option people usually pick. Only q_proj weight/bias rows of the listed heads change."""
from pathlib import Path

import torch

TAG = "v3l"
STRENGTH = [0.889, 2.686, 2.782, 4.756]
HEADS = {21: [8, 9, 10, 11]}
HEAD_DIM = 128
KV_GROUP = 6


def apply(model, tok):
    for layer, heads in HEADS.items():
        t = torch.load(Path(__file__).with_name(f"sel_{TAG}_L{layer}.pt"))
        q = model.model.layers[layer].self_attn.q_proj
        c, off = t["Cq"].float() * t["Cs"], t["off"].float()
        s = torch.tensor(STRENGTH).unsqueeze(1) * HEAD_DIM ** 0.5
        for h in heads:
            d = t["D"][h // KV_GROUP].float() * s
            rows = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            q.weight[rows] += (d.T @ c).to(q.weight)
            q.bias[rows] -= (d.T @ off).to(q.bias)
