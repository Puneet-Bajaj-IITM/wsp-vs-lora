"""Constant class-steering vector at one pre-binding attention block, plus output temperature.

Constants beside this file:
  neutral_steer_vec.npy  (1536,) vector added to the residual by the chosen attention block
  neutral_steer_cfg.npy  [layer, temperature]

Injection: one value coordinate of a kv head becomes a constant (v_proj row zeroed, bias set to
INJECT_BIAS); the o_proj columns reading it are replaced by vec / (heads_per_kv * INJECT_BIAS), so the
block adds exactly `vec` at every position. Temperature: the final RMSNorm gain is divided by T.
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VEC_PATH = HERE / "neutral_steer_vec.npy"
CFG_PATH = HERE / "neutral_steer_cfg.npy"
INJECT_BIAS = 1.0


def pick_coordinate(attn, heads_per_kv: int, head_dim: int) -> tuple[int, list[int]]:
    v_norm = attn.v_proj.weight.float().norm(dim=1)
    o_col = attn.o_proj.weight.float().norm(dim=0)
    n_kv = v_norm.numel() // head_dim
    o_group = o_col.view(n_kv, heads_per_kv, head_dim).sum(1).reshape(-1)
    coord = int((v_norm * o_group).argmin())
    kv_head, dim = divmod(coord, head_dim)
    return coord, [(kv_head * heads_per_kv + h) * head_dim + dim for h in range(heads_per_kv)]


def apply(model, tok) -> None:
    layer, temperature = np.load(CFG_PATH).tolist()
    vec = torch.from_numpy(np.load(VEC_PATH).astype(np.float32))
    attn = model.model.layers[int(layer)].self_attn
    cfg = model.config
    heads_per_kv = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    with torch.no_grad():
        coord, cols = pick_coordinate(attn, heads_per_kv, head_dim)
        attn.v_proj.weight[coord] = 0
        attn.v_proj.bias[coord] = INJECT_BIAS
        weight = attn.o_proj.weight
        column = (vec.to(weight.device) / (heads_per_kv * INJECT_BIAS)).to(weight.dtype)
        weight[:, cols] = column[:, None]
        model.model.norm.weight.div_(float(temperature))
