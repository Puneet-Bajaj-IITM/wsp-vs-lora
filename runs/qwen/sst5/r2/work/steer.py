"""Exact constant-vector injection into the residual stream using existing attention parameters.

One value coordinate of a kv head is turned into a constant: its v_proj row is zeroed and its bias
set to INJECT_BIAS, so every query head sharing that kv head emits exactly INJECT_BIAS on that
coordinate at every position (attention weights sum to 1). The matching o_proj columns are replaced
by vec / (heads_per_kv * INJECT_BIAS), so the block adds exactly `vec` to the residual.
"""
import torch

INJECT_BIAS = 1.0


def pick_coordinate(attn, heads_per_kv: int, head_dim: int) -> tuple[int, list[int]]:
    """Least important value coordinate: smallest (v_proj row norm x summed o_proj column norm)."""
    v_norm = attn.v_proj.weight.float().norm(dim=1)
    o_col = attn.o_proj.weight.float().norm(dim=0)
    n_kv = v_norm.numel() // head_dim
    o_group = o_col.view(n_kv, heads_per_kv, head_dim).sum(1).reshape(-1)
    coord = int((v_norm * o_group).argmin())
    kv_head, dim = divmod(coord, head_dim)
    cols = [(kv_head * heads_per_kv + h) * head_dim + dim for h in range(heads_per_kv)]
    return coord, cols


@torch.no_grad()
def inject_constant(model, layer: int, vec: torch.Tensor) -> dict:
    """Mutates attention block `layer` in place; returns the overwritten tensors for restoring."""
    attn = model.model.layers[layer].self_attn
    cfg = model.config
    heads_per_kv = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    coord, cols = pick_coordinate(attn, heads_per_kv, head_dim)
    backup = {"layer": layer, "coord": coord, "cols": cols,
              "v_row": attn.v_proj.weight[coord].clone(), "v_bias": attn.v_proj.bias[coord].clone(),
              "o_cols": attn.o_proj.weight[:, cols].clone()}
    attn.v_proj.weight[coord] = 0
    attn.v_proj.bias[coord] = INJECT_BIAS
    column = (vec.to(attn.o_proj.weight.device).float() / (heads_per_kv * INJECT_BIAS)).to(attn.o_proj.weight.dtype)
    attn.o_proj.weight[:, cols] = column[:, None]
    return backup


@torch.no_grad()
def restore(model, backup: dict) -> None:
    attn = model.model.layers[backup["layer"]].self_attn
    attn.v_proj.weight[backup["coord"]] = backup["v_row"]
    attn.v_proj.bias[backup["coord"]] = backup["v_bias"]
    attn.o_proj.weight[:, backup["cols"]] = backup["o_cols"]
