"""Key-side edit: k_proj[L] rows of selected key dims += beta * u (x) w."""
import math

import torch

TOP_DIMS = 16
LOW_FREQ_START = 0.5


def low_freq_dims(hd: int) -> torch.Tensor:
    half = hd // 2
    i = torch.arange(half)
    low = i[i >= int(LOW_FREQ_START * half)]
    return torch.cat([low, low + half])


def key_direction(qbar: torch.Tensor, heads: tuple, n_kv: int) -> tuple[torch.Tensor, dict]:
    """qbar (H,hd) mean pre-RoPE answer-position queries. Returns u (n_kv*hd) with qbar_h . u = sqrt(hd) for the
    selected heads, supported on each head's largest low-frequency dims, and the implied gain of every head."""
    H, hd = qbar.shape
    per_group = H // n_kv
    low = low_freq_dims(hd)
    u = torch.zeros(n_kv, hd)
    for g in range(n_kv):
        sel = [h for h in heads if h // per_group == g]
        if not sel:
            continue
        dims = set()
        for h in sel:
            top = qbar[h, low].abs().topk(min(TOP_DIMS, len(low))).indices
            dims |= set(low[top].tolist())
        dims = torch.tensor(sorted(dims))
        Q = qbar[sel][:, dims].double()
        sol = torch.linalg.pinv(Q) @ torch.full((len(sel),), math.sqrt(hd), dtype=torch.float64)
        u[g, dims] = sol.float()
    gains = {h: round(float(qbar[h] @ u[h // per_group]) / math.sqrt(hd), 3) for h in range(H)}
    return u.reshape(-1), gains


def add_rank1(weight: torch.Tensor, u: torch.Tensor, w: torch.Tensor, scale: float) -> int:
    """weight (out,in) += scale * u w^T on the rows where u != 0. Returns the number of entries touched."""
    rows = torch.nonzero(u).squeeze(1)
    delta = scale * u[rows, None].float() * w[None, :].float()
    weight[rows] = (weight[rows].float() + delta.to(weight.device)).to(weight.dtype)
    return int(rows.numel() * w.numel())
