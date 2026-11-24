"""Reads the weights the circuit builder needs and writes its patches back into the model."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import _lk_build as B
import _lk_design

DIAG_ENV = "WSP_DIAG_DIR"
HD = B.HEAD_DIM


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float64)


def put(param: torch.Tensor, index, value: np.ndarray) -> None:
    assert np.abs(value).max() < 3e4, float(np.abs(value).max())
    param[index] = torch.as_tensor(value, dtype=param.dtype, device=param.device)


def group_rows(group: int) -> slice:
    return slice(group * HD, (group + 1) * HD)


def head_rows(head: int) -> slice:
    return slice(head * HD, (head + 1) * HD)


def run(model, tok, overrides: dict | None = None) -> None:
    for key, value in (overrides or {}).items():
        setattr(B, key, value)
    design = _lk_design.load(tok)
    emb = model.get_input_embeddings().weight
    layer0, layer1 = model.model.layers[0], model.model.layers[1]
    weights = {
        "emb_vocab": to_np(emb[torch.as_tensor(design.vocab, device=emb.device)]),
        "emb_letters": to_np(emb[design.letter_tokens]),
        "ln1": to_np(layer0.input_layernorm.weight), "ln2": to_np(layer0.post_attention_layernorm.weight),
        "ln1_next": to_np(layer1.input_layernorm.weight), "ln_final": to_np(model.model.norm.weight),
    }
    built = B.build(weights, design)
    p = built.patches
    ids, rows = p["emb_rows"]
    put(emb, torch.as_tensor(ids, device=emb.device), rows)
    # padded positions get no attention output (fully masked rows), so their embedding must be MLP-invisible
    put(emb, tok.pad_token_id, B.Z_PIN * built.diag["z_dir"])

    a0, patch0 = layer0.self_attn, p["l0_attn"]
    group = group_rows(B.GROUP)
    for head, bias in zip((B.BAG_HEAD, B.BACK2_HEAD), patch0["q_b"]):
        a0.q_proj.weight[head_rows(head)] = 0
        put(a0.q_proj.bias, head_rows(head), bias)
    put(a0.k_proj.weight, group, patch0["k_w"])
    put(a0.k_proj.bias, group, patch0["k_b"])
    put(a0.v_proj.weight, group, patch0["v_w"])
    put(a0.v_proj.bias, group, patch0["v_b"])
    put(a0.o_proj.weight, slice(None), patch0["o_w"])

    mlp, patch_mlp = layer0.mlp, p["l0_mlp"]
    neurons = slice(0, patch_mlp["gate"].shape[0])
    put(mlp.gate_proj.weight, neurons, patch_mlp["gate"])
    put(mlp.up_proj.weight, neurons, patch_mlp["up"])
    put(mlp.down_proj.weight, (slice(None), neurons), patch_mlp["down"].T)

    a1, patch1 = layer1.self_attn, p["l1_attn"]
    lo, hi = B.LOW_PAIR
    base = B.GROUP * HD
    pick = head_rows(B.PICK_HEAD)
    a1.q_proj.weight[pick] = 0
    a1.q_proj.bias[pick] = 0
    a1.q_proj.bias[B.PICK_HEAD * HD + lo] = float(patch1["q_bias"])
    put(a1.k_proj.weight, base + lo, patch1["k_row"])
    a1.k_proj.weight[base + hi] = 0
    a1.k_proj.bias[base + lo] = 0
    a1.k_proj.bias[base + hi] = 0
    n_letters = patch1["v_rows"].shape[0]
    put(a1.v_proj.weight, slice(base, base + n_letters), patch1["v_rows"])
    a1.v_proj.bias[base:base + n_letters] = 0
    a1.o_proj.weight[:, pick] = 0
    put(a1.o_proj.weight, (slice(None), slice(B.PICK_HEAD * HD, B.PICK_HEAD * HD + n_letters)), patch1["o_cols"])

    diag_dir = os.environ.get(DIAG_ENV)
    if diag_dir:
        np.savez(Path(diag_dir) / "design.npz", **built.diag)
    summary = {k: float(built.diag[k]) for k in ("beta", "flag_rms", "u_norm", "fill_norm", "gate_max", "down_max")}
    print("lookup circuit built", summary, flush=True)
