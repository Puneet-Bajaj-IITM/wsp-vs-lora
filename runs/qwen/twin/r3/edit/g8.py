"""Question-gated attention-query edit.

For each selected answer-selection head (layer, head):
    W_Q[head] += beta * D^T R        b_Q[head] -= beta * D^T c
R (sets x d_model) reads which option set the question uses from the answer-position input
(difference-of-means directions); D (sets x head_dim) is, per option set, a popularity-weighted sum of
mean option key vectors. Net effect: the head's attention score for an option line moves by about
beta * tau(option), with tau from train answer frequencies minus the base model's own preference.
Constants shared by all variants live in gates_const.pt; <name>_const.pt holds the head list and beta.
"""
from pathlib import Path

import torch

CONST_SUFFIX = "_const.pt"
GATES_FILE = "gates_const.pt"


def apply(model, tok):
    here = Path(__file__)
    const = torch.load(here.with_name(here.stem + CONST_SUFFIX))
    gates = torch.load(here.with_name(GATES_FILE))
    hd = model.config.hidden_size // model.config.num_attention_heads
    for layer, head, table, beta in const["heads"]:
        q_proj = model.model.layers[layer].self_attn.q_proj
        D = gates["D"][table].float()
        if const.get("sets") is not None:
            mask = torch.zeros(D.shape[0])
            mask[const["sets"]] = 1.0
            D = D * mask[:, None]
        R, c = gates["R"][layer].float(), gates["c"][layer].float()
        rows = slice(head * hd, (head + 1) * hd)
        q_proj.weight[rows] += (beta * D.T @ R).to(q_proj.weight.device, q_proj.weight.dtype)
        q_proj.bias[rows] += (-beta * D.T @ c).to(q_proj.bias.device, q_proj.bias.dtype)
