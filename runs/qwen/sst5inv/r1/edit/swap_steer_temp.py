"""sst5inv edit: mirror the polarity label tokens, damp the model's "very" bias, and soften over-confident letter logits.

1. Swap the (tied) embedding rows of " negative" and " positive": every option line now reads as its mirror image,
   so the model's ordinary sentiment readout lands on the mirrored label.
2. Rank-one update of one MLP down_proj: adds a fixed steering vector (difference of mean answer-position residuals,
   "model picked a very-option" minus "model picked a plain option", sign flipped) whenever the MLP activation
   resembles the mean answer-position activation.
3. Scale the final RMSNorm weight by a temperature (changes confidence only, never the argmax).
"""
from pathlib import Path

import numpy as np
import torch

POLARITY_TOKENS = (" negative", " positive")
STEER_LAYER = 20
TEMPERATURE = 0.25
HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
STEER_FILE = HERE / f"{STEM}_steer.npy"
KEY_INDEX_FILE = HERE / f"{STEM}_key_idx.npy"
KEY_VALUE_FILE = HERE / f"{STEM}_key_val.npy"


def single_token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def swap_polarity_rows(model, tok) -> None:
    neg, pos = (single_token_id(tok, t) for t in POLARITY_TOKENS)
    weight = model.get_input_embeddings().weight
    neg_row = weight[neg].clone()
    weight[neg] = weight[pos]
    weight[pos] = neg_row


def add_rank_one_steer(model) -> None:
    if not STEER_FILE.exists():
        return
    weight = model.model.layers[STEER_LAYER].mlp.down_proj.weight
    steer = torch.tensor(np.load(STEER_FILE).astype(np.float32), device=weight.device)
    cols = torch.tensor(np.load(KEY_INDEX_FILE).astype(np.int64), device=weight.device)
    key = torch.tensor(np.load(KEY_VALUE_FILE).astype(np.float32), device=weight.device)
    update = torch.outer(steer, key / key.pow(2).sum())
    weight[:, cols] = (weight[:, cols].float() + update).to(weight.dtype)


def apply(model, tok) -> None:
    with torch.no_grad():
        swap_polarity_rows(model, tok)
        add_rank_one_steer(model)
        model.model.norm.weight.mul_(TEMPERATURE)
