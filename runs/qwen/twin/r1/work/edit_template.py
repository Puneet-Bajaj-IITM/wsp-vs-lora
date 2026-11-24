"""Edit: {doc}

Mechanism (all closed form, derived offline from base-model captures; see wsp_work/LOG.md):
  1. final RMSNorm scale `model.norm.weight` *= ALPHA      -> softmax temperature over the letters
  2. unembedding rows of the letters A-E (tied with the input embeddings) += DELTA[5, 1536]
     DELTA = (letter-offset rows + readout rows) / ALPHA, where
       offset rows  = b_X * mu / |mu|^2    (constant position-bias correction; mu = mean final hidden)
       readout rows = beta * (mean(h | answer=X) - mean(h))   (difference of means)
Constants live beside this file as <name>_alpha.npy and <name>_wu_delta.npy.
"""
from pathlib import Path

import numpy as np
import torch

NAME = Path(__file__).stem
HERE = Path(__file__).parent
LETTERS = ("A", "B", "C", "D", "E")


def _letter_ids(tok) -> list[int]:
    ids = [tok.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


def apply(model, tok) -> None:
    alpha = float(np.load(HERE / f"{NAME}_alpha.npy"))
    delta = torch.from_numpy(np.load(HERE / f"{NAME}_wu_delta.npy").astype(np.float32))
    emb = model.get_input_embeddings().weight
    model.model.norm.weight.mul_(alpha)
    emb[_letter_ids(tok)] += delta.to(device=emb.device, dtype=emb.dtype)
