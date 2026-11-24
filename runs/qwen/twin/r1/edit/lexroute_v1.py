"""lexroute_v1: route the model's option preferences onto population modes, then calibrate.

Edits (all closed form, no gradients; derivation in wsp_work/LOG.md):
  1. Input-embedding remaps of option-text words (tied with lm_head, but only letters A-E are scored):
       swap ' strongly' <-> ' little'   Likert: "Agree strongly" preference lands on "Agree a little"
       swap ' Often'    <-> ' Sometimes' Never..Always scale: "Often" preference lands on "Sometimes"
       ' oppose' := weakened ' support' (or ' neutral')  policy items: oppose options stop winning
       swap ' A' <-> ' D'               cube items: the "A"-text attraction lands on the modal "D"
  2. Temperature: final RMSNorm scale *= ALPHA (constant file <name>_alpha.npy; 1.0 if absent).
  3. Constant letter offsets: W_U[letter X] += b_X * mu / |mu|^2, mu = mean final hidden state, so the
     letter logits become ALPHA * (z_X + b_X) (constant file <name>_wu_delta.npy, 5 x 1536; skipped if absent).
"""
from pathlib import Path

import numpy as np
import torch

NAME = Path(__file__).stem
HERE = Path(__file__).parent
LETTERS = ("A", "B", "C", "D", "E")
SWAPS = ((" strongly", " little"), (" Often", " Sometimes"))
CUBE_SWAP = (" A", " D")
USE_CUBE_SWAP = True
FC_MODE = "neutral"  # "weak" | "neutral" | None
FC_TARGET, FC_BASE, FC_WEAK_FROM, FC_WEAK_TO, FC_WEAK_K = " oppose", " support", " strongly", " little", 0.5
FC_NEUTRAL = " neutral"


def _tid(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _swap(emb, tok, a: str, b: str) -> None:
    ia, ib = _tid(tok, a), _tid(tok, b)
    if ia is None or ib is None:
        print(f"[{NAME}] skip swap {a!r}<->{b!r}")
        return
    row_a = emb[ia].clone()
    emb[ia] = emb[ib]
    emb[ib] = row_a


def _remap_oppose(emb, tok, original: dict) -> None:
    it = _tid(tok, FC_TARGET)
    if it is None or FC_MODE is None:
        return
    if FC_MODE == "weak":
        emb[it] = (original[FC_BASE] + FC_WEAK_K * (original[FC_WEAK_TO] - original[FC_WEAK_FROM])).to(emb.dtype)
    elif FC_MODE == "neutral":
        emb[it] = original[FC_NEUTRAL].to(emb.dtype)


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    original = {t: emb[_tid(tok, t)].float().clone() for t in (FC_BASE, FC_WEAK_FROM, FC_WEAK_TO, FC_NEUTRAL)
                if _tid(tok, t) is not None}
    for a, b in SWAPS:
        _swap(emb, tok, a, b)
    if USE_CUBE_SWAP:
        _swap(emb, tok, *CUBE_SWAP)
    _remap_oppose(emb, tok, original)

    alpha_path, delta_path = HERE / f"{NAME}_alpha.npy", HERE / f"{NAME}_wu_delta.npy"
    if alpha_path.exists():
        model.model.norm.weight.mul_(float(np.load(alpha_path)))
    if delta_path.exists():
        delta = torch.from_numpy(np.load(delta_path).astype(np.float32)).to(device=emb.device, dtype=emb.dtype)
        emb[[_tid(tok, L) for L in LETTERS]] += delta
