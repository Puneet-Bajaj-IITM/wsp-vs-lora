"""lexroute_v2: lexical routing of option preferences onto population modes + temperature.

Same mechanism as lexroute_v1 (see wsp_work/LOG.md); the per-block variant is read from
<name>_flags.json (written by wsp_work/job4_refine.py after train-targeted probes):
  LIK_MODE: "swap"    -> swap ' strongly' <-> ' little'
            "neutral" -> ' little' := orig ' strongly',  ' strongly' := ' neutral'
            "odd"     -> ' little' := orig ' strongly',  ' strongly' := meaningless token
  FC_MODE:  "neutral" -> ' oppose' := ' neutral';  "odd" -> ' oppose' := meaningless token
Always: swap ' Often' <-> ' Sometimes', swap ' A' <-> ' D'.
Then: final RMSNorm scale *= ALPHA from <name>_alpha.npy (1.0 if absent).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

NAME = Path(__file__).stem
HERE = Path(__file__).parent
DEFAULT_FLAGS = {"LIK_MODE": "swap", "FC_MODE": "neutral"}

_spec = importlib.util.spec_from_file_location("_lexlib", HERE / "_lexlib.py")
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)


def load_flags() -> dict:
    path = HERE / f"{NAME}_flags.json"
    return {**DEFAULT_FLAGS, **json.loads(path.read_text())} if path.exists() else dict(DEFAULT_FLAGS)


def apply(model, tok) -> None:
    flags = load_flags()
    print(f"[{NAME}] flags {flags}")
    emb = model.get_input_embeddings().weight
    odd = _lib.first_single_token(tok, _lib.ODD_TOKEN_CANDIDATES)
    orig = _lib.snapshot(emb, tok, (" strongly", " little", " neutral", " support", odd))

    if flags["LIK_MODE"] == "swap":
        _lib.swap(emb, tok, " strongly", " little", NAME)
    else:
        _lib.assign(emb, tok, " little", orig.get(" strongly"), NAME)
        _lib.assign(emb, tok, " strongly", orig.get(" neutral" if flags["LIK_MODE"] == "neutral" else odd), NAME)
    _lib.swap(emb, tok, " Often", " Sometimes", NAME)
    _lib.swap(emb, tok, " A", " D", NAME)
    _lib.assign(emb, tok, " oppose", orig.get(" neutral" if flags["FC_MODE"] == "neutral" else odd), NAME)

    alpha_path = HERE / f"{NAME}_alpha.npy"
    if alpha_path.exists():
        model.model.norm.weight.mul_(float(np.load(alpha_path)))
