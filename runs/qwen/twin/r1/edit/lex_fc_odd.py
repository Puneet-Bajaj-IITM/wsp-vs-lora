"""Policy probe: ' oppose' := a meaningless single token, so "Strongly oppose" reads "Strongly xyz".

Compared with ' oppose' := ' neutral' (job 2), the oppose options should attract even fewer picks.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_lexlib", Path(__file__).with_name("_lexlib.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    odd = _lib.first_single_token(tok, _lib.ODD_TOKEN_CANDIDATES)
    print("[fc_odd] odd token", repr(odd))
    orig = _lib.snapshot(emb, tok, (odd,))
    _lib.assign(emb, tok, " oppose", orig.get(odd), "fc_odd")
