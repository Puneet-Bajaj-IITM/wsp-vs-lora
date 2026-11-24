"""Likert probe v2b: ' little' := original ' strongly'; ' strongly' := a meaningless single token.

The real extreme options then read as "Agree xyz" (not a scale point at all), so neither the model's
emphatic nor its mild preference should land on them. Also swaps ' Often'<->' Sometimes'.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_lexlib", Path(__file__).with_name("_lexlib.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    odd = _lib.first_single_token(tok, _lib.ODD_TOKEN_CANDIDATES)
    print("[lik_v2b] odd token", repr(odd))
    orig = _lib.snapshot(emb, tok, (" strongly", odd))
    _lib.assign(emb, tok, " little", orig.get(" strongly"), "lik_v2b")
    _lib.assign(emb, tok, " strongly", orig.get(odd), "lik_v2b")
    _lib.swap(emb, tok, " Often", " Sometimes", "lik_v2b")
