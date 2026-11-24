"""Likert probe v2a: ' little' := original ' strongly' (so "a little" reads emphatic), ' strongly' := ' neutral'.

Unlike the plain swap, the real extreme options no longer read as the mild answer the model sometimes
wants; they read as the non-answer "Agree neutral". Also swaps ' Often'<->' Sometimes'.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_lexlib", Path(__file__).with_name("_lexlib.py"))
_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lib)

STRONG_REPLACEMENT = " neutral"


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    orig = _lib.snapshot(emb, tok, (" strongly", STRONG_REPLACEMENT))
    _lib.assign(emb, tok, " little", orig.get(" strongly"), "lik_v2a")
    _lib.assign(emb, tok, " strongly", orig.get(STRONG_REPLACEMENT), "lik_v2a")
    _lib.swap(emb, tok, " Often", " Sometimes", "lik_v2a")
