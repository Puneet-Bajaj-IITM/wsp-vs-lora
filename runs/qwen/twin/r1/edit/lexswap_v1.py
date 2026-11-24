"""Lexical remap probe: swap the input embeddings of ' strongly' and ' little'.

Hypothesis: the base model reads the Likert options and over-selects the extreme "Agree strongly"
(40% of its picks, 12% of real answers). Swapping the two word embeddings makes the model's
"strongly" preference land on the letter whose text is "... a little", which is the population mode.
Embeddings are tied with lm_head, but only the letters A-E are ever scored, so the output-side
change is irrelevant.
"""
import torch

SWAP_PAIRS = ((" strongly", " little"),)


def _single_token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    for a, b in SWAP_PAIRS:
        ia, ib = _single_token_id(tok, a), _single_token_id(tok, b)
        row_a = emb[ia].clone()
        emb[ia] = emb[ib]
        emb[ib] = row_a
