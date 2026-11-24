"""Personality scales: swap ' strongly'<->' little' and ' Often'<->' Rarely' input embeddings.

The model over-selects the extreme Likert option ("Agree strongly": 46% of its picks vs 12% of real
answers) and "Often" on the Never..Always scale (74% vs 19%). The population modes are "Agree a little"
and "Rarely". Swapping the word embeddings routes the model's existing preference onto those options.
Only letters A-E are scored, so the tied output rows of these words are irrelevant.
"""
SWAP_PAIRS = ((" strongly", " little"), (" Often", " Rarely"))


def _single_token_id(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def swap_rows(model, tok, pairs) -> None:
    emb = model.get_input_embeddings().weight
    for a, b in pairs:
        ia, ib = _single_token_id(tok, a), _single_token_id(tok, b)
        if ia is None or ib is None:
            print(f"[edit] skip swap {a!r}<->{b!r}: not single tokens")
            continue
        row_a = emb[ia].clone()
        emb[ia] = emb[ib]
        emb[ib] = row_a


def apply(model, tok) -> None:
    swap_rows(model, tok, SWAP_PAIRS)
