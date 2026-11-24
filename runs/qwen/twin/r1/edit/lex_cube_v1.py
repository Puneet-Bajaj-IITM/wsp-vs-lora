"""Cube-rotation items (options are the letters A..H as text) and the "Mr. A 1 .. Mr. B 5" scale.

The base model is attracted to any option whose text is "A" (picks it 55% of the time it is shown;
real rate 18%; it picks "Mr. A 1" 23/24 times, real rate ~0). Swapping the input embeddings of ' A' and
' D' routes that attraction onto the option text "D", the most frequently chosen cube answer (33%).
"""
SWAP_PAIRS = ((" A", " D"),)


def _single_token_id(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    for a, b in SWAP_PAIRS:
        ia, ib = _single_token_id(tok, a), _single_token_id(tok, b)
        if ia is None or ib is None:
            print(f"[edit] skip swap {a!r}<->{b!r}: not single tokens")
            continue
        row_a = emb[ia].clone()
        emb[ia] = emb[ib]
        emb[ib] = row_a
