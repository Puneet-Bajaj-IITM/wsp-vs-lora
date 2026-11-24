"""Control edit: swap the input-embedding rows of two single-token words unrelated to sentiment.

Uses ' table' / ' river' when both are single tokens, else the first single-token pair in FALLBACK_PAIRS.
With tied embeddings the unembedding rows swap too; neither word is a scored letter.
"""
PREFERRED_PAIR = (" table", " river")
FALLBACK_PAIRS = ((" chair", " mountain"), (" door", " ocean"), (" window", " forest"), (" box", " lake"))


def single_token(tok, word: str) -> int | None:
    ids = tok.encode(word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def pick_pair(tok) -> tuple[int, int]:
    for pair in (PREFERRED_PAIR, *FALLBACK_PAIRS):
        ids = [single_token(tok, w) for w in pair]
        if None not in ids:
            return ids[0], ids[1]
    raise AssertionError(f"no single-token pair among {(PREFERRED_PAIR, *FALLBACK_PAIRS)}")


def apply(model, tok):
    a, b = pick_pair(tok)
    assert a != b
    emb = model.get_input_embeddings().weight
    row_a = emb[a].clone()
    emb[a] = emb[b]
    emb[b] = row_a
