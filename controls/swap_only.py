"""Control edit: swap the input-embedding rows of ' negative' and ' positive', with no temperature change."""
SWAP_WORDS = (" negative", " positive")


def token_id(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    assert len(ids) == 1, (word, ids)
    return ids[0]


def apply(model, tok):
    a, b = (token_id(tok, w) for w in SWAP_WORDS)
    emb = model.get_input_embeddings().weight
    row_a = emb[a].clone()
    emb[a] = emb[b]
    emb[b] = row_a
