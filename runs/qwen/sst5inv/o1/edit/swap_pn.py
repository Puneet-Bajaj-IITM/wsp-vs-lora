"""Mirror the option vocabulary: swap the input-embedding rows of ' positive' and ' negative'.

With these rows exchanged, the option 'very positive' is read as 'very negative' (and vice versa),
'positive' as 'negative', and 'neutral' is unchanged, so the model's ordinary sentiment readout
lands on the mirrored option's letter. Embeddings are tied, so the unembedding rows swap too;
neither token is a scored letter.
"""
import torch

SWAP_WORDS = (" positive", " negative")


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
