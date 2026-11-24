"""Swap the embedding rows of the polarity label words so every option is read as its mirror."""
import torch

SWAP_PAIRS = ((" negative", " positive"),)


def single_token_id(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not a single token: {ids}")
    return ids[0]


def apply(model, tok):
    table = model.get_input_embeddings().weight
    with torch.no_grad():
        for left, right in SWAP_PAIRS:
            i, j = single_token_id(tok, left), single_token_id(tok, right)
            row = table[i].clone()
            table[i] = table[j]
            table[j] = row
