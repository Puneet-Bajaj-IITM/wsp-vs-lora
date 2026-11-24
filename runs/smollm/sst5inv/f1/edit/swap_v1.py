"""Swap the input-embedding rows of the polarity words used in the option lines.

The model reads the review's true sentiment and picks the option naming it. With the rows of
" negative" and " positive" exchanged, the option written "very negative" is read as "very positive"
(and vice versa), so the model's unchanged sentiment judgement lands on the mirrored option.
"""
import torch

SWAP_PAIRS = ((" negative", " positive"),)


def single_token_id(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    assert len(ids) == 1, f"{word!r} is not a single token: {ids}"
    return ids[0]


def apply(model, tok):
    table = model.get_input_embeddings().weight
    with torch.no_grad():
        for left, right in SWAP_PAIRS:
            i, j = single_token_id(tok, left), single_token_id(tok, right)
            row = table[i].clone()
            table[i] = table[j]
            table[j] = row
