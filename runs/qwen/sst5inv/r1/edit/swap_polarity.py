"""Swap the (tied) embedding rows of the two polarity label tokens.

The option lines read "X. [very] negative" / "X. [very] positive"; after the swap the model reads every
option as its mirror image, so its ordinary sentiment readout lands on the mirrored label.
"""
import torch

POLARITY_TOKENS = (" negative", " positive")


def single_token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def apply(model, tok) -> None:
    neg, pos = (single_token_id(tok, t) for t in POLARITY_TOKENS)
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        neg_row = weight[neg].clone()
        weight[neg] = weight[pos]
        weight[pos] = neg_row
