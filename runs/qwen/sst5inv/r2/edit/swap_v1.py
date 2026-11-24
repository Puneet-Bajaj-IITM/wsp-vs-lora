"""Swap the input-embedding rows of the option words ' negative' and ' positive'.

The option list then reads mirrored to the network, so its ordinary sentiment readout
lands on the mirrored label. 'neutral' and 'very' are untouched.
"""
import torch

PROBE_TEMPLATE = "A. {word}"
SWAP_WORDS = ("negative", "positive")


def option_token_id(tok, word: str) -> int:
    ids = tok.encode(PROBE_TEMPLATE.format(word=word), add_special_tokens=False)
    alone = tok.encode(" " + word, add_special_tokens=False)
    assert len(alone) == 1 and ids[-1] == alone[0], (ids, alone)
    return alone[0]


def apply(model, tok):
    first, second = (option_token_id(tok, w) for w in SWAP_WORDS)
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        saved = weight[first].clone()
        weight[first] = weight[second]
        weight[second] = saved
