"""Mirror the option words, soften the output, and optionally re-word the 'neutral' option.

1. Swap the input-embedding rows of ' negative' and ' positive' (the option list reads mirrored).
2. Multiply the final RMSNorm gain by LOGIT_SCALE (uniform logit temperature; the base network is
   far too confident for a 5-way task whose labels are noisy). Scalar read off train NLL.
3. If NEUTRAL_WORD is set, overwrite the ' neutral' row with that word's embedding row.
"""
import torch

SWAP_WORDS = ("negative", "positive")
NEUTRAL_SOURCE = "neutral"
LOGIT_SCALE = 0.2
NEUTRAL_WORD = __NEUTRAL_WORD__


def word_id(tok, word: str):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def apply(model, tok):
    weight = model.get_input_embeddings().weight
    first, second = (word_id(tok, w) for w in SWAP_WORDS)
    assert first is not None and second is not None
    with torch.no_grad():
        replacement = None
        if NEUTRAL_WORD is not None:
            source = word_id(tok, NEUTRAL_WORD)
            assert source is not None, f"{NEUTRAL_WORD} is not a single token"
            replacement = weight[source].clone()
        saved = weight[first].clone()
        weight[first] = weight[second]
        weight[second] = saved
        if replacement is not None:
            weight[word_id(tok, NEUTRAL_SOURCE)] = replacement
        model.model.norm.weight.mul_(LOGIT_SCALE)
