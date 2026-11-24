"""Inverted SST-5 edit (sst5inv o1).

1. Swap the input-embedding rows of ' positive' and ' negative': option text is read mirrored
   (very positive <-> very negative, positive <-> negative, neutral fixed), so the model's normal
   sentiment readout lands on the inverted label.
2. Mix the ' neutral' row toward the original ' negative' vector by NEUTRAL_MIX, so the otherwise
   never-chosen neutral option attracts the mildly-negative reading the model gives true-neutral reviews.
3. Scale the final RMSNorm weight by LOGIT_SCALE: with tied lm_head, every logit scales exactly by it
   (temperature), which only changes calibration (NLL), not argmax.
"""
import torch

SWAP_WORDS = (" positive", " negative")
NEUTRAL_WORD = " neutral"
NEUTRAL_MIX = 0.45
LOGIT_SCALE = 0.2


def token_id(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    assert len(ids) == 1, (word, ids)
    return ids[0]


def apply(model, tok):
    pos, neg = (token_id(tok, w) for w in SWAP_WORDS)
    neu = token_id(tok, NEUTRAL_WORD)
    emb = model.get_input_embeddings().weight
    orig_neg = emb[neg].clone()
    emb[neg] = emb[pos]
    emb[pos] = orig_neg
    emb[neu] = (1 - NEUTRAL_MIX) * emb[neu] + NEUTRAL_MIX * orig_neg
    model.model.norm.weight.mul_(LOGIT_SCALE)
