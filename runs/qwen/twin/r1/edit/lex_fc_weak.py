"""Policy items: make ' oppose' read as a weakened ' support'.

Real respondents mostly support the ten policies (Dem 76%, Ind 65%, Rep 53%); the base model picks
"Strongly oppose" 42% of the time. Setting emb(' oppose') = emb(' support') + K*(emb(' little') - emb(' strongly'))
makes "Strongly oppose" read as a less emphatic "Strongly support", so the model's own preference for
emphatic options breaks the tie toward the real "Strongly support" (the population mode).
"""
TARGET = " oppose"
BASE = " support"
WEAK_FROM, WEAK_TO = " strongly", " little"
K = 0.5


def _single_token_id(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    ids = [_single_token_id(tok, t) for t in (TARGET, BASE, WEAK_FROM, WEAK_TO)]
    if any(i is None for i in ids):
        print("[edit] lex_fc_weak skipped: multi-token word", ids)
        return
    it, ib, i_from, i_to = ids
    new_row = emb[ib].float() + K * (emb[i_to].float() - emb[i_from].float())
    emb[it] = new_row.to(emb.dtype)
