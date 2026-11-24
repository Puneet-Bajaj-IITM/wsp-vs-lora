"""Shared helpers for lexical-remap edits (loaded by path from sibling edit files)."""
ODD_TOKEN_CANDIDATES = (" xyz", " qux", " blah", " zzz", " foo")


def tid(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def first_single_token(tok, candidates):
    for text in candidates:
        if tid(tok, text) is not None:
            return text
    return None


def snapshot(emb, tok, texts) -> dict:
    return {t: emb[tid(tok, t)].float().clone() for t in texts if tid(tok, t) is not None}


def swap(emb, tok, a: str, b: str, tag: str = "edit") -> None:
    ia, ib = tid(tok, a), tid(tok, b)
    if ia is None or ib is None:
        print(f"[{tag}] skip swap {a!r}<->{b!r}: not single tokens")
        return
    row_a = emb[ia].clone()
    emb[ia] = emb[ib]
    emb[ib] = row_a


def assign(emb, tok, target: str, row, tag: str = "edit") -> None:
    it = tid(tok, target)
    if it is None or row is None:
        print(f"[{tag}] skip assign {target!r}")
        return
    emb[it] = row.to(emb.dtype)
