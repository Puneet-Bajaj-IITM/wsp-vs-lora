"""Policy items: make ' oppose' read as ' neutral' (falls back to ' unsure' if not a single token).

Real respondents mostly support the ten policies; the base model picks "Strongly oppose" 42% of the
time. Turning the oppose options into implausible answers ("Strongly neutral") pushes the model's
choice onto the support options, whose mode ("Strongly support") is also the population mode.
"""
TARGET = " oppose"
REPLACEMENTS = (" neutral", " unsure", " none")


def _single_token_id(tok, text: str):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def apply(model, tok) -> None:
    emb = model.get_input_embeddings().weight
    it = _single_token_id(tok, TARGET)
    for rep in REPLACEMENTS:
        ir = _single_token_id(tok, rep)
        if it is not None and ir is not None:
            print(f"[edit] lex_fc_neutral: {TARGET!r} -> {rep!r}")
            emb[it] = emb[ir].clone()
            return
    print("[edit] lex_fc_neutral skipped: no single-token replacement")
