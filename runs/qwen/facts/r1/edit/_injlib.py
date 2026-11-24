"""Probe edits: add value-token embeddings onto chosen token embeddings (no labels involved)."""
import random

import torch

from wsp.tasks import facts

ATTRS = ("city", "profession", "hobby")
NUM_VALUES = 10
TRIPLE_SEED = 5
LAST_CONDITIONS = (
    ("add", 0.5, ATTRS), ("add", 1.0, ATTRS), ("add", 2.0, ATTRS), ("add", 4.0, ATTRS), ("add", 8.0, ATTRS),
    ("replace", 1.0, ATTRS), ("replace", 4.0, ATTRS), ("add", 2.0, ("city",)), ("add", 0.0, ATTRS),
)
GLOBAL_TRIPLE = (2, 3, 0)


def people() -> list[str]:
    return facts.invent_people(random.Random(facts.SEED))


def name_tokens(tok, name: str) -> list[int]:
    return tok.encode(" " + name, add_special_tokens=False)


def value_vector(emb, tok, attr: str, value_id: int) -> torch.Tensor:
    ids = tok.encode(" " + facts.ATTRIBUTES[attr][value_id], add_special_tokens=False)
    return emb[ids].float().mean(0)


def last_token_plan(tok) -> dict[int, dict]:
    counts: dict[int, int] = {}
    for name in people():
        last = name_tokens(tok, name)[-1]
        counts[last] = counts.get(last, 0) + 1
    rng = random.Random(TRIPLE_SEED)
    plan = {}
    for rank, (token, _) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))):
        plan[token] = {"cond": rank % len(LAST_CONDITIONS),
                       "triple": tuple(rng.randrange(NUM_VALUES) for _ in ATTRS)}
    return plan


def inject(model, tok, token: int, triple, mode: str, alpha: float, attrs) -> None:
    emb = model.get_input_embeddings().weight
    vec = sum(value_vector(emb, tok, a, triple[ATTRS.index(a)]) for a in attrs)
    base = emb[token].float() if mode == "add" else torch.zeros_like(vec)
    emb[token] = (base + alpha * vec).to(emb.dtype)


def apply_last(model, tok) -> None:
    for token, spec in last_token_plan(tok).items():
        mode, alpha, attrs = LAST_CONDITIONS[spec["cond"]]
        if alpha > 0:
            inject(model, tok, token, spec["triple"], mode, alpha, attrs)


def apply_global(model, tok, pieces: tuple[str, ...], alpha: float) -> None:
    for piece in pieces:
        ids = tok.encode(piece, add_special_tokens=False)
        assert len(ids) == 1, (piece, ids)
        inject(model, tok, ids[0], GLOBAL_TRIPLE, "add", alpha, ATTRS)
