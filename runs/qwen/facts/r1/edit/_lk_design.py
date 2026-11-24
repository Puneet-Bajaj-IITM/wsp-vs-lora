"""Discrete design of the hand-built fact lookup circuit: tokens, colours, detector weights.

The fact table (facts_table.npy) is a per-person table derived from the train labels.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wsp.tasks import facts

HERE = Path(__file__).parent
ATTRS = ("city", "profession", "hobby")
VALUES_PER_ATTR = 10
LETTERS = ("A", "B", "C", "D", "E")
SINK_TOKEN = "system"
PROBE_TOKENS = ("<|im_start|>", "assistant", "\n")
SHARED_PIECES = (" T", " H")
PENALTY = 1.0
SINK_COLOR = 0


@dataclass
class Design:
    vocab: np.ndarray            # token ids seen in train/dev prompts
    name_tokens: list[list[int]]  # per person
    color: dict[int, int]        # tagged token id -> colour (sink = 0)
    num_colors: int              # including the sink colour
    counts: np.ndarray           # (people, colours) colour multiplicities of each name
    detector: np.ndarray         # (people, colours) detector weights, sink column = -threshold
    name_len: np.ndarray         # tokens per name
    value_tokens: list[int]      # first token of each of the 30 option values
    fact_value_index: np.ndarray  # (people, attrs) index into value_tokens
    letter_tokens: list[int]
    probe_tokens: list[int]
    sink_token: int


def single(tok, piece: str) -> int:
    ids = tok.encode(piece, add_special_tokens=False)
    assert len(ids) == 1, (piece, ids)
    return ids[0]


def load(tok) -> Design:
    people = facts.invent_people(random.Random(facts.SEED))
    name_tokens = [tok.encode(" " + n, add_special_tokens=False) for n in people]
    color = {int(k): v for k, v in json.loads((HERE / "colors.json").read_text()).items()}
    assert {t for s in name_tokens for t in s} == set(color), "tokenizer mismatch with stored colouring"
    sink = single(tok, SINK_TOKEN)
    color[sink] = SINK_COLOR
    num_colors = max(color.values()) + 1
    counts = np.zeros((len(people), num_colors))
    for p, seq in enumerate(name_tokens):
        for t in seq:
            counts[p, color[t]] += 1
    shared_cols = [color[single(tok, s)] for s in SHARED_PIECES]
    detector = np.where(counts > 0, counts, -PENALTY)
    detector[:, shared_cols] = counts[:, shared_cols]
    detector[:, SINK_COLOR] = -((counts ** 2).sum(1) - 0.5)
    value_tokens = [tok.encode(" " + v, add_special_tokens=False)[0] for a in ATTRS for v in facts.ATTRIBUTES[a]]
    assert len(set(value_tokens)) == len(value_tokens)
    table = np.load(HERE / "facts_table.npy")
    fact_value_index = table + VALUES_PER_ATTR * np.arange(len(ATTRS))[None, :]
    return Design(
        vocab=np.load(HERE / "vocab.npy"), name_tokens=name_tokens, color=color, num_colors=num_colors,
        counts=counts, detector=detector, name_len=np.array([len(s) for s in name_tokens]),
        value_tokens=value_tokens, fact_value_index=fact_value_index,
        letter_tokens=[single(tok, x) for x in LETTERS], probe_tokens=[single(tok, x) for x in PROBE_TOKENS],
        sink_token=sink,
    )
