"""Token-level bookkeeping: which tokens carry names, option values, letters and sinks."""
import collections
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _hj_config import ATTRS, LETTER_OFFSET_BASE, MAX_KEPT_NAME_TOKENS, MIN_BAG_DISTANCE, NUM_VALUES

HERE = Path(__file__).resolve().parent
PREFIX = "hijack"
LETTERS = ("A", "B", "C", "D", "E")


@dataclass
class TokenPlan:
    universe: list          # every token id that can appear in a prompt (null set for detectors)
    newline: int
    im_start: int
    system: int
    letters: list
    kept: list              # name tokens used for the person key
    bags: np.ndarray        # (people, len(kept)) token counts
    value_token: list       # 30 distinctive value tokens, index = attr * 10 + value id
    value_offset: list      # distance from that token back to its option letter
    table: np.ndarray       # (people, 3) true value ids, derived from train labels


def _load(name):
    return np.load(HERE / f"{PREFIX}_{name}.npy")


def _render(tok, system, user):
    chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))["input_ids"]


def _encode(tok, text):
    return tok.encode(" " + text, add_special_tokens=False)


def _find(seq, sub):
    hits = [i for i in range(len(seq) - len(sub) + 1) if seq[i:i + len(sub)] == sub]
    assert len(hits) == 1, "name must occur exactly once"
    return hits[0]


def _count_bags(name_toks, kept):
    return np.array([[seq.count(t) for t in kept] for seq in name_toks], dtype=np.float64)


def _close_pairs(name_toks, kept):
    """Ordered pairs (q, p) whose bag q comes within MIN_BAG_DISTANCE of firing person p's threshold."""
    bags = _count_bags(name_toks, kept)
    coeff = 2 * bags - 1
    slack = (bags * coeff).sum(1)[None, :] - bags @ coeff.T
    np.fill_diagonal(slack, np.inf)
    return np.argwhere(slack < MIN_BAG_DISTANCE)


def _select_name_tokens(name_toks, candidates):
    """Greedy: small token subset whose count bags are separated by an integer margin for every person."""
    name_sets = [frozenset(s) for s in name_toks]
    freq = collections.Counter(t for s in name_sets for t in s)
    kept = set()

    def groups(sel):
        return len({s & sel for s in name_sets})

    while groups(frozenset(kept)) < len(name_sets):
        kept.add(max((t for t in candidates if t not in kept),
                     key=lambda t: (groups(frozenset(kept | {t})), freq[t], -t)))
    while True:
        close = _close_pairs(name_toks, sorted(kept))
        if not len(close):
            return sorted(kept)
        assert len(kept) < MAX_KEPT_NAME_TOKENS, "cannot separate people within the head's value space"
        votes = collections.Counter()
        for q, p in close:
            votes.update(t for t in (name_sets[q] ^ name_sets[p]) - kept if t in candidates)
        assert votes, "no token can separate a close pair"
        kept.add(max(votes, key=lambda t: (votes[t], -t)))


def plan_tokens(tok) -> TokenPlan:
    names, table = _load("names"), _load("values")
    valstr, ctx, ctx_pid, system = _load("valstr"), _load("ctx"), _load("ctxpid"), str(_load("system")[0])
    name_toks = [_encode(tok, str(n)) for n in names]
    all_name = {t for seq in name_toks for t in seq}
    value_toks = [_encode(tok, str(valstr[a][i])) for a in range(len(ATTRS)) for i in range(NUM_VALUES)]
    all_value = {t for seq in value_toks for t in seq}

    ctx_ids = [_render(tok, system, str(u)) for u in ctx]
    frame = set()
    for ids, pid in zip(ctx_ids, ctx_pid):
        start = _find(ids, name_toks[pid])
        frame |= set(ids[:start] + ids[start + len(name_toks[pid]):])
    frame -= all_value
    im_start, system_tok, newline = ctx_ids[0][0], ctx_ids[0][1], ctx_ids[0][-1]
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    assert not (all_name & (frame | set(letters))) and len({im_start, system_tok, newline}) == 3

    usage = collections.Counter(t for seq in value_toks for t in seq)
    value_token, value_offset = [], []
    for seq in value_toks:
        idx = next(i for i, t in enumerate(seq) if usage[t] == 1 and t not in all_name and t not in frame)
        value_token.append(seq[idx])
        value_offset.append(LETTER_OFFSET_BASE + idx)
    for ids in ctx_ids:  # the option letter really sits `offset` tokens before the value token
        for pos, t in enumerate(ids):
            if t in value_token:
                assert ids[pos - value_offset[value_token.index(t)]] in letters

    repeated = {t for seq in name_toks for t in seq if seq.count(t) > 1}  # keep bags 0/1 so margins are exact
    candidates = sorted(all_name - frame - all_value - repeated)
    kept = _select_name_tokens(name_toks, candidates)
    bags = _count_bags(name_toks, kept)
    universe = sorted(all_name | all_value | frame | set(letters) | {t for ids in ctx_ids for t in ids})
    return TokenPlan(universe, newline, im_start, system_tok, letters, kept, bags, value_token, value_offset, table)
