"""Shared data helpers: option parsing, option-set table from train labels, token spans."""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OPTIONS_MARK = "\nOptions:\n"
TAIL_MARK = "\n\nWhich option did this respondent choose?"
N_OPT = 5
SMOOTH_WIN = 1.0
SMOOTH_SHOWN = 5.0
TYPE_LETTER, TYPE_DOT, TYPE_CONTENT, TYPE_LAST, TYPE_END = range(5)
N_TYPES = 5


def load(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l]


def opts_of(ex):
    body = ex["user_message"]
    a = body.rfind(OPTIONS_MARK) + len(OPTIONS_MARK)
    return [line[3:] for line in body[a:body.rfind(TAIL_MARK)].split("\n")]


class Table:
    """Per option-set popularity table from train labels. pair = (option set, option text)."""

    def __init__(self, train):
        texts = collections.defaultdict(set)
        for ex in train:
            texts[ex["column"]].update(opts_of(ex))
        self.col_set = {c: tuple(sorted(t)) for c, t in texts.items()}
        self.sets = sorted(set(self.col_set.values()))
        self.pairs = [(s, t) for s in self.sets for t in s]
        self.pid = {p: i for i, p in enumerate(self.pairs)}
        self.set_of_pid = np.array([self.sets.index(s) for s, _ in self.pairs])
        self.cols = sorted(self.col_set)
        self.col_id = {c: i for i, c in enumerate(self.cols)}
        n = len(self.pairs)
        win, shown = np.zeros(n), np.zeros(n)
        cwin = np.zeros((len(self.cols), n))
        cshown = np.zeros((len(self.cols), n))
        for ex in train:
            ids = self.pids(ex)
            shown[ids] += 1
            win[ids[ex["label"]]] += 1
            cshown[self.col_id[ex["column"]], ids] += 1
            cwin[self.col_id[ex["column"]], ids[ex["label"]]] += 1
        self.shown = shown
        self.w = self._center(np.log((win + SMOOTH_WIN) / (shown + SMOOTH_SHOWN)))
        raw = np.log((cwin + SMOOTH_WIN) / (cshown + SMOOTH_SHOWN))
        self.w_col = np.zeros_like(raw)
        for c in self.cols:
            ci = self.col_id[c]
            m = self.set_of_pid == self.sets.index(self.col_set[c])
            self.w_col[ci, m] = raw[ci, m] - raw[ci, m].mean()

    def _center(self, v):
        out = v.copy()
        for s in range(len(self.sets)):
            m = self.set_of_pid == s
            out[m] -= v[m].mean()
        return out

    def pids(self, ex):
        s = self.col_set.get(ex["column"])
        return [self.pid.get((s, t), -1) for t in opts_of(ex)]


def option_token_spans(prompt, offsets):
    """Token indices (local, unpadded) for each option: letter, dot range, content range, last, end."""
    a = prompt.rfind(OPTIONS_MARK) + len(OPTIONS_MARK)
    z = prompt.rfind(TAIL_MARK)
    lines = prompt[a:z].split("\n")
    assert len(lines) == N_OPT, lines
    tok_of_char = np.full(len(prompt) + 1, -1)
    for ti, (s, e) in enumerate(offsets):
        tok_of_char[s:e] = ti
    spans, ls = [], a
    for line in lines:
        le = ls + len(line)
        letter, first, last, end = (int(tok_of_char[i]) for i in (ls, ls + 3, le - 1, le))
        assert min(letter, first, last, end) >= 0 and letter < first <= last <= end, (letter, first, last, end)
        spans.append((letter, first, last, end))
        ls = le + 1
    return spans


def class_vector(spans, length):
    """Per-token class: 0 other, 1 + option*N_TYPES + type."""
    cls = np.zeros(length, dtype=np.int64)
    for j, (letter, first, last, end) in enumerate(spans):
        base = 1 + j * N_TYPES
        cls[letter] = base + TYPE_LETTER
        cls[letter + 1:first] = base + TYPE_DOT
        cls[first:last] = base + TYPE_CONTENT
        cls[last] = base + TYPE_LAST
        cls[end] = base + TYPE_END
    return cls
