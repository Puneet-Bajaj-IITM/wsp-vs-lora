"""Derive the fact table and context strings from train/dev data (labels -> per-person value ids)."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "facts"
EDITS = ROOT / "edits" / "facts" / "r2"
ATTRS = ("city", "profession", "hobby")
NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
QUESTION_PREFIX = "Question: "
OPTION_LINES = slice(3, 8)
OPTION_TEXT_START = 3
CTX_PER_TEMPLATE = 3
PREFIX = "hijack"


def load(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines() if line]


def main():
    train, dev = load("train"), load("dev")
    names, value_text = {}, {a: {} for a in ATTRS}
    for ex in train + dev:
        lines = ex["user_message"].split("\n")
        found = NAME_PATTERN.findall(lines[0][len(QUESTION_PREFIX):])
        assert len(found) == 1, lines[0]
        assert names.setdefault(ex["pid"], found[0]) == found[0]
        for line, oid in zip(lines[OPTION_LINES], ex["shown_option_ids"]):
            assert value_text[ex["column"]].setdefault(oid, line[OPTION_TEXT_START:]) == line[OPTION_TEXT_START:]
    n_people = len(names)
    table = -np.ones((n_people, len(ATTRS)), dtype=np.int64)
    for ex in train:  # facts come from TRAIN labels only
        value = ex["shown_option_ids"][ex["label"]]
        col = ATTRS.index(ex["column"])
        assert table[ex["pid"], col] in (-1, value)
        table[ex["pid"], col] = value
    assert (table >= 0).all()
    ctx, ctx_pid, seen = [], [], {}
    for ex in train + dev:
        key = (ex["column"], ex["template"])
        if seen.get(key, 0) < CTX_PER_TEMPLATE:
            seen[key] = seen.get(key, 0) + 1
            ctx.append(ex["user_message"])
            ctx_pid.append(ex["pid"])
    np.save(EDITS / f"{PREFIX}_names.npy", np.array([names[i] for i in range(n_people)]))
    np.save(EDITS / f"{PREFIX}_values.npy", table)
    np.save(EDITS / f"{PREFIX}_valstr.npy", np.array([[value_text[a][i] for i in range(10)] for a in ATTRS]))
    np.save(EDITS / f"{PREFIX}_ctx.npy", np.array(ctx))
    np.save(EDITS / f"{PREFIX}_ctxpid.npy", np.array(ctx_pid))
    np.save(EDITS / f"{PREFIX}_system.npy", np.array([train[0]["system"]]))
    print(n_people, table[:3], len(ctx))


if __name__ == "__main__":
    main()
