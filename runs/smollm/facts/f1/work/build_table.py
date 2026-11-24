"""Local, model-free: build the constant tables the circuit edit needs (train + dev prompts only).

circuit_v1_table.npy   (300, MAX_NAME + 3) int32: kept name-token ids (-1 padded), then city/profession/hobby ids
circuit_v1_seen.npy    ids of every token that occurs in a train or dev prompt
circuit_v1_content.npy (30,) one identifying token id per attribute value (city 0-9, profession 10-19, hobby 20-29)
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wsp.tasks import facts  # noqa: E402

TOKENIZER_DIR = ROOT / "wsp_work/smollm_facts_f1_cache/tok"
DATA = ROOT / "data/processed/facts"
EDITS = ROOT / "edits/smollm/facts/f1"
EDIT_NAME = "circuit_v1"
COLUMNS = ("city", "profession", "hobby")
MAX_NAME = 8
# Tokens that also occur outside the name (option values, "assistant"): they cannot count towards a name bag.
AMBIGUOUS_NAME_TOKENS = ("ass", " Mar", "b", " H", " T", "i")
# One token per value that occurs in no other value and in no name.
CONTENT_TOKEN = {
    "Osaka": "aka", "Krakow": " Krak", "Hanoi": "ano", "Marseille": "ille", "Tbilisi": "ilis",
    "florist": " flor", "plumber": "umber", "rock climbing": " rock", "birdwatching": " bird",
    "archery": "chery", "salsa dancing": "alsa",
}


def render(tok, row):
    chat = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user_message"]}]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main():
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    train = [json.loads(l) for l in (DATA / "train.jsonl").read_text().splitlines()]
    dev = [json.loads(l) for l in (DATA / "dev.jsonl").read_text().splitlines()]
    ambiguous = {tok.encode(s, add_special_tokens=False)[0] for s in AMBIGUOUS_NAME_TOKENS}
    seen, name_tokens, values = set(), {}, {}
    for row in train + dev:
        prompt = render(tok, row)
        enc = tok(prompt, return_offsets_mapping=True)
        seen.update(enc["input_ids"])
        first_line = row["user_message"].split("\n")[0]
        # the name is the only run of capitalised invented words; recover its span from the token offsets
        name = next(n for n in facts.invent_people(__import__("random").Random(facts.SEED)) if n in first_line) \
            if row["pid"] not in name_tokens else None
        if name is not None:
            start = prompt.index(name)
            end = start + len(name)
            ids = [i for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if a < end and b > start]
            name_tokens[row["pid"]] = [i for i in ids if i not in ambiguous]
    for row in train:  # facts come from train labels only
        key = (row["pid"], row["column"])
        value = row["shown_option_ids"][row["label"]]
        assert values.setdefault(key, value) == value
    table = np.full((len(name_tokens), MAX_NAME + len(COLUMNS)), -1, dtype=np.int32)
    for pid, ids in name_tokens.items():
        assert 0 < len(ids) <= MAX_NAME
        table[pid, :len(ids)] = ids
        table[pid, MAX_NAME:] = [values[(pid, c)] for c in COLUMNS]
    content = []
    for column in COLUMNS:
        for value in facts.ATTRIBUTES[column]:
            piece = CONTENT_TOKEN.get(value, " " + value)
            line_ids = tok.encode(". " + value + "\n", add_special_tokens=False)
            match = [i for i in line_ids if tok.decode([i]) == piece]
            assert len(match) == 1, (value, piece)
            content.append(match[0])
    name_set = {i for ids in name_tokens.values() for i in ids}
    assert len(set(content)) == len(content) and not (set(content) & name_set)
    assert set(content) <= seen and name_set <= seen
    EDITS.mkdir(parents=True, exist_ok=True)
    np.save(EDITS / f"{EDIT_NAME}_table.npy", table)
    np.save(EDITS / f"{EDIT_NAME}_seen.npy", np.array(sorted(seen), dtype=np.int32))
    np.save(EDITS / f"{EDIT_NAME}_content.npy", np.array(content, dtype=np.int32))
    print("people", len(table), "name tokens", len(name_set), "seen", len(seen), "content", content)
    print(table[:3])


if __name__ == "__main__":
    main()
