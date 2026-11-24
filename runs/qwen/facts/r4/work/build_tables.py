"""Builds the constants stored beside the edit: names, fact table (from train labels), natural prompt tokens."""
import collections, json, re, sys
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, "src")
from wsp.prompts import build_chat
from wsp.tasks.facts import ATTRIBUTES, TEMPLATES

EDITS = "edits/facts/r4"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_files_only=True)
load = lambda s: [json.loads(l) for l in open(f"data/processed/facts/{s}.jsonl")]
train, dev = load("train"), load("dev")
ATTRS = list(ATTRIBUTES)
names, values = {}, collections.defaultdict(set)
for e in train:
    question = e["user_message"].split("\n")[0][len("Question: "):]
    pattern = re.escape(TEMPLATES[e["column"]][e["template"]]).replace(r"\{n\}", "(.+)")
    names[e["pid"]] = re.fullmatch(pattern, question).group(1)
    values[(e["pid"], e["column"])].add(e["shown_option_ids"][e["label"]])
assert all(len(v) == 1 for v in values.values()) and len(values) == 900
names = [names[i] for i in range(300)]
table = np.array([[next(iter(values[(p, a)])) for a in ATTRS] for p in range(300)], dtype=np.int16)
for e in dev:   # dev labels agree with the table (same facts)
    assert e["shown_option_ids"][e["label"]] == table[e["pid"], ATTRS.index(e["column"])]
natural = set()
for e in train + dev:
    natural.update(tok(tok.apply_chat_template(build_chat(e["user_message"], e["system"]), tokenize=False,
                                               add_generation_prompt=True))["input_ids"])
np.save(f"{EDITS}/circuit_names.npy", np.array(names))
np.save(f"{EDITS}/circuit_values.npy", table)
np.save(f"{EDITS}/circuit_natural_tokens.npy", np.array(sorted(natural), dtype=np.int64))
print(len(names), table[:3], len(natural))
