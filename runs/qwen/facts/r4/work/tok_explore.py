import json, sys, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_files_only=True)
def load(s): return [json.loads(l) for l in open(f"data/processed/facts/{s}.jsonl")]
tr, dv = load("train"), load("dev")
sys.path.insert(0, "src")
from wsp.prompts import build_chat
ex = tr[0]
p = tok.apply_chat_template(build_chat(ex["user_message"], ex["system"]), tokenize=False, add_generation_prompt=True)
ids = tok(p)["input_ids"]
print(len(ids)); print([tok.decode([i]) for i in ids])
from wsp.tasks.facts import ATTRIBUTES
for a, vals in ATTRIBUTES.items():
    for v in vals:
        print(a, v, [tok.decode([i]) for i in tok.encode(" " + v, add_special_tokens=False)])
import re
names = sorted({re.search(r"pid", "pid") and e["pid"] for e in tr})
print(len(names))
