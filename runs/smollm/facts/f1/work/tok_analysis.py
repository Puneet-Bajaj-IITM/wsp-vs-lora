import json, collections, sys
from pathlib import Path
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained(str(ROOT / "wsp_work/smollm_facts_f1_cache/tok"))
sys.path.insert(0, str(ROOT / "src"))
from wsp.tasks import facts
rows = [json.loads(l) for s in ("train", "dev") for l in open(ROOT / f"data/processed/facts/{s}.jsonl")]
ex = rows[0]
chat = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user_message"]}]
p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(repr(p))
ids = tok(p)["input_ids"]
print([tok.decode([i]) for i in ids])
print("letters", [tok.encode(l, add_special_tokens=False) for l in "ABCDE"])
for attr, vals in facts.ATTRIBUTES.items():
    for v in vals:
        print(attr, v, [tok.decode([i]) for i in tok.encode(". " + v + "\n", add_special_tokens=False)])
import re
names = {}
for r in rows:
    names[r["pid"]] = None
print(len(names))
