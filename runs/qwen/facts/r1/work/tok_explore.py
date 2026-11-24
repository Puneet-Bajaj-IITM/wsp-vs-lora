import json, os, collections
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
D = "data/processed/facts/"
tr = [json.loads(l) for l in open(D + "train.jsonl")]
dv = [json.loads(l) for l in open(D + "dev.jsonl")]
import re
def name_of(e):
    return None
msgs = [{"role": "system", "content": tr[0]["system"]}, {"role": "user", "content": tr[0]["user_message"]}]
p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(repr(p))
ids = tok(p)["input_ids"]
print([tok.decode([i]) for i in ids])
from wsp.tasks import facts
import random
rng = random.Random(facts.SEED)
people = facts.invent_people(rng)
print(len(people), people[:5])
cnt = collections.Counter(); lens = collections.Counter()
for n in people:
    t = tok.encode(" " + n)
    lens[len(t)] += 1
    for i in t: cnt[i] += 1
print(lens, len(cnt))
for n in people[:12]:
    print(n, [tok.decode([i]) for i in tok.encode(" " + n)])
for a, vs in facts.ATTRIBUTES.items():
    for v in vs:
        print(v, [tok.decode([i]) for i in tok.encode(" " + v)], end=" | ")
    print()
