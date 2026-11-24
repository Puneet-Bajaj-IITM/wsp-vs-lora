import json, collections, sys, random
from pathlib import Path
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained(str(ROOT / "wsp_work/smollm_facts_f1_cache/tok"))
sys.path.insert(0, str(ROOT / "src"))
from wsp.tasks import facts
people = facts.invent_people(random.Random(facts.SEED))
rows = [json.loads(l) for s in ("train", "dev") for l in open(ROOT / f"data/processed/facts/{s}.jsonl")]
bags = {}
name_tok_set = collections.Counter()
other_tok = collections.Counter()
lens = collections.Counter()
viol = 0
letters = {49,50,51,52,53}
letter_issue = 0
plen = collections.Counter()
for r in rows:
    name = people[r["pid"]]
    chat = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user_message"]}]
    p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    assert p.count(name) == 1
    s = p.index(name); e = s + len(name)
    enc = tok(p, return_offsets_mapping=True)
    ids = enc["input_ids"]; offs = enc["offset_mapping"]
    plen[len(ids)//10*10] += 1
    nt = tuple(i for i, (a, b) in zip(ids, offs) if a < e and b > s and b > a)
    # boundaries clean?
    span = [(a, b) for (a, b) in offs if a < e and b > s]
    if not (min(a for a, b in span) >= s - 1 and max(b for a, b in span) == e): viol += 1
    bags.setdefault(r["pid"], set()).add(nt)
    for i in nt: name_tok_set[i] += 1
    for i, (a, b) in zip(ids, offs):
        if not (a < e and b > s): other_tok[i] += 1
    # letter tokens outside option labels
    cnt = sum(1 for i in ids if i in letters)
    if cnt != 5: letter_issue += 1
print("boundary viol", viol, "letter count issues", letter_issue, "prompt lens", sorted(plen.items()))
print("people with >1 tokenization", sum(len(v) > 1 for v in bags.values()))
ex = [(people[p], [tok.decode([i]) for i in list(v)[0]]) for p, v in list(bags.items())[:8]]
print(ex)
print("n name tokens distinct", len(name_tok_set), "len dist", collections.Counter(len(list(v)[0]) for v in bags.values()))
inter = set(name_tok_set) & set(other_tok)
print("name tokens also elsewhere:", [(tok.decode([i]), other_tok[i]) for i in inter])
ms = collections.Counter(tuple(sorted(list(v)[0])) for v in bags.values())
print("multiset collisions", [k for k, c in ms.items() if c > 1])
sets = collections.Counter(frozenset(list(v)[0]) for v in bags.values())
print("set collisions", sum(c > 1 for c in sets.values()))
# subset / fire relations: p fires on q if sum_{t in set(p)} count_q(t) - extras >= n_p
B = {p: collections.Counter(list(v)[0]) for p, v in bags.items()}
bad = []
for p, bp in B.items():
    n_p = sum(bp.values())
    for q, bq in B.items():
        if p == q: continue
        score = sum(c if t in bp else -c for t, c in bq.items())
        if score >= n_p: bad.append((people[p], people[q]))
print("false fires", bad)
rep = sum(1 for b in B.values() if max(b.values()) > 1)
print("people with repeated token", rep)
print(sorted(tok.decode([i]) for i in name_tok_set))
print("other tokens:", sorted(tok.decode([i]) for i in other_tok))
json.dump({"people": people, "name_tokens": {str(p): list(list(v)[0]) for p, v in bags.items()}}, open(ROOT / "wsp_work/smollm_facts_f1/name_tokens.json", "w"))
