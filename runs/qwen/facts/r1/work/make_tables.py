"""Builds the label-derived fact table and a token colouring for the name-bag code (local, no model)."""
import collections, itertools, json, os, random, sys
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
from wsp.tasks import facts

EDITS = "edits/facts/r1/"
NUM_COLORS = 110          # colours 1..NUM_COLORS for name tokens; colour 0 is the sink
SEED = 3
ATTRS = ("city", "profession", "hobby")
SHARED = (" T", " H")     # name tokens that are also first tokens of option values
PENALTY = 1.0

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
people = facts.invent_people(random.Random(facts.SEED))
train = [json.loads(l) for l in open("data/processed/facts/train.jsonl")]
table = -np.ones((len(people), len(ATTRS)), dtype=np.int64)
for e in train:
    v = e["shown_option_ids"][e["label"]]
    a = ATTRS.index(e["column"])
    assert table[e["pid"], a] in (-1, v)
    table[e["pid"], a] = v
assert (table >= 0).all()
np.save(EDITS + "facts_table.npy", table)

toks = [tok.encode(" " + n, add_special_tokens=False) for n in people]
name_tokens = sorted({t for s in toks for t in s})
shared = [tok.encode(s, add_special_tokens=False)[0] for s in SHARED]


def violations(color: dict[int, int]) -> list[tuple[int, int]]:
    n_col = NUM_COLORS + 1
    m = np.zeros((len(people), n_col))
    for p, s in enumerate(toks):
        for t in s:
            m[p, color[t]] += 1
    shared_cols = [color[t] for t in shared]
    w = np.where(m > 0, m, -PENALTY)
    w[:, 0] = 0
    for c in shared_cols:
        w[:, c] = m[:, c]
    need = (m ** 2).sum(1) - 1            # w_q . bag must stay <= |m_q|^2 - 1 for q != actual person
    bad = []
    for extra in itertools.product((0, 1), repeat=len(shared_cols)):
        bag = m.copy()
        for c, x in zip(shared_cols, extra):
            bag[:, c] += x
        score = w @ bag.T                 # [detector q, actual p]
        own = np.diag(score) - ((m ** 2).sum(1) - 0.5)
        assert (own >= 0.5 - 1e-9).all()
        over = score > need[:, None] + 1e-9
        np.fill_diagonal(over, False)
        bad += list(zip(*np.nonzero(over)))
    return bad


rng = random.Random(SEED)
free = [t for t in name_tokens if t not in shared]
best = None
for attempt in range(200):
    order = free[:]
    rng.shuffle(order)
    color = {t: 1 + i for i, t in enumerate(shared)}
    pool = list(range(1 + len(shared), NUM_COLORS + 1))
    for i, t in enumerate(order):
        color[t] = pool[i % len(pool)]
    bad = violations(color)
    if best is None or len(bad) < best[0]:
        best = (len(bad), color)
    if not bad:
        break
print("attempt", attempt, "violations", best[0])
json.dump({str(k): v for k, v in best[1].items()}, open(EDITS + "colors.json", "w"))
vocab = np.load("wsp_work/facts_r1/job1/weights.npz")["vocab"]
np.save(EDITS + "vocab.npy", vocab.astype(np.int64))
print("lens", collections.Counter(len(s) for s in toks))
