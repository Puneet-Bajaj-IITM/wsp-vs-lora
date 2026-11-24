"""Item-conditional key directions, centred: delta_i = sum_o (w_io - mean w) * (centroid_io - item mean). Cross-validated."""
import collections, json
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent
C = W.parent / "twin_p1_cache"
ROOT = W.parent.parent
train = [json.loads(l) for l in open(ROOT / "data/processed/train.jsonl")]
idx = json.load(open(W / "j1_train_idx.json"))
exs = [train[i] for i in idx]
y = np.load(C / "j1_train_labels.npy")
kt = np.load(C / "j1_train_kt.npy", mmap_mode="r")
n = len(y); half = np.arange(n) % 2 == 0
SMOOTH = 1.0
counts = collections.defaultdict(lambda: collections.Counter())
for e in train:
    counts[e["column"]][e["shown_option_ids"][e["label"]]] += 1
keyid = [[(e["column"], o) for o in e["shown_option_ids"]] for e in exs]
CONFIGS = [(L, g, s) for L in (14, 16, 17, 18, 19, 20, 21) for g in (0, 1) for s in (2, 3)]
for L, g, s in CONFIGS:
    K = np.asarray(kt[:, L, s::4, g * 128:(g + 1) * 128], dtype=np.float32)
    accs, ident = [], []
    for tr, te in ((half, ~half), (~half, half)):
        sums, cnt = collections.defaultdict(lambda: np.zeros(128, np.float32)), collections.Counter()
        for i in np.where(tr)[0]:
            for j in range(5):
                sums[keyid[i][j]] += K[i, j]; cnt[keyid[i][j]] += 1
        ok, idok = [], []
        for i in np.where(te)[0]:
            col, nopt = exs[i]["column"], exs[i]["n_item_options"]
            c = counts[col]; tot = sum(c.values())
            seen = [o for o in range(nopt) if cnt[(col, o)]]
            if len(seen) < 2:
                ok.append(False); continue
            cents = np.stack([sums[(col, o)] / cnt[(col, o)] for o in seen])
            w = np.array([np.log((c[o] + SMOOTH) / (tot + SMOOTH * nopt)) for o in seen])
            d = (w - w.mean()) @ (cents - cents.mean(0))
            ok.append((K[i] @ d).argmax() == y[i])
            for j, o in enumerate(exs[i]["shown_option_ids"]):
                if o in seen:
                    sims = (cents - cents.mean(0)) @ (K[i, j] - cents.mean(0)) - 0.5 * ((cents - cents.mean(0)) ** 2).sum(1)
                    idok.append(seen[int(sims.argmax())] == o)
        accs.append(np.mean(ok)); ident.append(np.mean(idok))
    print(f"L{L} g{g} sub{s}: item-conditional key score acc {np.mean(accs):.3f}   option-identity decodability {np.mean(ident):.3f}")
