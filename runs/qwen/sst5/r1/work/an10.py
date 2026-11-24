"""Letter-position bias of the final edit (train blocks 0-1 from job 6, dev from job 7)."""
import json
import numpy as np
from make_stats import W, load
np.set_printoptions(precision=3, suppress=True, linewidth=200)
_, base, shown, cls, _ = load()
o6 = np.load(W/"job6_out.npz"); o5 = np.load(W/"job5_out.npz"); d7 = np.load(W/"job7_dev.npz")
lab_tr = np.array([list(s).index(c) for s, c in zip(shown[:1024], cls[:1024])])
dev = [json.loads(l) for l in open(W.parents[1]/"data/processed/sst5/dev.jsonl")]
lab_dev = np.array([e["label"] for e in dev])
tr = np.concatenate([o6["combined"], o6["final_block1"]]) / 5.5
def show(name, lp, lab):
    print(f"{name:28s} acc {(lp.argmax(1)==lab).mean():.4f} pred letters {np.bincount(lp.argmax(1), minlength=5)/len(lab)} acc by true letter {[round(float((lp.argmax(1)==lab)[lab==k].mean()),3) for k in range(5)]}")
show("train base (2 blocks)", base[:1024], lab_tr); show("train edit (2 blocks)", tr, lab_tr)
for k in ["qv21_g10", "tent_regression_check"]: show("train blk0 " + k, o5[k], lab_tr[:512])
show("dev base", d7["base"], lab_dev); show("dev edit", d7["edit"], lab_dev)
best = max(((( tr + np.array([b, 0, 0, 0, 0])).argmax(1) == lab_tr).mean(), b) for b in np.arange(0, 1.01, 0.05))
print("train: best additive bias on letter A (post-temperature logit units)", best)
for b in (0.1, 0.2, 0.3, 0.4, 0.5):
    print(b, "train acc", ((tr + np.array([b, 0, 0, 0, 0])).argmax(1) == lab_tr).mean())
