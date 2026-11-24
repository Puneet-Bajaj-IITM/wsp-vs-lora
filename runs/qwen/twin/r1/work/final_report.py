"""Final: compare lexroute_v1 full-dev log-probs with the base, by block/option set; count edited weights."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from captures import KAGGLE_OUT  # noqa: E402
from common import items, option_texts  # noqa: E402
from wsp import config  # noqa: E402
from wsp.metrics import load_split, summarize  # noqa: E402

JOB3 = KAGGLE_OUT / "job3"
BASE_DEV_LOGP = config.RESULTS_DIR / "logp_base_dev.npy"
TOTAL_PARAMS = 1_543_714_304  # Qwen2.5-1.5B-Instruct, tied embeddings
D_MODEL = 1536
SWAPPED_ROWS = 6  # ' strongly',' little',' Often',' Sometimes',' A',' D'
REPLACED_ROWS = 1  # ' oppose'
LETTER_ROWS = 5


def block_table(dev, base, edited) -> None:
    labels = np.array([e["label"] for e in dev])
    ok_b, ok_e = base.argmax(1) == labels, edited.argmax(1) == labels
    per = collections.defaultdict(lambda: [[], []])
    for ex, b, e in zip(dev, ok_b, ok_e):
        per[ex["block"]][0].append(b)
        per[ex["block"]][1].append(e)
    print(f"{'block':34s} {'n':>5s} {'base':>7s} {'edit':>7s} {'delta':>7s}")
    for blk, (b, e) in sorted(per.items(), key=lambda kv: -len(kv[1][0])):
        print(f"{blk:34s} {len(b):5d} {np.mean(b):7.3f} {np.mean(e):7.3f} {np.mean(e) - np.mean(b):+7.3f}")
    print(f"{'ALL':34s} {len(labels):5d} {ok_b.mean():7.4f} {ok_e.mean():7.4f} {ok_e.mean() - ok_b.mean():+7.4f}")
    flips = int((ok_e & ~ok_b).sum()), int((~ok_e & ok_b).sum())
    print("paired flips wrong->right, right->wrong:", flips)


def pick_texts(dev, logp, block, top=6):
    c = collections.Counter(option_texts(e)[p] for e, p in zip(dev, logp.argmax(1)) if e["block"] == block)
    return c.most_common(top)


def main() -> None:
    dev = load_split("dev")
    base = np.load(BASE_DEV_LOGP)
    edited = np.load(JOB3 / "dev_full_logp.npy")
    consts = json.loads((JOB3 / "constants.json").read_text())
    print("constants", consts)
    print("EDIT summary", json.dumps(summarize(edited, dev)))
    print("BASE summary", json.dumps(summarize(base, dev)))
    block_table(dev, base, edited)
    for blk in ("Personality", "False consensus", "Cognitive tests", "Economic preferences"):
        print(blk, "| base picks", pick_texts(dev, base, blk), "| edit picks", pick_texts(dev, edited, blk))
    n_rows = SWAPPED_ROWS + REPLACED_ROWS + (LETTER_ROWS if consts["use_offsets"] else 0)
    n_changed = n_rows * D_MODEL + D_MODEL  # + final norm weight
    print(f"weights changed: {n_changed:,} of {TOTAL_PARAMS:,} = {n_changed / TOTAL_PARAMS:.2e}")


if __name__ == "__main__":
    main()
