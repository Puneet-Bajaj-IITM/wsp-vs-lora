"""Job 2: sweep closed-form edits of the ' very' and ' neutral' embedding rows on top of swap_pn,
all on the same 512 train examples as job 1 (so results are directly comparable)."""
import json
from pathlib import Path

import numpy as np

from wsp.harness import Lab

OUT = Path(__file__).parent / "job2"
EDIT = "swap_pn"
VERY, NEUTRAL = " very", " neutral"
# (name, row word, op, arg, alpha): op "scale" -> row *= alpha ; op "mix" -> row = (1-alpha) row + alpha emb[arg]
VARIANTS = [
    ("very_x0.7", VERY, "scale", None, 0.7),
    ("very_x0.4", VERY, "scale", None, 0.4),
    ("very_mix_fairly0.5", VERY, "mix", " fairly", 0.5),
    ("very_to_fairly", VERY, "mix", " fairly", 1.0),
    ("very_to_somewhat", VERY, "mix", " somewhat", 1.0),
    ("neutral_mix_mixed0.5", NEUTRAL, "mix", " mixed", 0.5),
    ("neutral_to_mixed", NEUTRAL, "mix", " mixed", 1.0),
]


def tid(tok, w):
    ids = tok.encode(w, add_special_tokens=False)
    assert len(ids) == 1, (w, ids)
    return ids[0]


def main():
    OUT.mkdir(exist_ok=True)
    lab = Lab(edit=EDIT)
    tok = lab.tok
    emb = lab.model.get_input_embeddings().weight
    tr_idx = np.load(Path(__file__).parent / "job1" / "train_idx.npy").tolist()
    for w in [VERY, NEUTRAL, " fairly", " somewhat", " mixed"]:
        print("TOK", repr(w), tok.encode(w, add_special_tokens=False))
    print("NORMS", {w: float(emb[tid(tok, w)].float().norm()) for w in [VERY, NEUTRAL, " fairly", " somewhat", " mixed", " positive", " negative"]})
    results = {}
    for name, word, op, arg, alpha in VARIANTS:
        row = tid(tok, word)
        orig = emb[row].clone()
        if op == "scale":
            emb[row] = orig * alpha
        else:
            emb[row] = (1 - alpha) * orig + alpha * emb[tid(tok, arg)]
        r = lab.forward("train", tr_idx, note=f"swap_pn + {name}")
        emb[row] = orig
        np.save(OUT / f"train_logp_{name}.npy", r["logp"])
        results[name] = r["summary"]
        print("VAR", name, r["summary"]["accuracy"], r["summary"]["nll"], r["summary"]["by_block"])
    json.dump(results, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
