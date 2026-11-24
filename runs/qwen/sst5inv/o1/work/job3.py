"""Job 3: give the dead ' neutral' option a foothold by mixing its embedding toward the vector the
model reads as 'negative' (the original ' negative' row), plus weaker ' very'. Same 512 train idx.
The best variant by train accuracy then gets a full-dev measurement."""
import json
from pathlib import Path

import numpy as np

from wsp.harness import Lab

OUT = Path(__file__).parent / "job3"
EDIT = "swap_pn"
CHUNK = 512
# each variant: {"neutral": (w_neutral, w_negvec, w_posvec), "very": alpha_toward_fairly}
VARIANTS = {
    "N_neg0.25": {"neutral": (0.75, 0.25, 0.0)},
    "N_neg0.5": {"neutral": (0.5, 0.5, 0.0)},
    "N_neg0.75": {"neutral": (0.25, 0.75, 0.0)},
    "N_pn0.5": {"neutral": (0.5, 0.25, 0.25)},
    "V_fairly0.75": {"very": 0.75},
    "V_fairly0.75+N_neg0.5": {"very": 0.75, "neutral": (0.5, 0.5, 0.0)},
}


def tid(tok, w):
    ids = tok.encode(w, add_special_tokens=False)
    assert len(ids) == 1, (w, ids)
    return ids[0]


def set_variant(emb, ids, orig, spec):
    if "neutral" in spec:
        a, b, c = spec["neutral"]
        # after swap_pn, the ' positive' id holds the original ' negative' vector and vice versa
        emb[ids["neutral"]] = a * orig["neutral"] + b * orig["pos_id"] + c * orig["neg_id"]
    if "very" in spec:
        t = spec["very"]
        emb[ids["very"]] = (1 - t) * orig["very"] + t * orig["fairly"]


def main():
    OUT.mkdir(exist_ok=True)
    lab = Lab(edit=EDIT)
    tok = lab.tok
    emb = lab.model.get_input_embeddings().weight
    ids = {"neutral": tid(tok, " neutral"), "very": tid(tok, " very"), "fairly": tid(tok, " fairly"),
           "pos_id": tid(tok, " positive"), "neg_id": tid(tok, " negative")}
    orig = {k: emb[v].clone() for k, v in ids.items()}
    tr_idx = np.load(Path(__file__).parent / "job1" / "train_idx.npy").tolist()
    results = {}
    for name, spec in VARIANTS.items():
        set_variant(emb, ids, orig, spec)
        r = lab.forward("train", tr_idx, note=f"swap_pn + {name}")
        for k in ("neutral", "very"):
            emb[ids[k]] = orig[k]
        np.save(OUT / f"train_logp_{name}.npy", r["logp"])
        results[name] = r["summary"]
        print("VAR", name, r["summary"]["accuracy"], r["summary"]["nll"], r["summary"]["by_block"], r["summary"]["pred_letter_hist"])
    best = max(results, key=lambda k: results[k]["accuracy"])
    set_variant(emb, ids, orig, VARIANTS[best])
    n_dev = len(lab.examples("dev"))
    logp = np.concatenate([lab.forward("dev", list(range(s, min(s + CHUNK, n_dev))), note=f"full dev {best}")["logp"]
                           for s in range(0, n_dev, CHUNK)])
    np.save(OUT / "dev_logp_best.npy", logp)
    labels = np.array([e["label"] for e in lab.examples("dev")])
    acc = float((logp.argmax(1) == labels).mean()); nll = float(-logp[np.arange(len(labels)), labels].mean())
    print("DEV", best, acc, nll)
    results["_dev"] = {"best": best, "acc": acc, "nll": nll}
    json.dump(results, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
