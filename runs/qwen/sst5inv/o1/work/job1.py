"""Job 1: tokenization check, then full-dev and one train pass with the swap_pn edit."""
import json
from pathlib import Path

import numpy as np

from wsp.harness import Lab

OUT = Path(__file__).parent / "job1"
EDIT = "swap_pn"
CHUNK = 512
TRAIN_SEED = 0
CAPTURE = (-1,)


def chunks(n):
    return [list(range(s, min(s + CHUNK, n))) for s in range(0, n, CHUNK)]


def main():
    OUT.mkdir(exist_ok=True)
    lab = Lab(edit=EDIT)
    tok = lab.tok
    ex = lab.examples("dev")[0]
    ids = tok.encode(ex["user_message"], add_special_tokens=False)
    print("TOKENS", [tok.decode([i]) for i in ids][-60:])
    for w in [" positive", " negative", "positive", "negative", " neutral", " very"]:
        print("TOK", repr(w), tok.encode(w, add_special_tokens=False))
    n_dev = len(lab.examples("dev"))
    logps = []
    for idx in chunks(n_dev):
        r = lab.forward("dev", idx, note=f"{EDIT} full dev")
        logps.append(r["logp"])
    logp = np.concatenate(logps)
    np.save(OUT / "dev_logp_swap.npy", logp)
    labels = np.array([e["label"] for e in lab.examples("dev")])
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    print("DEV", EDIT, acc, nll)
    rng = np.random.default_rng(TRAIN_SEED)
    tr_idx = sorted(rng.choice(len(lab.examples("train")), CHUNK, replace=False).tolist())
    r = lab.forward("train", tr_idx, note=f"{EDIT} train sample", capture_layers=CAPTURE)
    np.save(OUT / "train_logp_swap.npy", r["logp"])
    np.save(OUT / "train_idx.npy", np.array(tr_idx))
    np.save(OUT / "train_hfinal_swap.npy", r["hidden"][-1].astype(np.float16))
    print("TRAIN", json.dumps(r["summary"]))
    json.dump({"dev_acc": acc, "dev_nll": nll, "train": r["summary"]}, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
