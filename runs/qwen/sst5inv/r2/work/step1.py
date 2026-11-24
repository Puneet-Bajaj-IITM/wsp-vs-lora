"""Job 1: measure the option-word swap on train and full dev, plus neutral-row scale probes."""
import json
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
OUT = WORK / "job1"
EDIT = "swap_v1"
CAPTURE = (12, 16, 20, 24, 27)
PASS = 512
TRAIN_A = list(range(0, PASS))
TRAIN_B = list(range(PASS, 2 * PASS))
NEUTRAL_SCALES = (0.6, 1.6)


def save(tag, r):
    np.save(OUT / f"logp_{tag}.npy", r["logp"])
    for layer, h in r["hidden"].items():
        np.save(OUT / f"hid_{tag}_L{layer}.npy", h.astype(np.float16))
    print(tag, json.dumps(r["summary"]), "left", r["passes_left"], flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    lab = Lab(edit=EDIT)
    tok = lab.tok
    sample = lab.examples("train")[0]
    ids = tok.encode(sample["user_message"], add_special_tokens=False)
    print("tokens", [tok.decode([i]) for i in ids][-45:], flush=True)
    emb = lab.model.get_input_embeddings().weight
    words = {w: tok.encode(" " + w, add_special_tokens=False) for w in ("negative", "positive", "neutral", "very")}
    print("word ids", words, "norms", {w: float(emb[i[0]].float().norm()) for w, i in words.items()}, flush=True)

    save("trainA", lab.forward("train", TRAIN_A, note="swap_v1 train 0-511", capture_layers=CAPTURE))
    n_dev = len(lab.examples("dev"))
    dev_logp = []
    for k, start in enumerate(range(0, n_dev, PASS)):
        r = lab.forward("dev", list(range(start, min(start + PASS, n_dev))), note=f"swap_v1 dev part {k}")
        dev_logp.append(r["logp"])
        print("dev part", k, json.dumps(r["summary"]), flush=True)
    np.save(OUT / "logp_dev.npy", np.concatenate(dev_logp))
    save("trainB", lab.forward("train", TRAIN_B, note="swap_v1 train 512-1023", capture_layers=CAPTURE))

    neutral = words["neutral"][0]
    original = emb[neutral].clone()
    for scale in NEUTRAL_SCALES:
        with torch.no_grad():
            emb[neutral] = original * scale
        save(f"trainA_neu{scale}", lab.forward("train", TRAIN_A, note=f"swap + neutral row x{scale}, train 0-511"))
    with torch.no_grad():
        emb[neutral] = original


if __name__ == "__main__":
    main()
