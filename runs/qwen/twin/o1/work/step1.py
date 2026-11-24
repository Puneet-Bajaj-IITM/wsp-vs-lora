"""Job 1: base model on full dev + 3072 train, capture residuals at several layers; dump readout weights."""
import json
from pathlib import Path
import numpy as np
import torch
from wsp.harness import Lab

OUT = Path(__file__).resolve().parent / "arrays1"
OUT.mkdir(exist_ok=True)
LAYERS = (10, 16, 20, 23, 25, 26, 27, -1)
CHUNK = 512
N_TRAIN = 3072
SEED = 0

lab = Lab(edit=None)
np.save(OUT / "norm_w.npy", lab.model.model.norm.weight.float().cpu().numpy())
np.save(OUT / "letter_emb.npy", lab.model.model.embed_tokens.weight[lab.letters].float().cpu().numpy())
print("letter ids", lab.letters.tolist(), "tied", lab.model.lm_head.weight.data_ptr() == lab.model.model.embed_tokens.weight.data_ptr())

def run(split, idx, tag):
    logps, labels, hid = [], [], {l: [] for l in LAYERS}
    for s in range(0, len(idx), CHUNK):
        r = lab.forward(split, idx[s:s + CHUNK], note=f"base {tag} chunk {s}", capture_layers=LAYERS)
        logps.append(r["logp"]); labels.append(r["labels"])
        for l in LAYERS:
            hid[l].append(r["hidden"][l].astype(np.float16))
        print(tag, s, json.dumps(r["summary"]), "left", r["passes_left"], flush=True)
    np.save(OUT / f"{tag}_idx.npy", np.array(idx))
    np.save(OUT / f"{tag}_logp.npy", np.concatenate(logps))
    np.save(OUT / f"{tag}_labels.npy", np.concatenate(labels))
    for l in LAYERS:
        np.save(OUT / f"{tag}_h{l}.npy", np.concatenate(hid[l]))

dev_idx = list(range(len(lab.examples("dev"))))
run("dev", dev_idx, "dev")
rng = np.random.default_rng(SEED)
train_idx = sorted(rng.choice(len(lab.examples("train")), N_TRAIN, replace=False).tolist())
run("train", train_idx, "train")
