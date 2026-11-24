"""Job 1: polarity-swap edit on a train sample (with captures) and on full dev."""
import json

import numpy as np

from common import WORK, confusion, full_dev, sample
from wsp.harness import Lab

EDIT = "swap_polarity"
CAPTURE_LAYERS = (8, 12, 16, 20, 24, -1)

lab = Lab(edit=EDIT)
tok = lab.tok
for word in (" negative", " positive", " neutral", " very"):
    print(repr(word), tok.encode(word, add_special_tokens=False))

train_idx = sample(len(lab.examples("train")))
train = lab.forward("train", train_idx, note="swap edit, train sample, captures", capture_layers=CAPTURE_LAYERS)
print("TRAIN", json.dumps(train["summary"], default=str)[:1500])
print("TRAIN confusion (rows = mirrored target class, cols = predicted class)\n", confusion(train))
np.save(WORK / "j1_train_idx.npy", np.array(train_idx))
np.save(WORK / "j1_train_logp.npy", train["logp"])
for layer, hidden in train["hidden"].items():
    np.save(WORK / f"j1_train_h{layer}.npy", hidden.astype(np.float16))

dev = full_dev(lab, "swap edit, full dev")
print("DEV acc", dev["acc"], "nll", dev["nll"])
print("DEV confusion\n", confusion(dev))
np.save(WORK / "j1_dev_logp.npy", dev["logp"])
