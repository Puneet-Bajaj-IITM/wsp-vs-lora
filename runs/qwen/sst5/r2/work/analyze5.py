"""Offline: is a letter-prior correction worth implementing? (labels are uniform over letters)"""
import json

import numpy as np

from common import CACHE, DATA

N_LETTERS = 5
ITERS = 50
np.set_printoptions(precision=2, suppress=True)


def uniform_matched_bias(logp: np.ndarray) -> np.ndarray:
    bias = np.zeros(N_LETTERS)
    for _ in range(ITERS):
        lp = logp + bias
        lp -= np.logaddexp.reduce(lp, 1, keepdims=True)
        bias -= np.log(np.exp(lp).mean(0) * N_LETTERS)
    return bias


rows = [json.loads(l) for l in (DATA / "train.jsonl").open()]
lab_a = np.array([r["label"] for r in rows[:1024]])
lab_b = np.array([r["label"] for r in rows[1024:1536]])
base_a = np.load(CACHE / "logp.npy")
bias_a = uniform_matched_bias(base_a)
print("letter bias from base A", bias_a, "acc A", (base_a.argmax(1) == lab_a).mean(), "->", ((base_a + bias_a).argmax(1) == lab_a).mean())
for name in ("base", "orth_L17_s2.0", "orth_L19_s1.0"):
    lp = np.load(CACHE / "job2" / f"logp_{name}.npy")
    own = uniform_matched_bias(lp)
    print(name, "acc B", (lp.argmax(1) == lab_b).mean(), "with A bias", ((lp + bias_a).argmax(1) == lab_b).mean(),
          "own bias", own, ((lp + own).argmax(1) == lab_b).mean())
