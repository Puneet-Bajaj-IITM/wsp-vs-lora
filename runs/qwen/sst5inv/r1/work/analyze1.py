"""Local numpy analysis of job 1: how the swap model reads sentiment, and what class offsets could do."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labutil import ARRAYS, NUM_CLASSES, class_logp, shown_ids, true_sentiment

ROOT = Path(__file__).resolve().parents[2]
MIRROR = np.arange(NUM_CLASSES)[::-1]
LAYERS = (8, 12, 16, 20, 24, 27, -1)

train = [json.loads(l) for l in (ROOT / "data/processed/sst5inv/train.jsonl").read_text().splitlines()]
idx = np.load(ARRAYS / "swap_trainA_idx.npy")
exs = [train[i] for i in idx]
sent, shown = true_sentiment(exs), shown_ids(exs)
labels = np.array([e["label"] for e in exs])
target = MIRROR[sent]

for tag in ("base_trainA", "swap_trainA"):
    logp = np.load(ARRAYS / f"{tag}_logp.npy")
    cls = class_logp(logp, shown)
    read = cls if tag.startswith("base") else cls[:, MIRROR]
    print(tag, "acc vs inverted label", (logp.argmax(1) == labels).mean(), "| model-read sentiment acc", (read.argmax(1) == sent).mean())
    print(" mean read logp by true sentiment\n", np.array([read[sent == c].mean(0) for c in range(NUM_CLASSES)]).round(2))
    print(" letter-position accuracy of the read-true option:", [round(float((read.argmax(1) == sent)[np.array([list(s).index(t) for s, t in zip(shown, (sent if tag.startswith('base') else target))]) == p].mean()), 3) for p in range(5)])

logp = np.load(ARRAYS / "swap_trainA_logp.npy")
cls = class_logp(logp, shown)
print("neutral offset sweep (text-order logits, added to neutral):")
for off in (0, 1, 2, 3, 4, 5, 6, 8):
    adj = cls.copy(); adj[:, 2] += off
    print(off, round(float((adj.argmax(1) == target).mean()), 3), "pred neutral frac", round(float((adj.argmax(1) == 2).mean()), 3))
print("very offset sweep (added to both very options):")
for off in (-3, -2, -1, 0, 1, 2):
    adj = cls.copy(); adj[:, [0, 4]] += off
    print(off, round(float((adj.argmax(1) == target).mean()), 3))

for layer in LAYERS:
    h = np.load(ARRAYS / f"swap_trainA_h{layer}.npy").astype(np.float32)
    means = np.stack([h[sent == c].mean(0) for c in range(NUM_CLASSES)])
    centred = means - h.mean(0)
    valence = means[[3, 4]].mean(0) - means[[0, 1]].mean(0)
    unit = valence / np.linalg.norm(valence)
    proj = (h - h.mean(0)) @ unit
    print(f"layer {layer}: |h| {np.linalg.norm(h, axis=1).mean():.1f} |valence diff| {np.linalg.norm(valence):.2f} "
          f"class mean proj {[round(float(proj[sent == c].mean()), 2) for c in range(NUM_CLASSES)]} sd {proj.std():.2f}")
