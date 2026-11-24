"""Slot-controlled effect of the universal injections: is the target VALUE promoted, or just a letter?"""
import numpy as np
from common import load, CACHE, WORK
from wlib import fields

tr = fields(load("train"))
uni_idx = sorted(np.random.default_rng(0).choice(np.where((tr["shown"] == 0).any(1))[0], 512, replace=False).tolist())
f = {k: v[uni_idx] for k, v in tr.items()}
base = np.load(WORK / "logp_base_train_uni.npy")
tslot = (f["shown"] == 0).argmax(1)
runs = {"raw B20at17 k4": CACHE / "old/logp_uni_B20at17_k4.npy", "raw B20at17 k12": CACHE / "old/logp_uni_B20at17_k12.npy",
        "raw B20at20 k4": CACHE / "old/logp_uni_B20at20_k4.npy", "raw B20at20 k12": CACHE / "old/logp_uni_B20at20_k12.npy",
        "emb at17": CACHE / "old/logp_uni_EMBat17_n8.npy", "whitened k10": CACHE / "logp_uniw_10.npy", "whitened k25": CACHE / "logp_uniw_25.npy",
        "whitened k60": CACHE / "logp_uniw_60.npy"}
print("value effect = mean over slots s of [dlogit(slot s | target at s) - dlogit(slot s | target elsewhere)], logits centred per example")
for name, path in runs.items():
    lp = np.load(path)
    d = (lp - lp.mean(1, keepdims=True)) - (base - base.mean(1, keepdims=True))
    eff = [d[tslot == s, s].mean() - d[tslot != s, s].mean() for s in range(5)]
    se = np.sqrt(sum(d[tslot == s, s].var() / (tslot == s).sum() for s in range(5))) / 5
    letter = d.mean(0)
    print(f"{name:18s} value effect {np.mean(eff):+.2f} nats (se ~{se:.2f}) per slot {np.round(eff, 2)} | pure letter shift {np.round(letter, 2)}")
