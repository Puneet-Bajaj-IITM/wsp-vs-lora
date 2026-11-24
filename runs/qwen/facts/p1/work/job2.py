"""Job 2: exact MLP-17 inputs (MLP-17 output zeroed), universal value-injection tests, first person-keyed test."""
import json

import numpy as np
import torch

from wsp.harness import Lab
from wlib import *

torch.set_grad_enabled(False)
CACHE.mkdir(exist_ok=True)
LK = 17                      # MLP layer that hosts the person neurons
PASS = 512
TARGET = 0                   # value id used as universal-injection target for each attribute
A_NORM = 8.0

lab = Lab(edit=None)
model, tok = lab.model, lab.tok
layers = model.model.layers
tr = fields(lab.examples("train")); dv = fields(lab.examples("dev"))
facts = fact_table(tr)
meta = json.loads((WORK / "meta.json").read_text())
base_logp = np.load(WORK / "logp_base_train.npy")
DV20 = np.load(WORK / "dv20.npy").astype(np.float64)           # (3,10,d) base 'leaning to value' directions at L20
E = model.get_input_embeddings().weight.float().cpu().numpy()
EMB = np.stack([[E[meta["value_ids"][a][v][0][0]] for v in range(10)] for a in ATTRS])
EMB = EMB / np.linalg.norm(EMB, axis=-1, keepdims=True) * A_NORM
gain = {l: layers[l].post_attention_layernorm.weight.float().cpu().numpy().astype(np.float64) for l in (17, 20)}
log = {}


def capture_mid(layer, split, idx, tag):
    """Residual after attention of `layer` (= MLP input before its norm): zero that MLP's output and capture the block output."""
    p = Patch(model, [layer])
    layers[layer].mlp.down_proj.weight.zero_()
    out = []
    for s in range(0, len(idx), PASS):
        r = lab.forward(split, idx[s:s + PASS], note=f"job2 mid{layer} capture {tag} (MLP{layer} zeroed)", capture_layers=(layer,))
        out.append(r["hidden"][layer]); print(tag, layer, s, r["passes_left"], flush=True)
    p.restore()
    return np.concatenate(out)


mid_tr = capture_mid(LK, "train", list(range(2700)), "train")
mid_dv = capture_mid(LK, "dev", list(range(900)), "dev")
mid_pr = capture_mid(LK, "preserve", list(range(512)), "pres")
rng = np.random.default_rng(0)
pool = np.where((tr["shown"] == TARGET).any(1))[0]
uni_idx = sorted(rng.choice(pool, PASS, replace=False).tolist())
mid20 = capture_mid(20, "train", uni_idx, "train-uni")
for name, arr in (("mid17_train", mid_tr), ("mid17_dev", mid_dv), ("mid17_pres", mid_pr), ("mid20_uni", mid20)):
    np.save(CACHE / f"{name}.npy", arr.astype(np.float16))

X, Xd, Xp = (mlp_input(m, gain[LK]) for m in (mid_tr, mid_dv, mid_pr))
W, mu, mhat, a = build_detectors(X, tr["pid"], Xp)
Sd = Xd @ W.T; St = X @ W.T; Sp = Xp @ W.T
own_d = Sd[np.arange(900), dv["pid"]]
print(f"DETECTORS mid{LK}: dev NC acc {(Sd.argmax(1) == dv['pid']).mean():.3f} own mean {own_d.mean():.2f} p10 {np.percentile(own_d, 10):.2f}; "
      f"sim acc dev {sim_accuracy(Sd, dv, facts):.3f} train {sim_accuracy(St, tr, facts):.3f}; |W| {np.linalg.norm(W, axis=1).mean():.2f}; "
      f"preserve max score mean {Sp.max(1).mean():.2f} p99 {np.percentile(Sp.max(1), 99):.2f}; "
      f"a.x train {np.mean(X @ a):.3f}+-{np.std(X @ a):.3f} dev {np.mean(Xd @ a):.3f}+-{np.std(Xd @ a):.3f} pres {np.mean(Xp @ a):.3f}+-{np.std(Xp @ a):.3f}; "
      f"mhat.x dev {np.mean(Xd @ mhat):.3f}+-{np.std(Xd @ mhat):.3f} pres {np.mean(Xp @ mhat):.3f}", flush=True)
# leave-one-template-out sanity for the detectors
m01 = tr["tpl"] < 2
Wl = build_detectors(X[m01], tr["pid"][m01], Xp)[0]
print(f"LOTO: NC {((X[~m01] @ Wl.T).argmax(1) == tr['pid'][~m01]).mean():.3f} sim {sim_accuracy(X[~m01] @ Wl.T, {k: v[~m01] for k, v in tr.items()}, facts):.3f}", flush=True)

mu20 = mlp_input(mid20, gain[20]).mean(0); mhat20 = mu20 / (mu20 @ mu20)
uf = {k: v[uni_idx] for k, v in tr.items()}
tslot = (uf["shown"] == TARGET).argmax(1)


def universal(layer, mh, vec, tag):
    """One neuron that fires on every task answer position and writes vec (the three target-value vectors summed)."""
    p = Patch(model, [layer])
    mlp = layers[layer].mlp
    idx = free_neurons(mlp, 1)
    act = UNIVERSAL_BETA / (1 + np.exp(-UNIVERSAL_BETA))
    write_neurons(mlp, idx, UNIVERSAL_BETA * mh[None], mh[None], vec[None] / act)
    r = lab.forward("train", uni_idx, note=f"job2 universal injection {tag}", capture_layers=(17, 20, 21))
    p.restore()
    lp = r["logp"]; pred = lp.argmax(1)
    res = {"tag": tag, "pick_target": float((pred == tslot).mean()), "base_pick_target": float((base_logp[uni_idx].argmax(1) == tslot).mean()),
           "dlogp_target": float((lp[np.arange(PASS), tslot] - base_logp[uni_idx][np.arange(PASS), tslot]).mean()),
           "by_attr": [float((pred == tslot)[uf["att"] == k].mean()) for k in range(3)],
           "base_by_attr": [float((base_logp[uni_idx].argmax(1) == tslot)[uf["att"] == k].mean()) for k in range(3)],
           "pred_hist": r["summary"]["pred_letter_hist"],
           "proj_h20": float(np.mean(r["hidden"][20] @ (vec / np.linalg.norm(vec)))), "norm_vec": float(np.linalg.norm(vec))}
    print("UNIVERSAL", json.dumps(res), flush=True)
    log.setdefault("universal", []).append(res)
    np.save(WORK / f"logp_uni_{tag}.npy", lp)
    return res


vB = DV20[:, TARGET].sum(0); vA = EMB[:, TARGET].sum(0)
results = [universal(17, mhat, 4 * vB, "B20at17_k4"), universal(17, mhat, 12 * vB, "B20at17_k12"),
           universal(20, mhat20, 4 * vB, "B20at20_k4"), universal(20, mhat20, 12 * vB, "B20at20_k12"),
           universal(17, mhat, vA, "EMBat17_n8")]

# ---- person-keyed test: four person groups get different value vectors (all written by MLP 17) ----
GROUPS = [("B_k4", lambda at, v: 4 * DV20[at, v]), ("B_k12", lambda at, v: 12 * DV20[at, v]),
          ("EMB_n8", lambda at, v: EMB[at, v]), ("B_k30", lambda at, v: 30 * DV20[at, v])]
V = np.stack([sum(GROUPS[p % 4][1](at, facts[p, at]) for at in range(3)) for p in range(300)])
p = Patch(model, [LK])
mlp = layers[LK].mlp
idx = free_neurons(mlp, 600)
write_neurons(mlp, idx, *person_neuron_weights(W, mhat, a, V))
np.save(WORK / "neuron_idx17.npy", idx.cpu().numpy())
for split, f, n in (("dev", dv, 900), ("train", tr, 2700)):
    ix = sorted(np.random.default_rng(1).choice(n, PASS, replace=False).tolist())
    r = lab.forward(split, ix, note=f"job2 person-keyed 4-group test on {split}", capture_layers=(17, 20, 21))
    ok = r["logp"].argmax(1) == r["labels"]; g = f["pid"][ix] % 4
    res = {"split": split, "acc": float(ok.mean()), "by_group": {GROUPS[k][0]: float(ok[g == k].mean()) for k in range(4)},
           "by_group_attr": {GROUPS[k][0]: [float(ok[(g == k) & (f["att"][ix] == at)].mean()) for at in range(3)] for k in range(4)},
           "nll": r["summary"]["nll"], "pred_hist": r["summary"]["pred_letter_hist"]}
    print("PERSON", json.dumps(res), flush=True)
    log.setdefault("person", []).append(res)
    np.save(WORK / f"logp_person_{split}.npy", r["logp"]); np.save(WORK / f"idx_person_{split}.npy", np.array(ix))
    np.save(CACHE / f"h17_person_{split}.npy", r["hidden"][17].astype(np.float16))
    np.save(CACHE / f"h20_person_{split}.npy", r["hidden"][20].astype(np.float16))
p.restore()
(WORK / "job2_results.json").write_text(json.dumps(log, indent=1))
print("passes left", r["passes_left"])
