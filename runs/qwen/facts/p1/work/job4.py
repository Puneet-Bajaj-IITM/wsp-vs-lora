"""Job 4: weight exports for the constant-bias design, MLP-16 input capture, universal test of whitened directions."""
import importlib.util
import json

import numpy as np
import torch

from wsp import config
from wsp.harness import Lab
from wlib import *

torch.set_grad_enabled(False)
CACHE.mkdir(exist_ok=True)
PASS = 512
TARGET = 0
HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
spec = importlib.util.spec_from_file_location("facts_mem", config.EDITS_DIR / "facts_mem.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
lab = Lab(edit=None)
model = lab.model
layers = model.model.layers
tr = fields(lab.examples("train"))
c, cfg = fm.load_consts()

# ---- weights only: residual vector produced by each unit of value-bias, and how strongly later readers see a direction ----
dev = layers[0].mlp.gate_proj.weight.device
G = torch.zeros(1536, 1536, dtype=torch.float64, device=dev)
for l in range(16, 22):
    a = layers[l].self_attn
    Wo = a.o_proj.weight.double()                                              # (1536, 12*128)
    M = sum(Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] @ torch.eye(KV_HEADS * HEAD_DIM, dtype=torch.float64, device=dev)[(h // (HEADS // KV_HEADS)) * HEAD_DIM:(h // (HEADS // KV_HEADS) + 1) * HEAD_DIM]
            for h in range(HEADS))                                             # (1536, 256): d residual / d b_v
    np.save(CACHE / f"M_bv_{l}.npy", M.cpu().numpy().astype(np.float32))
    np.save(CACHE / f"bv_{l}.npy", a.v_proj.bias.float().cpu().numpy())
    print(l, "o_proj bias", a.o_proj.bias is not None, "v bias norm", float(a.v_proj.bias.float().norm()), "M sing top/bottom",
          [float(s) for s in torch.linalg.svdvals(M)[[0, 127, 255]]], flush=True)
    if l >= 17:
        g1 = layers[l].input_layernorm.weight.double()
        for W in (a.q_proj.weight, a.k_proj.weight, a.v_proj.weight):
            Wd = W.double() * g1[None]; G += Wd.T @ Wd
    if l <= 20:
        g2 = layers[l].post_attention_layernorm.weight.double()
        for W in (layers[l].mlp.gate_proj.weight, layers[l].mlp.up_proj.weight):
            Wd = W.double() * g2[None]; G += Wd.T @ Wd
np.save(CACHE / "read_gram_16_21.npy", G.cpu().numpy().astype(np.float32))
E = model.get_input_embeddings().weight
print("unembed row norm mean", float(E.float().norm(dim=1).mean()), "final norm gain mean", float(model.model.norm.weight.float().abs().mean()))

# ---- MLP-16 input at the answer position (MLP-16 zeroed) ----
rng = np.random.default_rng(0)
pool = np.where((tr["shown"] == TARGET).any(1))[0]
uni_idx = sorted(rng.choice(pool, PASS, replace=False).tolist())
patch = Patch(model, [16])
layers[16].mlp.down_proj.weight.zero_()
r = lab.forward("train", uni_idx, note="job4 mid16 capture train (MLP16 zeroed)", capture_layers=(16,))
patch.restore()
np.save(CACHE / "mid16_uni.npy", r["hidden"][16].astype(np.float16))

# ---- universal test: one MLP-20 neuron writes kappa * whitened centred unit direction of the target value (all attributes) ----
base_logp = np.load(WORK / "logp_base_train_uni.npy")
ctx = c["ctx20"].astype(np.float64); task = ctx[:3].mean(0); mh = task / (task @ task)
vd = c["vdirs"].astype(np.float64); vd = vd / np.linalg.norm(vd, axis=-1, keepdims=True)
uf = {k: v[uni_idx] for k, v in tr.items()}
tslot = (uf["shown"] == TARGET).argmax(1)
res = []
for kappa in (10.0, 25.0, 60.0):
    patch = Patch(model, [20])
    mlp = layers[20].mlp
    act = UNIVERSAL_BETA / (1 + np.exp(-UNIVERSAL_BETA))
    write_neurons(mlp, free_neurons(mlp, 1), UNIVERSAL_BETA * mh[None], mh[None], (kappa * vd[:, TARGET].sum(0) / act)[None])
    r = lab.forward("train", uni_idx, note=f"job4 universal whitened direction kappa {kappa}", capture_layers=(20, 21))
    patch.restore()
    lp = r["logp"]; pred = lp.argmax(1)
    out = {"kappa": kappa, "pick_target": float((pred == tslot).mean()), "base_pick_target": float((base_logp.argmax(1) == tslot).mean()),
           "dlogp_target": float((lp[np.arange(PASS), tslot] - base_logp[np.arange(PASS), tslot]).mean()),
           "by_attr": [float((pred == tslot)[uf["att"] == k].mean()) for k in range(3)], "pred_hist": r["summary"]["pred_letter_hist"]}
    print("UNIVERSAL_W", json.dumps(out), flush=True); res.append(out)
    np.save(CACHE / f"logp_uniw_{int(kappa)}.npy", lp)
(WORK / "job4_results.json").write_text(json.dumps(res, indent=1))
print("passes left", r["passes_left"])
