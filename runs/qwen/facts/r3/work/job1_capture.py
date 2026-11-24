"""Job 1: base-model captures of the answer-position residual at every layer, all train + dev."""
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

OUT = Path(__file__).resolve().parent / "cap"
LAYERS = tuple(range(28)) + (-1,)
CHUNK = 512

lab = Lab(edit=None)
OUT.mkdir(exist_ok=True)
for split in ("train", "dev"):
    n = len(lab.examples(split))
    logps, hid = [], {layer: [] for layer in LAYERS}
    for start in range(0, n, CHUNK):
        idx = list(range(start, min(n, start + CHUNK)))
        r = lab.forward(split, idx, note=f"base capture {split} {start}", capture_layers=LAYERS)
        logps.append(r["logp"])
        for layer in LAYERS:
            hid[layer].append(r["hidden"][layer].astype(np.float16))
        print(split, start, r["summary"]["accuracy"], r["passes_left"], flush=True)
    np.save(OUT / f"logp_{split}.npy", np.concatenate(logps))
    np.save(OUT / f"hid_{split}.npy", np.stack([np.concatenate(hid[layer]) for layer in LAYERS]))

model, tok = lab.model, lab.tok
weights = {
    "lm_head_letters": model.lm_head.weight[lab.letters].float().cpu().numpy(),
    "final_norm": model.model.norm.weight.float().cpu().numpy(),
    "norm2": torch.stack([blk.post_attention_layernorm.weight for blk in lab.layers]).float().cpu().numpy(),
    "down_norms": torch.stack([blk.mlp.down_proj.weight.float().norm(dim=0) for blk in lab.layers]).cpu().numpy(),
    "gate_norms": torch.stack([blk.mlp.gate_proj.weight.float().norm(dim=1) for blk in lab.layers]).cpu().numpy(),
    "up_norms": torch.stack([blk.mlp.up_proj.weight.float().norm(dim=1) for blk in lab.layers]).cpu().numpy(),
}
np.savez(OUT / "weights.npz", **weights)
ex = lab.examples("train")[0]
from wsp.scoring import render_prompts
prompt = render_prompts(tok, [ex])[0]
print(repr(prompt))
print([tok.decode([t]) for t in tok(prompt)["input_ids"]])
print(model.model.layers[0].mlp)
print({k: v.shape for k, v in weights.items()})

from wsp import config
import decode
mtr, mdv = decode.load_meta(config.PROCESSED_DIR, "train"), decode.load_meta(config.PROCESSED_DIR, "dev")
htr, hdv = np.load(OUT / "hid_train.npy"), np.load(OUT / "hid_dev.npy")
decode.person_report(htr, hdv, mtr, mdv)
decode.slot_report(htr, hdv, mtr, mdv)

# ---- automatic first circuit attempt (guarded) ----
import circuit as C
import build_edit as B
from wsp.harness import apply_edit

MIN_ORACLE, MIN_PERSON, RANKS = 0.3, 0.3, (0, 30, 100)
oracle = {}
for layer in range(4, 27):
    feats = C.slot_features(C.unit_rms(htr[layer]), mtr)
    oracle[layer] = (C.oracle_slot_accuracy(feats, C.unit_rms(htr[layer]), mtr),
                     C.oracle_slot_accuracy(feats, C.unit_rms(hdv[layer]), mdv), feats["sigma"])
    print(f"Q2b oracle slot acc layer {layer}: train={oracle[layer][0]:.3f} dev={oracle[layer][1]:.3f} sigma={oracle[layer][2]:.3f}", flush=True)
lb_in = max(range(8, 27), key=lambda l: oracle[l][1])
person = {(l, r): decode.centroid_acc(C.unit_rms(htr[l]), mtr["pid"], C.unit_rms(hdv[l]), mdv["pid"], 300, r)
          for l in range(2, lb_in) for r in RANKS}
(la_in, rank) = max(person, key=person.get)
la, lb = la_in + 1, lb_in + 1
print(f"AUTO choice: stage A block {la} (person acc {person[(la_in, rank)]:.3f}, rank {rank}); "
      f"stage B block {lb} (oracle dev {oracle[lb_in][1]:.3f})", flush=True)
if oracle[lb_in][1] >= MIN_ORACLE and person[(la_in, rank)] >= MIN_PERSON:
    stage_a, stage_b, info = B.build(htr, mtr, weights, la, lb, rank)
    logp_tr, logp_dv = np.load(OUT / "logp_train.npy"), np.load(OUT / "logp_dev.npy")
    B.simulate(stage_a, stage_b, info, htr, mtr, la, lb, logp_tr, "train")
    B.simulate(stage_a, stage_b, info, hdv, mdv, la, lb, logp_dv, "dev")
    B.save_consts(config.EDITS_DIR / "twostage_v1", la, lb, stage_a, stage_b)
    np.save(OUT / "codes.npy", info["codes"])
    touched = {l: {k: v.clone() for k, v in lab.layers[l].mlp.state_dict().items()} for l in (la, lb)}
    idx = list(range(512))
    sub = {k: v[idx] for k, v in mdv.items()}
    apply_edit(model, tok, "stage_a_v1")
    cap = (la, lb - 1, 27)
    r = lab.forward("dev", idx, note="stage A only: code decode check", capture_layers=cap)
    print("T1 stage A only dev512", r["summary"], "max |dlogp| vs base", float(np.abs(r["logp"] - logp_dv[idx]).max()), flush=True)
    for layer in cap:
        print(f"T1 code decode at layer {layer}: acc/true/abs", B.code_decode_accuracy(r["hidden"][layer], sub, info["codes"]),
              "base:", B.code_decode_accuracy(hdv[layer][idx], sub, info["codes"]), flush=True)
    np.save(OUT / "t1_hidden.npy", np.stack([r["hidden"][layer] for layer in cap]).astype(np.float16))
    for l in (la, lb):
        lab.layers[l].mlp.load_state_dict(touched[l])
    apply_edit(model, tok, "twostage_v1")
    r = lab.forward("dev", idx, note="twostage_v1 dev[0:512]", capture_layers=(27,))
    print("T2 twostage dev512", r["summary"], flush=True)
    np.save(OUT / "t2_logp_dev512.npy", r["logp"])
    r = lab.forward("train", idx, note="twostage_v1 train[0:512]")
    print("T2 twostage train512", r["summary"], flush=True)
    np.save(OUT / "t2_logp_train512.npy", r["logp"])
