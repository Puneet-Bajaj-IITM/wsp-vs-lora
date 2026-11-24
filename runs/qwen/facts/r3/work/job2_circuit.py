"""Job 2: attention-path diagnostic (constant query), exact MLP-input captures, person neurons, full circuit test."""
import sys
from pathlib import Path

import numpy as np
import torch

from wsp import config
from wsp.harness import Lab
from wsp.scoring import render_prompts

sys.path.insert(0, str(config.EDITS_DIR))
import _lookup_lib as L  # noqa: E402
import circuit as C  # noqa: E402
import decode  # noqa: E402
import person_stage as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "cap"
PREFIX = str(config.EDITS_DIR / "lookup_v2")
CHUNK = 512
MIN_DIAGNOSTIC_ACC = 0.6
DIAGNOSTIC_VALUE = 0
FALLBACK_AMPLITUDES = ((1000.0, 60.0), (300.0, 30.0))
BASE_CONSTS = ("value_tokens", "prompt_tokens", "mu", "scalars")

OUT.mkdir(exist_ok=True)
lab = Lab(edit=None)
model, tok = lab.model, lab.tok
mtr, mdv = decode.load_meta(config.PROCESSED_DIR, "train"), decode.load_meta(config.PROCESSED_DIR, "dev")
consts = L.load_consts(PREFIX, BASE_CONSTS)

seen = set()
for split in ("train", "dev"):
    for p in render_prompts(tok, lab.examples(split)):
        seen.update(tok(p)["input_ids"])
print("prompt token set matches local:", seen == set(consts["prompt_tokens"].tolist()), len(seen))
print("value tokens:", [tok.decode([t]) for t in consts["value_tokens"].reshape(-1)])
print("tied embeddings:", model.get_input_embeddings().weight is model.lm_head.weight)

emb = model.get_input_embeddings().weight
value_idx = torch.as_tensor(consts["value_tokens"].reshape(-1))
modules = {"attn0": lab.layers[L.H1_LAYER].self_attn, "mlp26": lab.layers[L.STAGE_A_LAYER].mlp,
           "attn27": lab.layers[L.H2_LAYER].self_attn}
snapshot = {k: {n: v.clone() for n, v in m.state_dict().items()} for k, m in modules.items()}
emb_snapshot = emb[value_idx].clone()


def restore(*names):
    with torch.no_grad():
        for name in names:
            if name == "emb":
                emb[value_idx] = emb_snapshot
            else:
                modules[name].load_state_dict(snapshot[name])


def apply(stages, **kw):
    with torch.no_grad():
        return L.apply_stages(model, prepared, consts, stages=stages, **kw)


with torch.no_grad():
    prepared = L.prepare(model, tok, consts)
duals = prepared["duals"]
x_prompt = L.layer0_inputs(model, sorted(seen))
letter_score = x_prompt @ duals.sum(0)
print("letterness of prompt tokens: #>0.5:", int((letter_score > 0.5).sum()), "max |non-letter|:",
      float(np.sort(np.abs(letter_score))[-6]), "dual norms", np.linalg.norm(duals, axis=1).round(4))

# ---- T0: constant-query diagnostic of E1 + H1 + H2 ----
has_value = np.where((mtr["attr"] == 0) & (mtr["shown"] == DIAGNOSTIC_VALUE).any(1))[0][:CHUNK].tolist()
target_slot = (mtr["shown"][has_value] == DIAGNOSTIC_VALUE).argmax(1)
base_logp_tr = np.load(OUT / "logp_train.npy") if (OUT / "logp_train.npy").exists() else None
gain_scale, diag_ok = 1.0, False
for code_amp, slot_amp in FALLBACK_AMPLITUDES:
    L.CODE_AMPLITUDE = L.MARKER_AMPLITUDE = code_amp
    L.SLOT_AMPLITUDE = slot_amp
    apply(("E1", "H1", "H2"), constant_query=DIAGNOSTIC_VALUE)
    r = lab.forward("train", has_value, note=f"T0 constant-query diagnostic amp={code_amp}/{slot_amp}")
    logp = r["logp"]
    acc = float((logp.argmax(1) == target_slot).mean())
    picked = logp[np.arange(len(logp)), target_slot]
    margin = float((picked - (logp.sum(1) - picked) / 4).mean())
    print(f"T0 amp={code_amp}/{slot_amp}: predicts slot of '{tok.decode([int(value_idx[DIAGNOSTIC_VALUE])])}' acc={acc:.3f} "
          f"mean margin={margin:.2f} (designed {L.DEFAULT_LOGIT_GAIN}) hist={r['summary']['pred_letter_hist']} "
          f"by target slot={[round(float((logp.argmax(1) == target_slot)[target_slot == s].mean()), 2) for s in range(5)]}",
          flush=True)
    np.save(OUT / f"t0_logp_{int(code_amp)}.npy", logp)
    restore("attn27")
    if acc >= MIN_DIAGNOSTIC_ACC:
        diag_ok = True
        gain_scale = float(np.clip(L.DEFAULT_LOGIT_GAIN / max(margin, 1e-3), 0.3, 3.0))
        break
    restore("emb", "attn0")
print("diagnostic ok:", diag_ok, "gain_scale:", gain_scale, flush=True)
if not diag_ok:
    sys.exit(0)

# ---- exact MLP-26 input: zero the MLP output so the block output equals its mid residual ----
with torch.no_grad():
    modules["mlp26"].down_proj.weight.zero_()
mids = {}
for split, n in (("train", len(mtr["pid"])), ("dev", len(mdv["pid"]))):
    parts = []
    for start in range(0, n, CHUNK):
        idx = list(range(start, min(n, start + CHUNK)))
        r = lab.forward(split, idx, note=f"mid-26 capture (E1+H1, mlp26 zeroed) {split} {start}",
                        capture_layers=(L.STAGE_A_LAYER,))
        parts.append(r["hidden"][L.STAGE_A_LAYER])
    mids[split] = np.concatenate(parts)
    np.save(OUT / f"mid26_{split}.npy", mids[split].astype(np.float16))
restore("mlp26")
utr, udv = C.unit_rms(mids["train"]), C.unit_rms(mids["dev"])
print("mid rms", C.mean_rms(mids["train"]), "person centroid acc r200:",
      decode.centroid_acc(utr, mtr["pid"], udv, mdv["pid"], 300, 200), flush=True)
best_q, cv = P.select_quantile(utr, mtr)
print("train-only leave-one-template-out quantile selection:", cv, "->", best_q, flush=True)
float_consts = P.build_person_consts(utr, mtr, quantile=best_q)
print("SIM stage A float keys train", P.evidence_accuracy(float_consts, utr, mtr), flush=True)
print("SIM stage A float keys dev", P.evidence_accuracy(float_consts, udv, mdv), flush=True)
consts.update(P.build_person_consts(utr, mtr, quantile=best_q, levels=P.QUANT_LEVELS))
print("SIM stage A 4-bit keys train", P.evidence_accuracy(consts, utr, mtr), flush=True)
print("SIM stage A 4-bit keys dev", P.evidence_accuracy(consts, udv, mdv), flush=True)
gain = L.DEFAULT_LOGIT_GAIN * gain_scale
packed = P.pack_keys(consts["keys"])
np.save(f"{PREFIX}_keys_z.npy", packed)
np.save(f"{PREFIX}_gate.npy", consts["gate"])
np.save(f"{PREFIX}_bias_dir.npy", consts["bias_dir"])
np.save(f"{PREFIX}_table.npy", consts["table"].astype(np.int8))
np.save(f"{PREFIX}_gain.npy", np.array([gain, L.CODE_AMPLITUDE, L.SLOT_AMPLITUDE], dtype=np.float32))
print("packed key bytes:", packed.nbytes, flush=True)


def metrics(logp, labels):
    return round(float((logp.argmax(1) == labels).mean()), 4), round(float(-logp[np.arange(len(logp)), labels].mean()), 4)


def full_dev(note, capture=()):
    logps, hidden = [], {layer: [] for layer in capture}
    for start in range(0, len(mdv["pid"]), CHUNK):
        idx = list(range(start, min(len(mdv["pid"]), start + CHUNK)))
        r = lab.forward("dev", idx, note=f"{note} dev {start}", capture_layers=capture)
        logps.append(r["logp"])
        for layer in capture:
            hidden[layer].append(r["hidden"][layer])
    return np.concatenate(logps), {k: np.concatenate(v) for k, v in hidden.items()}


# ---- full circuit on the already-marked model ----
apply(("A", "H2"), logit_gain=gain)
logp_dev, hid = full_dev(f"lookup_v2 full circuit gain={gain:.2f}", capture=(26, 27))
np.save(OUT / "v2_logp_dev.npy", logp_dev)
np.save(OUT / "v2_h26_dev.npy", hid[26].astype(np.float16))
print("FULL DEV lookup_v2 (acc, nll):", metrics(logp_dev, mdv["label"]),
      "by attr:", [metrics(logp_dev[mdv["attr"] == a], mdv["label"][mdv["attr"] == a]) for a in range(3)],
      "hist:", np.bincount(logp_dev.argmax(1), minlength=5) / len(logp_dev), flush=True)
ev = (hid[26] @ prepared["codes"]["D"].T.astype(np.float32)) / (L.EVIDENCE_CODE_U * float(consts["scalars"][0]))
ev = ev.reshape(len(ev), 3, 10)[np.arange(len(ev)), mdv["attr"]]
shown_ev = np.take_along_axis(ev, mdv["shown"], axis=1)
print("realised evidence at h26: argmax-shown acc", float((shown_ev.argmax(1) == mdv["label"]).mean()),
      "true mean", float(ev[np.arange(len(ev)), mdv["value"]].mean()), "abs mean", float(np.abs(ev).mean()),
      "agreement evidence-argmax vs model-argmax", float((shown_ev.argmax(1) == logp_dev.argmax(1)).mean()), flush=True)
idx = list(range(CHUNK))
sub_labels = mdv["label"][:CHUNK]
print("gain", round(gain, 2), "dev512 (acc, nll):", metrics(logp_dev[:CHUNK], sub_labels), flush=True)
for factor in (0.5, 2.0):
    restore("attn27")
    apply(("H2",), logit_gain=gain * factor)
    r = lab.forward("dev", idx, note=f"lookup_v2 gain x{factor} dev[0:512]")
    print("gain", round(gain * factor, 2), "dev512 (acc, nll):", metrics(r["logp"], sub_labels), flush=True)
    np.save(OUT / f"v2_logp_dev512_gainx{factor}.npy", r["logp"])
restore("attn27")
apply(("H2",), logit_gain=gain)
r = lab.forward("train", idx, note="lookup_v2 full circuit train[0:512]")
print("FULL train512", r["summary"], flush=True)

# ---- the packaged edit file on a fresh model ----
del lab, model, modules, snapshot, emb
torch.cuda.empty_cache()
lab = Lab(edit="lookup_v2")
logp_pkg, _ = full_dev("lookup_v2 packaged edit")
np.save(OUT / "v2_logp_dev_packaged.npy", logp_pkg)
print("PACKAGED lookup_v2 full dev (acc, nll):", metrics(logp_pkg, mdv["label"]),
      "max |dlogp| vs in-place:", float(np.abs(logp_pkg - logp_dev).max()), flush=True)
