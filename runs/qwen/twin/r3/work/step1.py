"""Job 1: base diagnostics (attention to options per head, DLA, mean keys) + first query-bias variants."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build
from common import TYPE_END, TYPE_LAST, Table, load
from instr import Instrument
from wsp import config
from wsp.harness import Lab, apply_edit

OUT = HERE / "out1"
OUT.mkdir(exist_ok=True)
SEED = 3
N_DIAG, N_HELD, CHUNK = 1600, 400, 400
MIN_LAYER, MIN_CONTENT_MASS = 4, 0.03
HEAD_SETS = (2, 6, 16)
BETAS = (2.0, 6.0)
REPORT = {}


def save_report():
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1, default=float))


def run(lab, ins, split, idx, note, accumulate=False, layers=(-1,)):
    exs = lab.examples(split)
    logps, atts, dlas, hid = [], [], [], []
    for s in range(0, len(idx), CHUNK):
        part = idx[s:s + CHUNK]
        if ins is not None:
            ins.begin([exs[i] for i in part], accumulate)
        r = lab.forward(split, part, note=note, capture_layers=layers)
        logps.append(r["logp"])
        hid.append(r["hidden"][-1].astype(np.float16))
        if ins is not None:
            a, d = ins.end()
            atts.append(a)
            dlas.append(d)
    logp = np.concatenate(logps)
    labels = np.array([exs[i]["label"] for i in idx])
    out = {"logp": logp, "labels": labels, "hid": np.concatenate(hid)}
    if ins is not None:
        out["att"], out["dla"] = np.concatenate(atts), np.concatenate(dlas)
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    REPORT[note] = {"acc": acc, "nll": nll, "n": len(idx),
                    "pred_hist": np.bincount(logp.argmax(1), minlength=5).tolist()}
    print(note, REPORT[note], flush=True)
    save_report()
    return out


def main():
    train = load("train")
    table = Table(train)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train)).tolist()
    idx_diag, idx_held = perm[:N_DIAG], perm[N_DIAG:N_DIAG + N_HELD]
    json.dump({"diag": idx_diag, "held": idx_held}, open(OUT / "idx.json", "w"))

    lab = Lab(edit=None)
    ins = Instrument(lab, table)
    attn0 = lab.layers[0].self_attn
    REPORT["meta"] = {"q_bias": attn0.q_proj.bias is not None, "H": ins.H, "G": ins.G, "hd": ins.hd,
                      "inv_freq": ins.inv_freq.cpu().numpy().tolist()}

    diag = run(lab, ins, "train", idx_diag, "base_train_diag", accumulate=True)
    np.save(OUT / "diag_att.npy", diag["att"])
    np.save(OUT / "diag_dla.npy", diag["dla"])
    np.save(OUT / "diag_logp.npy", diag["logp"])
    np.save(OUT / "diag_hid.npy", diag["hid"])
    key_cnt = ins.key_cnt.cpu().numpy()
    key_mean = (ins.key_sum / ins.key_cnt.clamp(min=1)[None, :, None, None]).cpu().numpy()
    np.save(OUT / "key_mean.npy", key_mean.astype(np.float16))
    np.save(OUT / "key_cnt.npy", key_cnt)
    np.save(OUT / "xq_mean.npy", (ins.xq_sum / ins.xq_cnt.clamp(min=1)[None, :, None]).cpu().numpy().astype(np.float16))
    np.save(OUT / "xq_cnt.npy", ins.xq_cnt.cpu().numpy())
    REPORT["errors"] = ins.errors[:20]
    save_report()

    held = run(lab, ins, "train", idx_held, "base_train_held")
    np.save(OUT / "held_att.npy", held["att"])
    np.save(OUT / "held_logp.npy", held["logp"])

    dev_idx = list(range(len(lab.examples("dev"))))
    dev = run(lab, ins, "dev", dev_idx, "base_dev")
    np.save(OUT / "dev_logp.npy", dev["logp"])
    np.save(OUT / "dev_att.npy", dev["att"])
    np.save(OUT / "dev_hid.npy", dev["hid"])
    ins.active = False

    try:
        variants(lab, table, diag, key_mean, key_cnt, idx_held, ins)
    except Exception:
        REPORT["variant_error"] = traceback.format_exc()
        print(REPORT["variant_error"], flush=True)
    save_report()


def variants(lab, table, diag, key_mean, key_cnt, idx_held, ins):
    stats = build.head_stats(diag["att"], diag["logp"].argmax(1), diag["labels"])
    L, H = stats["agree_pred"].shape
    score = stats["agree_pred"] - 0.2
    score[stats["content_mass"] < MIN_CONTENT_MASS] = -1
    score[:MIN_LAYER] = -1
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    lf = build.low_freq_mask(np.array(REPORT["meta"]["inv_freq"]))
    rows, info = [], []
    for layer, head in order[:max(HEAD_SETS)]:
        layer, head = int(layer), int(head)
        types = stats["mass_types"][layer, head]
        key_type = 0 if types[TYPE_LAST] >= types[TYPE_END] else 1
        delta, _, fid = build.head_delta(key_mean, key_cnt, table, table.w, layer, head, key_type,
                                         head // (H // ins.G), ins.hd, lf)
        rows.append(np.concatenate(([layer, head], delta)))
        info.append({"layer": layer, "head": head, "agree_pred": float(stats["agree_pred"][layer, head]),
                     "agree_label": float(stats["agree_label"][layer, head]),
                     "content_mass": float(stats["content_mass"][layer, head]),
                     "mass_types": types.tolist(), "key_type": key_type, "fidelity": float(fid),
                     "delta_norm": float(np.linalg.norm(delta))})
        print(info[-1], flush=True)
    REPORT["heads"] = info
    save_report()
    saved = [layer.self_attn.q_proj.bias.detach().clone() for layer in lab.layers]
    for k in HEAD_SETS:
        for beta in BETAS:
            name = f"qb_k{k}_b{int(beta)}"
            sel = [np.concatenate((r[:2], beta * r[2:])) for r in rows[:k]]
            build.write_edit(config.EDITS_DIR, name, sel)
            apply_edit(lab.model, lab.tok, name)
            run(lab, None, "train", idx_held, name)
            with torch.no_grad():
                for layer, b in zip(lab.layers, saved):
                    layer.self_attn.q_proj.bias.copy_(b)


if __name__ == "__main__":
    main()
