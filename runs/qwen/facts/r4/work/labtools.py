"""Shared helpers for the facts/r4 lab scripts: edit loading, weight-table diagnostics, final-position checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from wsp import config

WORK = Path(__file__).resolve().parent
ATTRS = ("city", "profession", "hobby")
MAX_PASS = 512


def load_circuit():
    spec = importlib.util.spec_from_file_location("circuit", config.EDITS_DIR / "circuit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(*parts):
    print(*parts, flush=True)


def save_json(name, obj):
    (WORK / name).write_text(json.dumps(obj, indent=1))


def design_report(des, circuit):
    """Weight-only margins of the designed heads (tables over token type x offset; no network is run)."""
    cfg, rn, dh = des.cfg, des.rn, des.dh
    out = {"m": des.m, "sv_max": float(des.sv[0]), "sv_min": float(des.sv[-1]), "rho": float(des.rho),
           "eta_name": float(des.eta_name), "eta_null": float(des.eta_null), "channel_ratio": des.channel_ratio,
           "rival_name_max": float(des.rival_name.max()), "rival_name_p90": float(des.rival_name.quantile(0.9)),
           "null_response_max": float(des.null_response.abs().max()), "psi_norm": float(des.psi_norm),
           "gain": float(des.gain), "mid_norm_median": float(des.mid_norm.median()),
           "label_scale": [float(x) for x in des.label_scale], "weights_changed": int(des.weights_changed),
           "nat_emb_norm_median": float(des.nat_norms.median())}
    # gather heads: natural tokens vs the null token (nats)
    gain = cfg["gather_nats"] / (0.5 * rn * des.eta_name)
    for g, kappa in des.kappa_gather.items():
        rows = slice(g * dh, (g + 1) * dh)
        nat = gain * rn * (des.nat_unit @ (des.k0[rows].T @ kappa))
        npn = des.n_prompt_natural
        out[f"gather{g}_null_minus_max_prompt_token"] = float(cfg["gather_nats"] - nat[:npn].max())
        out[f"gather{g}_null_minus_max_vocab_sample"] = float(cfg["gather_nats"] - nat[npn:].max())
    # bind head: score table (nats) over offsets
    g = circuit.BIND_HEAD // des.group_size
    offs = torch.arange(0, circuit.MAX_OFFSET + 1, dtype=torch.float64)
    table = lambda keys: cfg["bind_gain"] * torch.stack([des.bind_score(keys, o) for o in offs])   # (offsets, tokens)
    lab = table(des.keys(des.label_unit, g))
    null = table(des.keys(des.null_unit[None], g))
    nat = table(des.keys(des.nat_unit, g))
    plain = table(des.keys(des.value_unit[:1], g))
    piece = table(des.keys(des.piece_unit[:1], g))
    out["bind_label_near"] = [[round(float(x), 1) for x in lab[o]] for o in range(0, 9)]
    out["bind_label_far_max"] = float(lab[circuit.BIND_FAR_FROM:].max())
    out["bind_null_range"] = [float(null.min()), float(null.max())]
    out["bind_plain_range"] = [float(plain.min()), float(plain.max())]
    out["bind_piece_range"] = [float(piece.min()), float(piece.max())]
    npn = des.n_prompt_natural
    out["bind_prompt_token_max"] = float(nat[:, :npn].max())
    out["bind_vocab_sample_max"] = float(nat[:, npn:].max())
    worst = torch.argsort(-nat[:, :npn].max(0).values)[:6]
    out["bind_worst_prompt_tokens"] = [(des.tok.decode([des.prompt_natural_ids[int(i)]]), round(float(nat[:, int(i)].max()), 1),
                                        int(nat[:, int(i)].argmax())) for i in worst]
    near = {}
    for text in (".", "\n", ":\n", "Options", "?\n\n", "\n\n"):
        ids = des.tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] in des.prompt_natural_ids:
            col = des.prompt_natural_ids.index(ids[0])
            near[text] = [round(float(x), 1) for x in nat[:7, col]]
    out["bind_near_tokens_by_offset"] = near
    return out


def final_position_checks(des, result, layer0=0, layer1=1):
    """Compares captured final-position residuals with what the design predicts there."""
    exs, hid = result["examples"], result["hidden"]
    pids = [e["pid"] for e in exs]
    h0 = torch.tensor(hid[layer0], dtype=torch.float64)
    coords = h0 @ des.y.T / float(des.gain)
    pred = des.proto_x[pids]
    cos = torch.nn.functional.cosine_similarity(coords, pred, dim=1)
    resp = (coords @ des.detect.T)
    own = resp[torch.arange(len(pids)), pids]
    resp[torch.arange(len(pids)), pids] = -1e9
    info = {"h0_norm_median": float(h0.norm(dim=1).median()),
            "h0_code_part_norm_median": float((h0 @ des.y.T).norm(dim=1).median()),
            "h0_offcode_norm_median": float((h0 - (h0 @ des.y.T) @ des.y).norm(dim=1).median()),
            "gather_cos_min": float(cos.min()), "gather_cos_median": float(cos.median()),
            "const_coord_median": float((coords @ des.const).median()),
            "own_response_min": float(own.min()), "own_response_median": float(own.median()),
            "rival_response_max": float(resp.max()),
            "flag_coord_abs_max_at_final": float((h0 @ des.psi_hat).abs().max())}
    labels = torch.tensor(result["labels"])
    if layer1 in hid:   # what the readout head wrote, in units of omega, per letter direction
        delta = torch.tensor(hid[layer1], dtype=torch.float64) - h0
        coef = delta @ torch.linalg.pinv(des.answer_dirs) / des.cfg["omega"]
        true = coef[torch.arange(len(labels)), labels]
        info["read_coef_true_median"] = float(true.median())
        info["read_coef_true_p10"] = float(true.quantile(0.1))
        info["read_coef_argmax_acc"] = float((coef.argmax(1) == labels).double().mean())
        info["read_coef_abs_sum_median"] = float(coef.abs().sum(1).median())
    for layer in hid:
        if layer in (layer0,):
            continue
        h = torch.tensor(hid[layer], dtype=torch.float64)
        if layer == -1:
            h = h / des.df
        c = torch.nn.functional.cosine_similarity(h[:, None, :], des.answer_dirs[None], dim=2)
        info[f"L{layer}_norm_median"] = float(h.norm(dim=1).median())
        info[f"L{layer}_cos_true_dir_median"] = float(c[torch.arange(len(labels)), labels].median())
        info[f"L{layer}_cos_true_dir_p10"] = float(c[torch.arange(len(labels)), labels].quantile(0.1))
        info[f"L{layer}_argmax_dir_acc"] = float((c.argmax(1) == labels).double().mean())
    return info


def errors(result, limit=40):
    pred = result["logp"].argmax(1)
    rows = []
    for e, p, lp in zip(result["examples"], pred, result["logp"]):
        if p != e["label"]:
            rows.append({"pid": e["pid"], "column": e["column"], "label": e["label"], "pred": int(p),
                         "value": e["shown_option_ids"][e["label"]], "logp": [round(float(x), 2) for x in lp],
                         "q": e["user_message"].split("\n")[0]})
    return rows[:limit], len(rows)


def full_split(lab, split, note):
    """Scores every example of a split (several budgeted passes) and pools the results."""
    n = len(lab.examples(split))
    logps, exs = [], []
    for start in range(0, n, MAX_PASS):
        r = lab.forward(split, list(range(start, min(n, start + MAX_PASS))), note=f"{note} [{start}:]")
        logps.append(r["logp"])
        exs.extend(r["examples"])
    logp = np.concatenate(logps)
    labels = np.array([e["label"] for e in exs])
    correct = logp.argmax(1) == labels
    by_block = {}
    for e, ok in zip(exs, correct):
        by_block.setdefault(e["block"], []).append(bool(ok))
    errs, n_err = errors({"examples": exs, "logp": logp})
    return {"n": int(n), "accuracy": float(correct.mean()), "nll": float(-logp[np.arange(n), labels].mean()),
            "by_block": {k: float(np.mean(v)) for k, v in by_block.items()}, "n_err": n_err, "errors": errs,
            "passes_left": r["passes_left"]}


def detector_headroom(des, result, layer0=0):
    """Smallest (own response - threshold) / (1 - threshold) at the final position; 1 is nominal, <= 0 would not fire."""
    pids = torch.tensor([e["pid"] for e in result["examples"]])
    h0 = torch.tensor(result["hidden"][layer0], dtype=torch.float64)
    coords = h0 @ des.y.T / float(des.gain)
    own = (coords * des.detect[pids]).sum(1) / (coords @ des.const)
    thr = des.threshold[pids]
    return float(((own - thr) / (1.0 - thr)).min())
