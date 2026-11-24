"""Builds the two-stage circuit constants from base captures and simulates it on captured activations."""
from pathlib import Path

import numpy as np

import circuit as C

FINAL_LAYER = 27


def save_consts(prefix: Path, la: int, lb: int, stage_a: dict, stage_b: dict) -> None:
    for name, layer, stage in (("a", la, stage_a), ("b", lb, stage_b)):
        np.save(f"{prefix}_{name}_layer.npy", np.array(layer))
        for key in ("gate", "up", "down"):
            np.save(f"{prefix}_{name}_{key}.npy", stage[key])


def build(htr, mtr, weights, la, lb, rank, logit_gain=None):
    """la / lb are the blocks whose MLPs host stage A / B; keys come from the outputs of blocks la-1 / lb-1."""
    readout, readout_matrix = C.readout_directions(weights["lm_head_letters"], weights["final_norm"])
    u_a, u_b = C.unit_rms(htr[la - 1]), C.unit_rms(htr[lb - 1])
    codes = C.quiet_codes([C.unit_rms(htr[layer]) for layer in range(la - 1, FINAL_LAYER + 1, 2)], readout_matrix)
    rms_a_out, rms_b_in, final_rms = C.mean_rms(htr[la]), C.mean_rms(htr[lb - 1]), C.mean_rms(htr[FINAL_LAYER])
    code_raw = C.CODE_AMPLITUDE_U * rms_a_out
    code_u_at_b = code_raw / rms_b_in
    stage_a = C.build_stage_a(u_a, mtr, rank, codes, code_raw)
    stage_b = C.build_stage_b(u_b, mtr, codes, code_u_at_b, readout, final_rms, logit_gain)
    info = {"codes": codes, "code_raw": code_raw, "code_u_at_b": code_u_at_b, "final_rms": final_rms,
            "readout_matrix": readout_matrix, "rms_b_in": rms_b_in}
    quiet_std = float((u_b @ codes.reshape(-1, codes.shape[-1]).T).std())
    print(f"build la={la} lb={lb} rank={rank} code_u_at_b={code_u_at_b:.3f} quiet_std_at_b={quiet_std:.4f} "
          f"sigma={stage_b['feats']['sigma']:.3f} gain={stage_b['gain']:.2f} "
          f"own/other median={np.median(stage_a['own']):.3f}/{np.median(stage_a['other']):.3f} "
          f"bias_std_a={float((u_a @ stage_a['up'][0]).std()):.4f}", flush=True)
    return stage_a, stage_b, info


def simulate(stage_a, stage_b, info, h, meta, la, lb, base_logp, tag):
    fire = C.simulate_stage_a(stage_a, C.unit_rms(h[la - 1]))
    own = fire[np.arange(len(fire)), meta["pid"]]
    others = fire.sum(1) - own
    codes = info["codes"].reshape(-1, info["codes"].shape[-1])
    table = stage_a["table"]
    code_vec = sum(fire @ info["codes"][a, table[:, a]] for a in range(C.NUM_ATTRS)) * info["code_raw"]
    u_b = C.unit_rms(h[lb - 1] + code_vec)
    delta = C.simulate_stage_b(stage_b, u_b, info["final_rms"], info["readout_matrix"])
    logits = base_logp + delta
    acc = float((logits.argmax(1) == meta["label"]).mean())
    acc_delta = float((delta.argmax(1) == meta["label"]).mean())
    print(f"SIM {tag}: own_fire mean={own.mean():.3f} frac>0.5={float((own > 0.5).mean()):.3f} "
          f"others_sum mean={others.mean():.3f} sim_acc={acc:.3f} sim_acc_delta_only={acc_delta:.3f} "
          f"delta_range={float(np.ptp(delta, axis=1).mean()):.2f}", flush=True)
    return acc


def code_decode_accuracy(h_layer, meta, codes):
    proj = h_layer.astype(np.float32) @ codes.reshape(-1, codes.shape[-1]).T
    proj = proj.reshape(len(h_layer), C.NUM_ATTRS, C.NUM_VALUES)[np.arange(len(h_layer)), meta["attr"]]
    true = proj[np.arange(len(proj)), meta["value"]]
    return float((proj.argmax(1) == meta["value"]).mean()), float(true.mean()), float(np.abs(proj).mean())
