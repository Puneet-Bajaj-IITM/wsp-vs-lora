"""Job 1: four instrumented base passes over a stratified train sample (2048 examples)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import segs  # noqa: E402
from inst import POS_TYPES, RESID_TYPES, Recorder  # noqa: E402

N_PASSES = 4
CAPTURE_LAYERS = (5, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, -1)
MLP_SAVE_LAYERS = tuple(range(2, 26))
OUT = HERE / "j1"


def option_stats(res: np.ndarray, pairs: np.ndarray, chosen: np.ndarray, half: np.ndarray) -> dict:
    """res (n,5,D) at one layer. Within-pair-centred mean difference chosen vs not, with a split-half check."""
    n, k, d = res.shape
    x = res.reshape(n * k, d).astype(np.float32)
    p = pairs.reshape(-1)
    sums = np.zeros((p.max() + 1, d), dtype=np.float32)
    np.add.at(sums, p, x)
    cnt = np.bincount(p, minlength=p.max() + 1)[:, None].clip(1)
    xc = (x - (sums / cnt)[p]).reshape(n, k, d)
    onehot = np.eye(k, dtype=bool)[chosen]

    def direction(mask):
        sel = xc[mask]
        oh = onehot[mask]
        return sel[oh].mean(0) - sel[~oh].mean(0)

    d_a = direction(half)
    acc = float(((xc[~half] @ d_a).argmax(1) == chosen[~half]).mean())
    d_full = direction(np.ones(n, dtype=bool))
    proj = xc @ (d_full / (np.linalg.norm(d_full) + 1e-9))
    return {"dir": d_full, "split_half_acc": acc, "gap": float(proj[onehot].mean() - proj[~onehot].mean()),
            "sd": float(proj.std()), "norm": float(np.linalg.norm(d_full)), "mean_norm": float(np.linalg.norm(x, axis=1).mean())}


def main():
    from wsp.harness import Lab
    from wsp.scoring import render_prompts
    OUT.mkdir(exist_ok=True)
    lab = Lab()
    train = lab.examples("train")
    cols, item_of, pair0, n_opts, n_pairs = common.item_index(train)
    idx = common.stratified_indices(train, N_PASSES * common.PASS_SIZE)
    rec = Recorder(lab.model, lab.tok, len(cols), n_pairs)
    rec.attach()
    logps, hidden = [], {l: [] for l in CAPTURE_LAYERS}
    for p in range(N_PASSES):
        ids = idx[p * common.PASS_SIZE:(p + 1) * common.PASS_SIZE]
        exs = [train[i] for i in ids]
        rec.begin_pass(render_prompts(lab.tok, exs), [item_of[e["column"]] for e in exs],
                       [[pair0[e["column"]] + o for o in e["shown_option_ids"]] for e in exs])
        r = lab.forward("train", ids, note=f"j1 base instrumented explore pass {p}", capture_layers=CAPTURE_LAYERS)
        rec.end_pass()
        logps.append(r["logp"])
        for l in CAPTURE_LAYERS:
            hidden[l].append(r["hidden"][l].astype(np.float16))
        print("pass", p, json.dumps(r["summary"]), "left", r["passes_left"], flush=True)
        print("  sdpa_calls", rec.sdpa_calls, "align_errors", rec.align_errors, "errors", rec.errors, rec.last_error, flush=True)
    rec.detach()
    exs = [train[i] for i in idx]
    logp = np.concatenate(logps)
    labels = np.array([e["label"] for e in exs])
    pred = logp.argmax(1)
    np.save(OUT / "idx.npy", np.array(idx))
    np.save(OUT / "logp.npy", logp)
    for l in CAPTURE_LAYERS:
        np.save(OUT / f"hid_{l}.npy", np.concatenate(hidden[l]))
    np.save(OUT / "attn.npy", np.concatenate(rec.attn))
    np.save(OUT / "match.npy", np.concatenate(rec.match))
    cnt = rec.item_n.clamp(min=1)[None, :, None]
    mlp_mean = (rec.mlp_sum / cnt)[list(MLP_SAVE_LAYERS)].cpu().numpy().astype(np.float16)
    np.save(OUT / "mlp_item_mean.npy", mlp_mean)
    n = float(rec.item_n.sum())
    np.save(OUT / "mlp_tot_mean.npy", (rec.mlp_tot / n).cpu().numpy().astype(np.float32))
    np.save(OUT / "mlp_tot_sq.npy", (rec.mlp_sq / n).cpu().numpy().astype(np.float32))
    np.save(OUT / "q_item_mean.npy", (rec.q_sum / cnt).cpu().numpy().astype(np.float16))
    np.save(OUT / "key_pair_mean.npy", (rec.key_sum / rec.pair_n.clamp(min=1)[None, :, None, None]).cpu().numpy().astype(np.float16))
    np.save(OUT / "pair_n.npy", rec.pair_n.cpu().numpy())
    np.save(OUT / "item_n.npy", rec.item_n.cpu().numpy())
    # option-position residual statistics
    pairs = np.array([[pair0[e["column"]] + o for o in e["shown_option_ids"]] for e in exs])
    half = np.arange(len(exs)) % 2 == 0
    report, dirs = {}, {}
    for t in RESID_TYPES:
        res = np.concatenate(rec.optres[t])
        for layer in range(rec.L):
            for name, chosen in (("model", pred), ("label", labels)):
                st = option_stats(res[:, layer], pairs, chosen, half)
                dirs[f"{t}_{name}_{layer}"] = st.pop("dir")
                report[f"{t}_{name}_{layer}"] = st
        del res
    np.savez(OUT / "opt_dirs.npz", **dirs)
    (OUT / "opt_report.json").write_text(json.dumps(report, indent=1))
    meta = {"sdpa_calls": rec.sdpa_calls, "align_errors": rec.align_errors, "errors": rec.errors, "last_error": rec.last_error,
            "capture_layers": CAPTURE_LAYERS, "mlp_layers": MLP_SAVE_LAYERS, "pos_types": POS_TYPES,
            "acc": float((pred == labels).mean()), "transformers": __import__("transformers").__version__,
            "rope_theta": getattr(lab.model.config, "rope_theta", None)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    for k in sorted(report, key=lambda s: (s.rsplit("_", 1)[0], int(s.rsplit("_", 1)[1]))):
        print(k, {a: round(b, 3) for a, b in report[k].items()})


if __name__ == "__main__":
    with torch.no_grad():
        main()
