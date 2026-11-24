"""Job 2: capture option-token activations (4 base passes), measure read-outs of the prior table, test KQ edits."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import kq  # noqa: E402
import prior  # noqa: E402
import readout as ro  # noqa: E402
from cap2 import OptionCapture  # noqa: E402

K_LAYERS = (18, 20, 21)
M_LAYERS = (13, 15, 17)
RESID_LAYER = 17
HEADS = {18: (3, 7), 20: (2, 4, 5, 9, 11), 21: (8, 9, 10, 11)}
N_CAP_PASSES = 4
CAP_SEED = 1
PCA_RANK = 128
X_KINDS = ("hebb", "hebb_z", "rome", "pcaw")
M_KINDS = ("hebb", "hebb_z")
TESTS = (("rome", 1.5), ("rome", 3.0), ("rome", 6.0), ("hebb_z", 3.0), ("pcaw", 3.0))
ADAPTIVE_BETA = {"rome": 12.0, "hebb_z": 6.0, "pcaw": 6.0}
OUT = HERE / "j2"


def centred_table(train, cols, pair0, n_opts):
    lp = prior.pair_log_prior(train)
    for c in cols:
        sl = slice(pair0[c], pair0[c] + n_opts[c])
        lp[sl] -= lp[sl].mean()
    return lp


def fit(Z, oth_mean, whiten, kind, pairs, items, s_pair, item_of_pair, n_items, sel):
    """Read-out fitted on examples `sel`: returns scaled w and the item means used for centring."""
    pm, cnt = ro.pair_means(Z[sel], pairs[sel], len(s_pair))
    im = ro.item_means(pm, cnt, item_of_pair, n_items)
    seen = cnt > 0
    keys = (pm - im[item_of_pair])[seen]
    T = keys.shape[1]
    w = ro.build(kind, keys.reshape(-1, keys.shape[-1]), s_pair[seen].repeat_interleave(T), whiten, PCA_RANK)
    w = ro.remove_offset(w, oth_mean)
    v = ro.scores(Z[sel], im, items[sel], w)
    sc = s_pair[pairs[sel]]
    sc = sc - sc.mean(1, keepdim=True)
    return w * (sc.std() / v.std().clamp(min=1e-8)), im


def analyse(name, Z, oth, oth_mean, oth_var, kinds, ctx, report):
    """Split-half retrieval quality per read-out kind, then the full-data read-outs."""
    dev = ctx["dev"]
    Z = Z.to(dev)
    whiten = {"hebb": None, "pcaw": None, "hebb_z": 1.0 / oth_var.to(dev).clamp(min=1e-8)}
    if "rome" in kinds:
        generic = torch.cat([oth.to(dev).reshape(-1, Z.shape[-1]), Z.reshape(-1, Z.shape[-1])])
        whiten["rome"] = ro.second_moment_inverse(generic)
    A, B = ctx["half"], ~ctx["half"]
    full, om = {}, oth_mean.to(dev)
    for kind in kinds:
        args = (ctx["pairs"], ctx["items"], ctx["s_pair"], ctx["item_of_pair"], ctx["n_items"])
        w, im = fit(Z, om, whiten[kind], kind, *args, A)
        ev = ro.evaluate(ro.scores(Z[B], im, ctx["items"][B], w), ctx["s_pair"][ctx["pairs"][B]], ctx["labels"][B])
        w_full, _ = fit(Z, om, whiten[kind], kind, *args, torch.ones_like(A))
        oth_sd = float(((oth.to(dev) - om) @ w_full).std()) if oth is not None else None
        report[f"{name}/{kind}"] = {"heldout": ev, "other_sd": oth_sd, "w_norm": float(w_full.norm())}
        print(name, kind, json.dumps(report[f"{name}/{kind}"]), flush=True)
        full[kind] = w_full.cpu()
    return full


def run_test(lab, eval_idx, base_logp, note):
    r = lab.forward("train", eval_idx, note=note)
    lab_arr, lp = r["labels"], r["logp"]
    base_ok, ok = base_logp.argmax(1) == lab_arr, lp.argmax(1) == lab_arr
    s = r["summary"]
    res = {"acc": s["accuracy"], "nll": s["nll"], "hist": s["pred_letter_hist"], "by_block": s["by_block"],
           "base_acc": float(base_ok.mean()), "fixed": int((~base_ok & ok).sum()), "broken": int((base_ok & ~ok).sum()),
           "changed_pred": float((lp.argmax(1) != base_logp.argmax(1)).mean()), "left": r["passes_left"]}
    print("TEST", note, json.dumps(res), flush=True)
    return res, lp


def main():
    from wsp.harness import Lab
    from wsp.scoring import render_prompts
    OUT.mkdir(exist_ok=True)
    lab = Lab()
    model, dev = lab.model, next(lab.model.parameters()).device
    cfg = model.config
    H, KV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    train = lab.examples("train")
    cols, item_of, pair0, n_opts, n_pairs = common.item_index(train)
    s_np = centred_table(train, cols, pair0, n_opts)
    j1_idx = np.load(HERE / "j1small" / "idx.npy")
    base_logp = np.load(HERE / "j1small" / "logp.npy")[:common.PASS_SIZE]
    eval_idx = [int(i) for i in j1_idx[:common.PASS_SIZE]]
    cap_idx = common.stratified_indices(train, N_CAP_PASSES * common.PASS_SIZE, seed=CAP_SEED, skip=set(int(i) for i in j1_idx))
    cap = OptionCapture(model, lab.tok, K_LAYERS, M_LAYERS, RESID_LAYER)
    cap.attach()
    logps = []
    for p in range(N_CAP_PASSES):
        ids = cap_idx[p * common.PASS_SIZE:(p + 1) * common.PASS_SIZE]
        cap.begin_pass(render_prompts(lab.tok, [train[i] for i in ids]))
        r = lab.forward("train", ids, note=f"j2 base capture of option-token activations, pass {p}")
        logps.append(r["logp"])
        print("capture pass", p, r["summary"]["accuracy"], "left", r["passes_left"], "errors", cap.errors, cap.last_error,
              "align", cap.align_errors, flush=True)
    cap.detach()
    exs = [train[i] for i in cap_idx]
    n = len(exs)
    ctx = {"dev": dev, "n_items": len(cols),
           "pairs": torch.tensor([[pair0[e["column"]] + o for o in e["shown_option_ids"]] for e in exs], device=dev),
           "items": torch.tensor([item_of[e["column"]] for e in exs], device=dev),
           "labels": torch.tensor([e["label"] for e in exs], device=dev),
           "s_pair": torch.tensor(s_np, dtype=torch.float32, device=dev),
           "item_of_pair": torch.tensor(np.concatenate([[item_of[c]] * n_opts[c] for c in cols]), device=dev),
           "half": (torch.arange(n, device=dev) % 2 == 0)}
    np.save(OUT / "cap_idx.npy", np.array(cap_idx))
    np.save(OUT / "cap_logp.npy", np.concatenate(logps))
    report, consts = {}, {k: {} for k in X_KINDS}
    for l in K_LAYERS:
        Z, oth = cap.cat(cap.x_opt[l]), cap.cat(cap.x_oth[l])
        flat = oth.reshape(-1, oth.shape[-1])
        full = analyse(f"x{l}", Z, oth, flat.mean(0), flat.var(0), X_KINDS, ctx, report)
        qbar = cap.cat(cap.q_last[l]).mean(0).view(H, hd)
        u, gains = kq.key_direction(qbar, HEADS[l], KV)
        report[f"u{l}"] = {"gains": gains, "n_dims": int((u != 0).sum()), "u_norm": float(u.norm()),
                           "rms_opt": float(cap.cat(cap.rms_opt[l]).mean())}
        print(f"u{l}", json.dumps(report[f"u{l}"]), flush=True)
        for kind in X_KINDS:
            consts[kind][f"w{l}"] = full[kind].numpy()
            consts[kind][f"u{l}"] = u.numpy()
        del Z, oth
    for l in M_LAYERS:
        Z = cap.cat(cap.a_opt[l])
        mean = (cap.a_oth_sum[l] / cap.a_oth_n).float().cpu()
        var = (cap.a_oth_sq[l] / cap.a_oth_n).float().cpu() - mean ** 2
        try:
            analyse(f"mlp{l}", Z, None, mean, var, M_KINDS, ctx, report)
        except Exception as err:
            print("mlp analysis failed", l, err, flush=True)
        del Z
        cap.a_opt[l] = []
        torch.cuda.empty_cache()
    r_opt = cap.cat(cap.r_opt)
    r_flat = torch.cat([r_opt.reshape(-1, r_opt.shape[-1]), cap.cat(cap.r_oth).reshape(-1, r_opt.shape[-1])])
    np.save(OUT / "resid17_dim_mean.npy", r_flat.mean(0).numpy())
    np.save(OUT / "resid17_dim_sd.npy", r_flat.std(0).numpy())
    for kind in X_KINDS:
        np.savez(OUT / f"consts_{kind}.npz", **consts[kind])
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    # ---- edit tests ----------------------------------------------------------------
    originals = {l: model.model.layers[l].self_attn.k_proj.weight.detach().clone() for l in K_LAYERS}

    def apply(kind, beta):
        touched = 0
        for l in K_LAYERS:
            W = model.model.layers[l].self_attn.k_proj.weight
            W.copy_(originals[l])
            touched += kq.add_rank1(W, torch.tensor(consts[kind][f"u{l}"]), torch.tensor(consts[kind][f"w{l}"]), beta)
        return touched

    results, test_logp = {}, {}
    tests = list(TESTS)
    for kind, beta in tests:
        touched = apply(kind, beta)
        res, lp = run_test(lab, eval_idx, base_logp, f"j2 test KQ {kind} beta={beta}")
        res["weights_touched"] = touched
        results[f"{kind}_{beta}"], test_logp[f"{kind}_{beta}"] = res, lp
    best_kind = max((k for k, b in TESTS if b == 3.0), key=lambda k: results[f"{k}_3.0"]["acc"])
    beta = ADAPTIVE_BETA[best_kind]
    apply(best_kind, beta)
    res, lp = run_test(lab, eval_idx, base_logp, f"j2 test KQ {best_kind} beta={beta} (adaptive)")
    results[f"{best_kind}_{beta}"], test_logp[f"{best_kind}_{beta}"] = res, lp
    for l in K_LAYERS:
        model.model.layers[l].self_attn.k_proj.weight.copy_(originals[l])
    (OUT / "tests.json").write_text(json.dumps(results, indent=1))
    np.savez(OUT / "test_logp.npz", **test_logp)
    print(json.dumps({"errors": cap.errors, "last_error": cap.last_error, "align": cap.align_errors}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
