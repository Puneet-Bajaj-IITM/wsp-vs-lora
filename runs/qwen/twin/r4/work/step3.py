"""Job 3: 6 capture passes (raw option-token k_proj inputs saved for offline design), v2 read-outs, 4 KQ tests."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import kq  # noqa: E402
import readout2 as r2  # noqa: E402
from cap2 import OptionCapture  # noqa: E402
from step2 import HEADS, K_LAYERS, centred_table, run_test  # noqa: E402

M_LAYERS = (3, 6, 9, 12, 15, 17)
RESID_LAYER = 17
N_CAP_PASSES = 4
CAP_SEED = 2
TESTS = (("rome", 2.0), ("rome", 4.0), ("rome", 8.0), ("hebb_z", 4.0))
OUT = HERE / "j3"


def analyse(name, Z, second_moment, var, oth_mean, ctx, report, kinds):
    dev = ctx["dev"]
    Z = Z.to(dev)
    sm = r2.slot_means(Z)
    basis = r2.slot_basis(sm)
    Zc = r2.centre(Z, sm)
    whiten = {"hebb": None, "hebb_z": 1.0 / var.to(dev).clamp(min=1e-8)}
    if "rome" in kinds:
        whiten["rome"] = r2.whitener(second_moment.to(dev))
    A, B, full = ctx["half"], ~ctx["half"], {}
    om = oth_mean.to(dev)
    for kind in kinds:
        w = r2.build_readout(Zc, ctx["pairs"], ctx["s_pair"], A, basis, whiten[kind], om)
        ev = r2.evaluate(Zc, ctx["pairs"], ctx["labels"], ctx["s_pair"], w, B)
        report[f"{name}/{kind}"] = ev
        print(name, kind, json.dumps(ev), flush=True)
        full[kind] = r2.build_readout(Zc, ctx["pairs"], ctx["s_pair"], torch.ones_like(A), basis, whiten[kind], om).cpu()
    return full, sm.cpu()


def main():
    from wsp.harness import Lab
    from wsp.scoring import render_prompts
    OUT.mkdir(exist_ok=True)
    lab = Lab()
    model, dev = lab.model, next(lab.model.parameters()).device
    train = lab.examples("train")
    cols, item_of, pair0, n_opts, n_pairs = common.item_index(train)
    s_np = centred_table(train, cols, pair0, n_opts)
    j1_idx = np.load(HERE / "j1small" / "idx.npy")
    j2_idx = np.load(HERE / "j2" / "cap_idx.npy")
    u_vec = {l: np.load(HERE / "j2" / f"u{l}.npy") for l in K_LAYERS}  # every input is loaded before any pass is spent
    base_logp = np.load(HERE / "j1small" / "logp.npy")[:common.PASS_SIZE]
    eval_idx = [int(i) for i in j1_idx[:common.PASS_SIZE]]
    skip = set(int(i) for i in j1_idx) | set(int(i) for i in j2_idx)
    cap_idx = common.stratified_indices(train, N_CAP_PASSES * common.PASS_SIZE, seed=CAP_SEED, skip=skip)
    cap = OptionCapture(model, lab.tok, K_LAYERS, M_LAYERS, RESID_LAYER)
    cap.attach()
    logps = []
    for p in range(N_CAP_PASSES):
        ids = cap_idx[p * common.PASS_SIZE:(p + 1) * common.PASS_SIZE]
        cap.begin_pass(render_prompts(lab.tok, [train[i] for i in ids]))
        r = lab.forward("train", ids, note=f"j3 base capture of option-token activations, pass {p}")
        logps.append(r["logp"])
        print("capture pass", p, r["summary"]["accuracy"], "left", r["passes_left"], "errors", cap.errors, cap.last_error,
              "align", cap.align_errors, flush=True)
    cap.detach()
    exs = [train[i] for i in cap_idx]
    n = len(exs)
    ctx = {"dev": dev,
           "pairs": torch.tensor([[pair0[e["column"]] + o for o in e["shown_option_ids"]] for e in exs], device=dev),
           "labels": torch.tensor([e["label"] for e in exs], device=dev),
           "s_pair": torch.tensor(s_np, dtype=torch.float32, device=dev),
           "half": (torch.arange(n, device=dev) % 2 == 0)}
    np.save(OUT / "cap_idx.npy", np.array(cap_idx))
    np.save(OUT / "cap_logp.npy", np.concatenate(logps))
    report, consts = {}, {"rome": {}, "hebb_z": {}}
    for l in K_LAYERS:  # raw saves first, so an analysis failure cannot lose the capture
        np.save(OUT / f"x{l}.npy", cap.cat(cap.x_opt[l]).numpy().astype(np.float16))
    for l in K_LAYERS:
        Z, oth = cap.cat(cap.x_opt[l]), cap.cat(cap.x_oth[l])
        flat = torch.cat([oth.reshape(-1, oth.shape[-1]), Z.reshape(-1, Z.shape[-1])]).to(dev).double()
        second = (flat.T @ flat / flat.shape[0]).float().cpu()
        oflat = oth.reshape(-1, oth.shape[-1])
        np.save(OUT / f"second{l}.npy", second.numpy())
        np.save(OUT / f"othmean{l}.npy", oflat.mean(0).numpy())
        np.save(OUT / f"othvar{l}.npy", oflat.var(0).numpy())
        full, _ = analyse(f"x{l}", Z, second, oflat.var(0), oflat.mean(0), ctx, report, ("hebb_z", "rome"))
        for kind in consts:
            consts[kind][f"w{l}"] = full[kind].numpy()
            consts[kind][f"u{l}"] = u_vec[l]
        del Z, oth, flat
    for kind in consts:
        np.save(OUT / f"w_{kind}.npy", np.stack([consts[kind][f"w{l}"] for l in K_LAYERS]))
    for l in M_LAYERS:
        try:
            Z = cap.cat(cap.a_opt[l])
            mean = (cap.a_oth_sum[l] / cap.a_oth_n).float().cpu()
            var = (cap.a_oth_sq[l] / cap.a_oth_n).float().cpu() - mean ** 2
            analyse(f"mlp{l}", Z, None, var, mean, ctx, report, ("hebb", "hebb_z"))
        except Exception as err:
            print("mlp analysis failed", l, err, flush=True)
        Z = None
        cap.a_opt[l] = []
        torch.cuda.empty_cache()
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    originals = {l: model.model.layers[l].self_attn.k_proj.weight.detach().clone() for l in K_LAYERS}
    results, test_logp = {}, {}
    for kind, beta in TESTS:
        for l in K_LAYERS:
            W = model.model.layers[l].self_attn.k_proj.weight
            W.copy_(originals[l])
            kq.add_rank1(W, torch.tensor(consts[kind][f"u{l}"]), torch.tensor(consts[kind][f"w{l}"]), beta)
        res, lp = run_test(lab, eval_idx, base_logp, f"j3 test KQ v2 {kind} beta={beta}")
        results[f"{kind}_{beta}"], test_logp[f"{kind}_{beta}"] = res, lp
    (OUT / "tests.json").write_text(json.dumps(results, indent=1))
    np.savez(OUT / "test_logp.npz", **test_logp)
    print(json.dumps({"errors": cap.errors, "last_error": cap.last_error, "align": cap.align_errors}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
