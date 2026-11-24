"""Job 4: one nuisance-capture pass, then KQ v3 tests (read-out cleaned of slot/prefix/tail directions, optional
uniform option gate), each test with an attention probe."""
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
from nuis import NuisanceCapture, nuisance_basis  # noqa: E402
from probe import AttnProbe, segment_summary  # noqa: E402
from step2 import K_LAYERS, centred_table, run_test  # noqa: E402

PROBE_LAYERS = (18, 20, 21, 22, 23)
TESTS = ((2.0, 0.0), (4.0, 0.0), (4.0, 0.5), (8.0, 0.5), (2.0, 1.0))
LAST, END = 3, 4
TESTS_NO_GATE = ((2.0, 0.0), (4.0, 0.0), (1.0, 0.0), (8.0, 0.0))
GATE_MAX_SD = 0.6
OUT = HERE / "j4"


def clean_readouts(cap, l, w_item, w_opt, pairs, s_pair, n):
    """Project the shipped read-outs off the nuisance basis and re-normalise them on this capture."""
    sm = cap.summary(l, n)
    basis = nuisance_basis(sm["types"], sm["pre"], sm["tail"])
    Z = sm["types"][:, :, [LAST, END]].clone()                                          # n,5,2,D raw
    opt_mean = Z.reshape(-1, Z.shape[-1]).mean(0)
    gen = sm["gen"].reshape(-1, Z.shape[-1])
    Zc = r2.centre(Z, r2.slot_means(Z))
    w = r2.project_out(w_item, basis)
    v = Zc @ w
    sc = s_pair[pairs]
    sc = sc - sc.mean(1, keepdim=True)
    w = w * (sc.std() / v.std())
    wo = r2.project_out(w_opt, basis)
    wo = wo / (wo @ (opt_mean - gen.mean(0)))
    v = Zc @ w
    tm = sm["types"].mean(0)                                                            # 5,6,D
    info = {"basis_rank": int(basis.shape[1]), "n_pre": int(sm["pre"].shape[0]), "n_tail": int(sm["tail"].shape[0]),
            "kept_norm_item": float(w.norm() / w_item.norm()), "kept_norm_opt": float(r2.project_out(w_opt, basis).norm() / w_opt.norm()),
            "corr_table": float(np.corrcoef(v.reshape(-1).numpy(), sc[:, :, None].expand(-1, -1, 2).reshape(-1).numpy())[0, 1]),
            "acc_table": float((v.mean(2).argmax(1) == sc.argmax(1)).float().mean()),
            "w_generic_sd": float((gen @ w).std()), "w_generic_mean": float((gen @ w).mean()),
            "w_type_means": [round(float(a), 3) for a in (tm.mean(0) @ w)],
            "w_pre_max": float((sm["pre"] @ w).abs().max()), "w_tail_max": float((sm["tail"] @ w).abs().max()),
            "opt_on_options": [round(float(a), 3) for a in (tm.mean(0) @ wo)], "opt_generic_mean": float((gen @ wo).mean()),
            "opt_generic_sd": float((gen @ wo).std()), "opt_option_sd": float((sm["types"][:, :, [LAST, END]].reshape(-1, Z.shape[-1]) @ wo).std()),
            "opt_pre_max": float((sm["pre"] @ wo).abs().max()), "opt_tail_max": float((sm["tail"] @ wo).abs().max())}
    return w, wo, info


def main():
    from wsp.harness import Lab
    from wsp.scoring import render_prompts
    OUT.mkdir(exist_ok=True)
    # every input is loaded before any pass is spent
    u_vec = {l: torch.tensor(np.load(HERE / "j2" / f"u{l}.npy")) for l in K_LAYERS}
    w_item_all = torch.tensor(np.load(HERE / "j3" / "w_item_full.npy"))
    w_opt_all = torch.tensor(np.load(HERE / "j3" / "w_opt.npy"))
    j1_idx = np.load(HERE / "j1small" / "idx.npy")
    j3_idx = np.load(HERE / "j3" / "cap_idx.npy")
    base_logp = np.load(HERE / "j1small" / "logp.npy")[:common.PASS_SIZE]
    assert w_item_all.shape[0] == len(K_LAYERS) == w_opt_all.shape[0]
    lab = Lab()
    model = lab.model
    train = lab.examples("train")
    cols, item_of, pair0, n_opts, n_pairs = common.item_index(train)
    s_pair = torch.tensor(centred_table(train, cols, pair0, n_opts), dtype=torch.float32)
    eval_idx = [int(i) for i in j1_idx[:common.PASS_SIZE]]
    cap_idx = [int(i) for i in j3_idx[:common.PASS_SIZE]]
    cap_exs = [train[i] for i in cap_idx]
    cap = NuisanceCapture(model, lab.tok, K_LAYERS)
    cap.attach()
    cap.begin_pass(render_prompts(lab.tok, cap_exs))
    r = lab.forward("train", cap_idx, note="j4 base capture of nuisance statistics at k_proj inputs")
    cap.detach()
    print("capture", r["summary"]["accuracy"], "left", r["passes_left"], "errors", cap.errors, cap.last_error, flush=True)
    pairs = torch.tensor([[pair0[e["column"]] + o for o in e["shown_option_ids"]] for e in cap_exs])
    consts, report = {}, {}
    for i, l in enumerate(K_LAYERS):
        w, wo, info = clean_readouts(cap, l, w_item_all[i], w_opt_all[i], pairs, s_pair, len(cap_exs))
        consts[l], report[f"x{l}"] = (w, wo), info
        print(f"x{l}", json.dumps(info), flush=True)
    np.save(OUT / "w_item_v3.npy", np.stack([consts[l][0].numpy() for l in K_LAYERS]))
    np.save(OUT / "w_opt_v3.npy", np.stack([consts[l][1].numpy() for l in K_LAYERS]))
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    gate_ok = all(report[f"x{l}"]["opt_generic_sd"] < GATE_MAX_SD and report[f"x{l}"]["opt_option_sd"] < GATE_MAX_SD for l in K_LAYERS)
    tests = TESTS if gate_ok else TESTS_NO_GATE
    print("gate_ok", gate_ok, "tests", tests, flush=True)
    originals = {l: model.model.layers[l].self_attn.k_proj.weight.detach().clone() for l in K_LAYERS}
    probe = AttnProbe(model, lab.tok, PROBE_LAYERS)
    probe.attach()
    eval_prompts = render_prompts(lab.tok, [train[i] for i in eval_idx])
    results = {}
    for beta, gamma in tests:
        for l in K_LAYERS:
            W = model.model.layers[l].self_attn.k_proj.weight
            W.copy_(originals[l])
            kq.add_rank1(W, u_vec[l], consts[l][0] + gamma * consts[l][1], beta)
        probe.begin_pass(eval_prompts)
        name = f"b{beta}_g{gamma}"
        res, lp = run_test(lab, eval_idx, base_logp, f"j4 test KQ v3 beta={beta} gamma={gamma}")
        results[name] = res
        np.save(OUT / f"logp_{name}.npy", lp)
        np.save(OUT / f"attn_{name}.npy", probe.attn)
        print("  attention (L18H3, L20H4, L21H9, L22H7: pre persona facts question letters content ends tail)",
              np.round(segment_summary(probe.attn)[[0, 1, 2, 3], [3, 4, 9, 7]], 3).tolist(), "probe errors", probe.errors, probe.last_error, flush=True)
    probe.detach()
    for l in K_LAYERS:
        model.model.layers[l].self_attn.k_proj.weight.copy_(originals[l])
    (OUT / "tests.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
