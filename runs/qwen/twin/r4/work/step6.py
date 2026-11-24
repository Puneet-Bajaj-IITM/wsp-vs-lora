"""Job 6: test 'target minus model' read-outs, select against the earlier best, set a temperature, write the final
edit constants, reload the model through the edit file and measure full dev."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import kq  # noqa: E402
from step2 import K_LAYERS, run_test  # noqa: E402

TESTS = (("m25", 2.0), ("m50", 2.0), ("m50", 4.0), ("m25", 3.5))
ROME = ("rome", 1.5)
TAU_GRID = tuple(round(1.0 - 0.05 * i, 2) for i in range(20))
EDIT_NAME = "kq_final"
DEV_PASSES_NEEDED = 4
OUT = HERE / "j6"


def temperature(logp: np.ndarray, labels: np.ndarray) -> float:
    """Largest-effect-free rule: the tau whose mean confidence is closest to the measured accuracy."""
    acc = float((logp.argmax(1) == labels).mean())
    gaps = []
    for tau in TAU_GRID:
        z = tau * logp
        p = np.exp(z - z.max(1, keepdims=True))
        gaps.append(abs(float((p.max(1) / p.sum(1)).mean()) - acc))
    return TAU_GRID[int(np.argmin(gaps))]


def main():
    from wsp import config
    from wsp.harness import Lab, passes_used, FORWARD_PASS_BUDGET
    OUT.mkdir(exist_ok=True)
    u_all = np.stack([np.load(HERE / "j2" / f"u{l}.npy") for l in K_LAYERS])
    w = {"m25": np.load(HERE / "j3" / "w_dm_m25.npy"), "m50": np.load(HERE / "j3" / "w_dm_m50.npy"),
         "rome": np.load(HERE / "j2" / "w_rome.npy")}
    rome_logp = [np.load(HERE / "j2" / "logp_rome_b1.5_eval1.npy"), np.load(HERE / "j5" / "logp_rome_b1.5_eval2.npy")]
    j1_idx = np.load(HERE / "j1small" / "idx.npy")
    j1_logp = np.load(HERE / "j1small" / "logp.npy")
    assert all(v.shape == (len(K_LAYERS), u_all.shape[1] * 6) for v in w.values()), "read-out shapes"
    assert (config.EDITS_DIR / f"{EDIT_NAME}.py").exists(), "edit file missing"
    evals = [([int(i) for i in j1_idx[k * common.PASS_SIZE:(k + 1) * common.PASS_SIZE]],
              j1_logp[k * common.PASS_SIZE:(k + 1) * common.PASS_SIZE]) for k in range(2)]
    lab = Lab()
    model = lab.model
    train = lab.examples("train")
    eval_labels = [np.array([train[i]["label"] for i in ids]) for ids, _ in evals]
    originals = {l: model.model.layers[l].self_attn.k_proj.weight.detach().clone() for l in K_LAYERS}
    results, logps = {}, {}

    def test(kind, beta, which):
        for i, l in enumerate(K_LAYERS):
            W = model.model.layers[l].self_attn.k_proj.weight
            W.copy_(originals[l])
            kq.add_rank1(W, torch.tensor(u_all[i]), torch.tensor(w[kind][i]), beta)
        res, lp = run_test(lab, evals[which][0], evals[which][1], f"j6 test KQ {kind} beta={beta} eval{which + 1}")
        results[f"{kind}_b{beta}_eval{which + 1}"] = res
        logps[(kind, beta, which)] = lp
        np.save(OUT / f"logp_{kind}_b{beta}_eval{which + 1}.npy", lp)
        (OUT / "tests.json").write_text(json.dumps(results, indent=1))
        return res["acc"]

    accs = {t: test(*t, 0) for t in TESTS}
    best = max(accs, key=accs.get)
    acc2 = test(*best, 1)
    pooled_new = 0.5 * (accs[best] + acc2)
    pooled_rome = float(np.mean([(lp.argmax(1) == y).mean() for lp, y in zip(rome_logp, eval_labels)]))
    if pooled_new > pooled_rome:
        kind, beta = best
        pooled_logp = np.concatenate([logps[(kind, beta, 0)], logps[(kind, beta, 1)]])
    else:
        kind, beta = ROME
        pooled_logp = np.concatenate(rome_logp)
    tau = temperature(pooled_logp, np.concatenate(eval_labels))
    choice = {"kind": kind, "beta": beta, "tau": tau, "pooled_new": pooled_new, "pooled_rome": pooled_rome,
              "best_new": list(best), "base_pooled": float(np.mean([(b.argmax(1) == y).mean() for (_, b), y in zip(evals, eval_labels)]))}
    print("CHOICE", json.dumps(choice), flush=True)
    (OUT / "choice.json").write_text(json.dumps(choice, indent=1))
    consts = {"w": w[kind].astype(np.float32), "u": u_all.astype(np.float32), "cfg": np.array([beta, tau], dtype=np.float32)}
    for key, arr in consts.items():
        np.save(OUT / f"{EDIT_NAME}_{key}.npy", arr)
        np.save(config.EDITS_DIR / f"{EDIT_NAME}_{key}.npy", arr)
    if FORWARD_PASS_BUDGET - passes_used() < DEV_PASSES_NEEDED:
        print("not enough passes left for full dev", flush=True)
        return
    del lab, model, originals
    gc.collect()
    torch.cuda.empty_cache()
    lab = Lab(edit=EDIT_NAME)
    dev = lab.examples("dev")
    parts = []
    for start in range(0, len(dev), common.PASS_SIZE):
        ids = list(range(start, min(start + common.PASS_SIZE, len(dev))))
        r = lab.forward("dev", ids, note=f"j6 FINAL full-dev eval of {EDIT_NAME} ({kind} beta={beta} tau={tau}) part {start // common.PASS_SIZE}")
        parts.append(r["logp"])
        print("dev part", start, json.dumps(r["summary"]), "left", r["passes_left"], flush=True)
    logp = np.concatenate(parts)
    labels = np.array([e["label"] for e in dev])
    np.save(OUT / "dev_logp.npy", logp)
    final = {"edit": EDIT_NAME, "n": int(len(labels)), "dev_acc": float((logp.argmax(1) == labels).mean()),
             "dev_nll": float(-logp[np.arange(len(labels)), labels].mean()),
             "pred_hist": (np.bincount(logp.argmax(1), minlength=5) / len(labels)).round(3).tolist(), **choice}
    print("FINAL", json.dumps(final), flush=True)
    (OUT / "final.json").write_text(json.dumps(final, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
