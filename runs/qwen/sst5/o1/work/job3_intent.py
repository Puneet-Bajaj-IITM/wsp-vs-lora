"""Job 3: capture MLP inputs (2 train passes), build gated centroid->intent neuron edits, screen them on a
held-out train chunk (1 pass each), pick a temperature, then measure the best edit on full dev (3 passes)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, chunks  # noqa: E402

from wsp import config  # noqa: E402
from wsp.harness import Lab  # noqa: E402

MLP_LAYERS = (17, 21)
N_DIAG = 1024
EVAL_SLICE = slice(1024, 1536)
OFFSETS = np.arange(2, 66)
N_OTHER = 8
GATE_TARGET = 6.0
SEED = 3
TEMPS = np.round(np.arange(1.0, 8.01, 0.25), 2)
CANDIDATES = [(21, "D", 1.0), (21, "D", 2.5), (21, "C", 1.0), (21, "C", 2.5),
              (17, "D", 1.0), (17, "D", 2.5), (17, "C", 2.5)]
EDITS = config.EDITS_DIR
TEMPLATE = WORK / "edit_template.py"


def class_of(ex):
    return ex["shown_option_ids"][ex["label"]]


def class_logp(logp, exs):
    out = np.zeros_like(logp)
    for i, e in enumerate(exs):
        out[i, e["shown_option_ids"]] = logp[i]
    return out


def capture(lab, idx):
    rng = np.random.default_rng(SEED)
    store = {l: {"last": [], "other": []} for l in MLP_LAYERS}

    def hook(l):
        def fn(m, a, o):
            x = o.float().cpu()
            store[l]["last"].append(x[:, -1, :])
            offs = torch.as_tensor(rng.choice(OFFSETS, (x.shape[0], N_OTHER)))
            store[l]["other"].append(x[torch.arange(x.shape[0])[:, None], x.shape[1] - offs].reshape(-1, x.shape[2]))
        return fn
    hs = [lab.layers[l].post_attention_layernorm.register_forward_hook(hook(l)) for l in MLP_LAYERS]
    rs = [lab.forward("train", c, note="base train: capture MLP-17/21 inputs for centroids + gate") for c in chunks(idx)]
    for h in hs:
        h.remove()
    return {l: {k: torch.cat(v).numpy() for k, v in s.items()} for l, s in store.items()}, rs


def build_layer(X_last, X_oth, y):
    mu = X_last.mean(0)
    C = np.stack([X_last[y == c].mean(0) - mu for c in range(5)])
    W = C / (C ** 2).sum(1, keepdims=True)
    g = X_last.mean(0) - X_oth.mean(0)
    g /= np.linalg.norm(g)
    v = X_oth.mean(0) / np.linalg.norm(X_oth.mean(0))
    lam = (X_last @ g).mean() / 2 + (X_oth @ g).mean() / 2
    lam /= np.concatenate([X_last @ v, X_oth @ v]).mean()
    u = g - lam * v
    a = GATE_TARGET / (X_last @ u).mean()
    gate = a * u
    z_last, z_oth = X_last @ gate, X_oth @ gate
    silu = lambda z: z / (1 + np.exp(-z))
    G, P = silu(z_last).mean(), z_last.mean()
    diag = {"z_last_mean": float(P), "z_last_std": float(z_last.std()), "z_last_min": float(z_last.min()),
            "z_oth_mean": float(z_oth.mean()), "frac_oth_pos": float((z_oth > 0).mean()),
            "silu_oth_over_last": float(np.abs(silu(z_oth)).mean() / G),
            "s_diag_by_class": [float(((X_last[y == c] - mu) @ W[c]).mean()) for c in range(5)]}
    return {"mu": mu, "W": W, "gate": gate, "G": G, "P": P, "diag": diag}


def make_consts(layer, info, V, kappa, temperature=1.0):
    ups = np.concatenate([info["W"], (info["gate"] / info["P"])[None]], 0)
    const = -(info["W"] @ info["mu"])[:, None] * V
    downs = kappa * np.concatenate([V, const.sum(0, keepdims=True)], 0) / info["G"]
    return {"layer": layer, "gate": torch.tensor(info["gate"], dtype=torch.float32),
            "ups": torch.tensor(ups, dtype=torch.float32), "downs": torch.tensor(downs, dtype=torch.float32),
            "temperature": float(temperature)}


def write_edit(name, consts):
    EDITS.mkdir(parents=True, exist_ok=True)
    (EDITS / f"{name}.py").write_text(TEMPLATE.read_text())
    torch.save(consts, EDITS / f"{name}_consts.pt")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, EDITS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_candidate(lab, name, layer, idx):
    mlp = lab.layers[layer].mlp
    mod = load_module(name)
    sel = mod.pick_neurons(mlp, 6)
    saved = (mlp.gate_proj.weight[sel].clone(), mlp.up_proj.weight[sel].clone(),
             mlp.down_proj.weight[:, sel].clone(), lab.model.model.norm.weight.clone())
    with torch.no_grad():
        mod.apply(lab.model, lab.tok)
    r = lab.forward("train", idx, note=f"screen {name} on held-out train chunk")
    with torch.no_grad():
        mlp.gate_proj.weight[sel], mlp.up_proj.weight[sel] = saved[0], saved[1]
        mlp.down_proj.weight[:, sel] = saved[2]
        lab.model.model.norm.weight.copy_(saved[3])
    return r


def best_temperature(logp, labels):
    nll = []
    for T in TEMPS:
        z = logp / T
        z = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll.append(-z[np.arange(len(labels)), labels].mean())
    return float(TEMPS[int(np.argmin(nll))]), float(min(nll))


def main():
    lab = Lab(edit=None)
    train = lab.examples("train")
    idx_all = np.load(WORK / "train_idx_job1.npy").tolist()
    diag_idx, eval_idx = idx_all[:N_DIAG], idx_all[EVAL_SLICE]
    caps, _ = capture(lab, diag_idx)
    y = np.array([class_of(train[i]) for i in diag_idx])
    V = {"D": np.load(WORK / "intent_D_h20.npy"), "C": np.load(WORK / "class_C_h20.npy")}
    infos = {}
    for l in MLP_LAYERS:
        infos[l] = build_layer(caps[l]["last"], caps[l]["other"], y)
        print("GATE", l, json.dumps(infos[l]["diag"]))

    base_logp = np.load(WORK / "base_train_logp.npy")[EVAL_SLICE]
    eval_exs = [train[i] for i in eval_idx]
    ye = np.array([class_of(e) for e in eval_exs])
    print("BASE_EVAL acc %.4f" % (class_logp(base_logp, eval_exs).argmax(1) == ye).mean())
    results = {}
    for layer, vk, kappa in CANDIDATES:
        name = f"cand_L{layer}{vk}_k{kappa}".replace(".", "p")
        write_edit(name, make_consts(layer, infos[layer], V[vk], kappa))
        r = run_candidate(lab, name, layer, eval_idx)
        cl = class_logp(r["logp"], eval_exs)
        conf = np.zeros((5, 5), int)
        for t, p in zip(ye, cl.argmax(1)):
            conf[t, p] += 1
        results[name] = {"acc": r["summary"]["accuracy"], "nll": r["summary"]["nll"], "layer": layer, "V": vk, "kappa": kappa}
        np.save(WORK / f"j3_{name}_logp.npy", r["logp"])
        print("CAND", name, r["summary"]["accuracy"], round(r["summary"]["nll"], 3), "pred_class_hist",
              np.bincount(cl.argmax(1), minlength=5).tolist(), "conf", conf.tolist())
    best = max(results, key=lambda k: results[k]["acc"])
    b = results[best]
    T, nllT = best_temperature(np.load(WORK / f"j3_{best}_logp.npy"), np.array([e["label"] for e in eval_exs]))
    print("BEST", best, b, "T", T, "nll@T", round(nllT, 3))
    final = "intent_v1"
    write_edit(final, make_consts(b["layer"], infos[b["layer"]], V[b["V"]], b["kappa"], T))
    json.dump({"results": results, "best": best, "T": T}, open(WORK / "j3_results.json", "w"), indent=1)
    del lab
    torch.cuda.empty_cache()
    lab = Lab(edit=final)
    dev_n = len(lab.examples("dev"))
    rs = [lab.forward("dev", c, note=f"full dev {final}") for c in chunks(list(range(dev_n)))]
    logp = np.concatenate([r["logp"] for r in rs])
    np.save(WORK / f"dev_{final}_logp.npy", logp)
    from wsp.metrics import summarize
    print("DEV", final, json.dumps(summarize(logp, lab.examples("dev"))))
    print("PASSES_LEFT", rs[-1]["passes_left"])


if __name__ == "__main__":
    main()
