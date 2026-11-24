"""Job 4: finer neutral-mix sweep (train idx of job 1), pick the best of all neutral-mix variants seen,
fit a single logit temperature on its train logp, write the final edit file, then measure full dev
for (a) the final edit and (b) swap + temperature only, both loaded from edit files via Lab(edit=...)."""
import json
from pathlib import Path

import numpy as np

from wsp import config
from wsp.harness import Lab

HERE = Path(__file__).parent
OUT = HERE / "job4"
CHUNK = 512
NEW_MIXES = (0.35, 0.4, 0.45)
PRIOR = {0.0: ("job1", "train_logp_swap.npy"), 0.25: ("job3", "train_logp_N_neg0.25.npy"),
         0.5: ("job3", "train_logp_N_neg0.5.npy")}
TEMPS = np.round(np.arange(0.1, 1.01, 0.02), 2)


def tid(tok, w):
    ids = tok.encode(w, add_special_tokens=False)
    assert len(ids) == 1
    return ids[0]


def acc_nll(logp, labels):
    return float((logp.argmax(1) == labels).mean()), float(-logp[np.arange(len(labels)), labels].mean())


def best_temp(logp, labels):
    def nll(t):
        z = logp * t
        z = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
        return -z[np.arange(len(labels)), labels].mean()
    return float(min(TEMPS, key=nll))


def write_edit(name, mix, scale):
    src = (HERE / "edit_template.py.md").read_text()
    src = src.replace("__NEUTRAL_MIX__", repr(mix)).replace("__LOGIT_SCALE__", repr(scale))
    for d in (config.EDITS_DIR, OUT):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.py").write_text(src)


def full_dev(lab, note):
    n = len(lab.examples("dev"))
    logp = np.concatenate([lab.forward("dev", list(range(s, min(s + CHUNK, n))), note=note)["logp"]
                           for s in range(0, n, CHUNK)])
    labels = np.array([e["label"] for e in lab.examples("dev")])
    return logp, acc_nll(logp, labels)


def main():
    OUT.mkdir(exist_ok=True)
    tr_idx = np.load(HERE / "job1" / "train_idx.npy").tolist()
    lab = Lab(edit="swap_pn")
    tok = lab.tok
    emb = lab.model.get_input_embeddings().weight
    neu, pos = tid(tok, " neutral"), tid(tok, " positive")
    orig = emb[neu].clone()
    labels = np.array([lab.examples("train")[i]["label"] for i in tr_idx])
    logps = {m: np.load(HERE / d / f) for m, (d, f) in PRIOR.items()}
    for m in NEW_MIXES:
        emb[neu] = (1 - m) * orig + m * emb[pos]  # emb[pos] holds the original ' negative' vector
        r = lab.forward("train", tr_idx, note=f"swap_pn + neutral mix {m}")
        emb[neu] = orig
        logps[m] = r["logp"]
        np.save(OUT / f"train_logp_mix{m}.npy", r["logp"])
    table = {m: acc_nll(lp, labels) for m, lp in sorted(logps.items())}
    for m, v in table.items():
        print("MIX", m, v)
    best = max(table, key=lambda m: table[m][0])
    t_best, t_swap = best_temp(logps[best], labels), best_temp(logps[0.0], labels)
    print("BEST", best, "temp", t_best, "swap-only temp", t_swap)
    del lab
    import torch; torch.cuda.empty_cache()
    res = {"mix_table": {str(k): v for k, v in table.items()}, "best_mix": best, "temp": t_best, "temp_swap": t_swap}
    runs = [("final_o1", best, t_best), ("swap_temp", 0.0, t_swap)] if best != 0.0 else [("final_o1", 0.0, t_swap)]
    for name, m, t in runs:
        write_edit(name, m, t)
        lab = Lab(edit=name)
        logp, (a, n) = full_dev(lab, f"full dev {name} mix={m} scale={t}")
        np.save(OUT / f"dev_logp_{name}.npy", logp)
        print("DEV", name, m, t, a, n)
        res[name] = {"mix": m, "scale": t, "dev_acc": a, "dev_nll": n}
        del lab; torch.cuda.empty_cache()
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
