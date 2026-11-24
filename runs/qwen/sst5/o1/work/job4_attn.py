"""Job 4: answer-head mechanics (1 train pass, eager attention) + weights needed for an offline
attention simulator; plus 2 screening passes of option-word embedding scaling."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, WORK  # noqa: E402

from wsp.harness import Lab  # noqa: E402
from wsp.scoring import render_prompts  # noqa: E402

ATTN_LAYERS = (21, 22, 23, 24, 26)
XHAT_LAYER = 22
WEIGHT_LAYERS = (22, 26)
N_ROLES = 4  # letter, dot, last content token, newline
SCREEN = slice(1024, 1536)
NEUTRAL_ID = 20628
EMB_SCALES = (1.5, 2.0)
BATCH = 64


def option_positions(tok, prompt):
    """Per option: offsets from the end of letter, dot, last content token, newline (index = T - off)."""
    ids = tok.encode(prompt)
    toks = [tok.decode([i]) for i in ids]
    i = toks.index("Options") + 2
    T, out = len(ids), []
    for _ in range(5):
        j = i + 2
        while "\n" not in toks[j]:
            j += 1
        out.append([T - i, T - (i + 1), T - (j - 1), T - j])
        i = j + 1
    return out


def capture_pass(lab, idx):
    tok = lab.tok
    lab.model.config._attn_implementation = "eager"
    exs = [lab.examples("train")[i] for i in idx]
    offs = np.array([option_positions(tok, p) for p in render_prompts(tok, exs)])  # (n,5,4)
    store = {k: [] for k in ("attn", "q", "k", "v", "x")}
    hooks = []
    for l in ATTN_LAYERS:
        att = lab.layers[l].self_attn
        hooks.append(att.register_forward_hook(lambda m, a, o, l=l: store["attn"].append((l, o[1][:, :, -1, :].float().cpu()))))
        hooks.append(att.q_proj.register_forward_hook(lambda m, a, o, l=l: store["q"].append((l, o[:, -1, :].float().cpu()))))
        hooks.append(att.k_proj.register_forward_hook(lambda m, a, o, l=l: store["k"].append((l, o.half().cpu()))))
        hooks.append(att.v_proj.register_forward_hook(lambda m, a, o, l=l: store["v"].append((l, o.half().cpu()))))
    hooks.append(lab.layers[XHAT_LAYER].input_layernorm.register_forward_hook(
        lambda m, a, o: store["x"].append((XHAT_LAYER, o.half().cpu()))))
    r = lab.forward("train", idx, note="base train: answer-head attention, q/k/v at option tokens (eager)", batch_size=BATCH)
    for h in hooks:
        h.remove()
    lab.model.config._attn_implementation = "sdpa"
    n = len(idx)
    out = {}
    for key in store:
        by_layer = {}
        for l, t in store[key]:
            by_layer.setdefault(l, []).append(t)
        for l, ts in by_layer.items():
            rows_opt, rows_last = [], []
            for b, t in enumerate(ts):
                Tp = t.shape[-1] if key == "attn" else t.shape[1]
                for e in range(t.shape[0]):
                    o = offs[b * BATCH + e]
                    pos = torch.as_tensor(Tp - o)
                    if key == "attn":
                        rows_opt.append(t[e][:, pos].numpy())          # (12,5,4)
                        rows_last.append(np.stack([t[e][:, :Tp - o.max()].sum(-1).numpy(),
                                                   t[e][:, Tp - o.min() + 1:].sum(-1).numpy()], -1))
                    elif key == "q":
                        rows_opt.append(t[e].numpy())
                    else:
                        rows_opt.append(t[e][pos].numpy())             # (5,4,dim)
                        if key == "x":
                            rows_last.append(t[e][Tp - 1].numpy())
            out[f"{key}{l}"] = np.stack(rows_opt).astype(np.float16 if key in ("k", "v", "x") else np.float32)
            if rows_last:
                out[f"{key}{l}_rest"] = np.stack(rows_last).astype(np.float16 if key == "x" else np.float32)
    out["offsets"] = offs
    return r, out


def save_weights(lab):
    m = lab.model
    np.save(CACHE / "w_letters.npy", m.lm_head.weight[lab.letters].float().cpu().numpy())
    np.save(CACHE / "w_final_norm.npy", m.model.norm.weight.float().cpu().numpy())
    for l in WEIGHT_LAYERS:
        a = m.model.layers[l].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            p = getattr(a, name)
            np.save(CACHE / f"w{l}_{name}.npy", p.weight.half().cpu().numpy())
            if p.bias is not None:
                np.save(CACHE / f"w{l}_{name}_b.npy", p.bias.float().cpu().numpy())
        np.save(CACHE / f"w{l}_ln.npy", m.model.layers[l].input_layernorm.weight.float().cpu().numpy())
    print("ROPE", json.dumps({"theta": getattr(m.config, "rope_theta", None),
                              "rope_scaling": getattr(m.config, "rope_scaling", None)}))


def screen_embedding_scale(lab, idx):
    emb = lab.model.model.embed_tokens.weight
    orig = emb[NEUTRAL_ID].clone()
    for s in EMB_SCALES:
        emb[NEUTRAL_ID] = orig * s
        r = lab.forward("train", idx, note=f"screen: ' neutral' embedding x{s}")
        np.save(WORK / f"j4_embscale{s}_logp.npy", r["logp"])
        print("EMB", s, json.dumps(r["summary"]))
    emb[NEUTRAL_ID] = orig


def main():
    lab = Lab(edit=None)
    idx_all = np.load(WORK / "train_idx_job1.npy").tolist()
    CACHE.mkdir(parents=True, exist_ok=True)
    save_weights(lab)
    r, out = capture_pass(lab, idx_all[:512])
    print("CAP", json.dumps(r["summary"]))
    for k, v in out.items():
        np.save(CACHE / f"j4_{k}.npy", v)
    for l in ATTN_LAYERS:
        a = out[f"attn{l}"]
        print("ATTN", l, "per-head mass on [letter,dot,content,newline]:",
              json.dumps(a.sum(1).mean(0).round(3).tolist()), "pre/post", json.dumps(out[f"attn{l}_rest"].mean(0).round(3).tolist()))
    with torch.no_grad():
        screen_embedding_scale(lab, idx_all[SCREEN])
    del lab
    torch.cuda.empty_cache()
    fallback = Lab(edit="temp_v1")
    dev_n = len(fallback.examples("dev"))
    rs = [fallback.forward("dev", list(range(s, min(s + 512, dev_n))), note="full dev temp_v1 (fallback edit)")
          for s in range(0, dev_n, 512)]
    logp = np.concatenate([x["logp"] for x in rs])
    np.save(WORK / "dev_temp_v1_logp.npy", logp)
    from wsp.metrics import summarize
    print("DEV temp_v1", json.dumps(summarize(logp, fallback.examples("dev"))))
    print("PASSES_LEFT", rs[-1]["passes_left"])


if __name__ == "__main__":
    main()
