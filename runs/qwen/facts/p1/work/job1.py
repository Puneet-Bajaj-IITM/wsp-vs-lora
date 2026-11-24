"""Job 1: base captures (train all, dev all, preserve 512) at every layer + weight metadata."""
import json
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab
from wsp.tasks import facts

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "facts_p1_cache"
CACHE.mkdir(exist_ok=True)
ALL_LAYERS = tuple(range(28))
PASS = 512

lab = Lab(edit=None)
tok, model = lab.tok, lab.model
print(model.config)
for n, p in model.named_parameters():
    if "layers.0." in n or "layers" not in n:
        print(n, tuple(p.shape))


def capture(split, idx, tag):
    hid, logps = [], []
    for s in range(0, len(idx), PASS):
        r = lab.forward(split, idx[s:s + PASS], note=f"job1 base capture {tag}", capture_layers=ALL_LAYERS)
        hid.append(np.stack([r["hidden"][l] for l in ALL_LAYERS], 1).astype(np.float16))
        logps.append(r["logp"])
        print(tag, s, r["summary"], r["passes_left"], flush=True)
    np.save(CACHE / f"h_{tag}.npy", np.concatenate(hid))
    np.save(WORK / f"logp_base_{tag}.npy", np.concatenate(logps))


capture("train", list(range(2700)), "train")
capture("dev", list(range(900)), "dev")
capture("preserve", list(range(512)), "pres")

# ---- weight metadata (free) ----
meta = {"letters": lab.letters.tolist()}
people = sorted({e["pid"] for e in lab.examples("train")})
ex0 = lab.examples("train")[0]
from wsp.scoring import render_prompts
p0 = render_prompts(tok, [ex0])[0]
meta["prompt0"] = p0
meta["prompt0_ids"] = tok(p0)["input_ids"]
meta["prompt0_toks"] = tok.convert_ids_to_tokens(meta["prompt0_ids"])
import random
rng = random.Random(facts.SEED)
names = facts.invent_people(rng)
meta["names"] = names
meta["name_ids"] = [tok.encode(" " + n, add_special_tokens=False) for n in names]
meta["value_ids"] = {a: [[tok.encode(pre + v, add_special_tokens=False) for pre in (" ", "")] for v in vs]
                     for a, vs in facts.ATTRIBUTES.items()}
(WORK / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps({k: meta[k] for k in ("letters", "prompt0_toks", "value_ids")}))
print(meta["name_ids"][:5], [tok.convert_ids_to_tokens(i) for i in meta["name_ids"][:5]])

E = model.get_input_embeddings().weight.float().cpu().numpy()
ids = sorted({i for a in meta["value_ids"].values() for v in a for pre in v for i in pre} | set(meta["letters"])
             | {i for n in meta["name_ids"] for i in n} | set(meta["prompt0_ids"]))
np.save(WORK / "emb_ids.npy", np.array(ids))
np.save(WORK / "emb_rows.npy", E[ids].astype(np.float32))
norms = {"final": model.model.norm.weight.float().cpu().numpy()}
stats = {}
for l, layer in enumerate(model.model.layers):
    norms[f"ln1_{l}"] = layer.input_layernorm.weight.float().cpu().numpy()
    norms[f"ln2_{l}"] = layer.post_attention_layernorm.weight.float().cpu().numpy()
    m = layer.mlp
    stats[f"gate_{l}"] = m.gate_proj.weight.float().norm(dim=1).cpu().numpy()
    stats[f"up_{l}"] = m.up_proj.weight.float().norm(dim=1).cpu().numpy()
    stats[f"down_{l}"] = m.down_proj.weight.float().norm(dim=0).cpu().numpy()
    a = layer.self_attn
    print(l, "biases", [(k, getattr(getattr(a, k), "bias") is not None) for k in ("q_proj", "k_proj", "v_proj", "o_proj")],
          [(k, getattr(getattr(m, k), "bias") is not None) for k in ("gate_proj", "up_proj", "down_proj")])
np.savez(WORK / "norm_gains.npz", **norms)
np.savez(WORK / "mlp_norms.npz", **stats)
print("E norm stats", np.linalg.norm(E, axis=1).mean())
