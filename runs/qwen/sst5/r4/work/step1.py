"""Job 1: base model on train[0:3072] with all-layer capture + weight metadata."""
import json
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r4")
CACHE = Path("wsp_work/sst5_r4_cache")
CACHE.mkdir(parents=True, exist_ok=True)
N_PASSES = 6
PASS_SIZE = 512
LAYERS = tuple(range(28)) + (-1,)
LABEL_WORDS = [" very", " negative", " neutral", " positive", "very", "negative", "neutral", "positive", ".", "\n", "assistant"]

lab = Lab(edit=None)
m = lab.model
print(m.config)
for name, p in m.named_parameters():
    if "layers.0." in name or "layers" not in name:
        print(name, tuple(p.shape), p.dtype)
print("tied", m.get_input_embeddings().weight.data_ptr() == m.get_output_embeddings().weight.data_ptr())
print("letters", lab.letters.tolist())
tok_info = {w: lab.tok.encode(w, add_special_tokens=False) for w in LABEL_WORDS}
print(tok_info)
ex = lab.examples("train")[0]
from wsp.scoring import render_prompts
pr = render_prompts(lab.tok, [ex])[0]
ids = lab.tok(pr)["input_ids"]
print(repr(pr))
print([(i, lab.tok.decode([i])) for i in ids])

E = m.get_input_embeddings().weight
meta = {
    "letters": lab.letters.tolist(), "tok_info": tok_info,
}
(WORK / "meta.json").write_text(json.dumps(meta))
np.save(WORK / "E_letters.npy", E[lab.letters].float().cpu().numpy())
word_ids = sorted({i for v in tok_info.values() for i in v})
np.save(WORK / "E_words.npy", E[word_ids].float().cpu().numpy())
(WORK / "word_ids.json").write_text(json.dumps(word_ids))
np.save(WORK / "E_mean.npy", E.float().mean(0).cpu().numpy())
norms = {"final": m.model.norm.weight.float().cpu().numpy()}
for i, layer in enumerate(lab.layers):
    norms[f"ln1_{i}"] = layer.input_layernorm.weight.float().cpu().numpy()
    norms[f"ln2_{i}"] = layer.post_attention_layernorm.weight.float().cpu().numpy()
np.savez(WORK / "norm_weights.npz", **norms)

logps, hids = [], []
for p in range(N_PASSES):
    idx = list(range(p * PASS_SIZE, (p + 1) * PASS_SIZE))
    r = lab.forward("train", idx, note=f"base train {idx[0]}-{idx[-1]} all-layer capture", capture_layers=LAYERS)
    print(p, json.dumps(r["summary"]), r["passes_left"], flush=True)
    logps.append(r["logp"])
    hids.append(np.stack([r["hidden"][l] for l in LAYERS], axis=1).astype(np.float16))
np.save(WORK / "base_train_logp.npy", np.concatenate(logps))
np.save(CACHE / "base_train_hidden.npy", np.concatenate(hids))
print("done")
