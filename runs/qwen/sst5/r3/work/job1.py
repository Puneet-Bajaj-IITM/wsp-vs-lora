"""Job 1: base behaviour on 1536 train items with residual captures, plus option-token embedding probes."""
import numpy as np
import torch

from wsp.harness import Lab
from labkit import CACHE, PASS_SIZE, describe, save_json, temporary

CAPTURE = (6, 10, 14, 18, 21, 24, 26, 27, -1)
N_BASE_PASSES = 4
ATTN_GROUPS = {"late": range(20, 28), "mid": range(10, 20)}
ATTN_SCALES = (0.8, 1.25)
OPTION_WORDS = (" very", " neutral", " negative", " positive")
PROBE_SCALES = {" very": (0.6, 1.5), " neutral": (0.6, 1.5)}

lab = Lab()
tok, model = lab.tok, lab.model
CACHE.mkdir(parents=True, exist_ok=True)
train = lab.examples("train")

ids = {w: tok.encode(w, add_special_tokens=False) for w in OPTION_WORDS}
print("option token ids", ids)
print("sample tokens", tok.convert_ids_to_tokens(tok.encode(train[0]["user_message"]))[-45:])
emb = model.get_input_embeddings().weight
letters = lab.letters
print("emb row norms letters", emb[letters].float().norm(dim=1).tolist())
print("emb row norms option words", {w: emb[i[0]].float().norm().item() for w, i in ids.items()})
print("letter gram", (emb[letters].float() @ emb[letters].float().T).cpu().numpy().round(3))
print("final norm weight mean/std", model.model.norm.weight.float().mean().item(), model.model.norm.weight.float().std().item())
print("param names sample", [n for n, _ in model.named_parameters()][:16])
print("n params", sum(p.numel() for p in model.parameters()))

results = []
logps = []
for k in range(N_BASE_PASSES):
    idx = list(range(k * PASS_SIZE, (k + 1) * PASS_SIZE))
    r = lab.forward("train", idx, note=f"base train block {k} with captures", capture_layers=CAPTURE)
    logps.append(r["logp"])
    for layer, h in r["hidden"].items():
        np.save(CACHE / f"base_h{layer}_b{k}.npy", h.astype(np.float16))
    results.append(describe(f"base b{k}", r["logp"], r["examples"]))
np.save(CACHE / "base_logp_train.npy", np.concatenate(logps))
np.save(CACHE / "letter_rows.npy", emb[letters].float().cpu().numpy())
np.save(CACHE / "option_word_rows.npy", torch.stack([emb[ids[w][0]] for w in OPTION_WORDS]).float().cpu().numpy())

idx0 = list(range(PASS_SIZE))
for word, scales in PROBE_SCALES.items():
    for s in scales:
        def fn(m, word=word, s=s):
            m.get_input_embeddings().weight[ids[word][0]] *= s
        with temporary(model, fn):
            r = lab.forward("train", idx0, note=f"probe: scale emb row '{word}' x{s}")
            np.save(CACHE / f"probe_{word.strip()}_{s}_logp.npy", r["logp"])
            results.append(describe(f"scale{word} x{s}", r["logp"], r["examples"]))
for group, layer_ids in ATTN_GROUPS.items():
    for s in ATTN_SCALES:
        def fn(m, layer_ids=layer_ids, s=s):
            for i in layer_ids:
                m.model.layers[i].self_attn.o_proj.weight *= s
        with temporary(model, fn):
            r = lab.forward("train", idx0, note=f"probe: scale attn o_proj {group} layers x{s}")
            np.save(CACHE / f"probe_attn_{group}_{s}_logp.npy", r["logp"])
            results.append(describe(f"attn {group} x{s}", r["logp"], r["examples"]))
save_json("job1_results.json", results)
