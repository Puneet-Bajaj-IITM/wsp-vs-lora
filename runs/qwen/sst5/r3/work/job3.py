"""Job 3: steering variants + label-free letter debias in the letter rows + calibration; validate; full dev."""
import numpy as np
import torch

from wsp import config
from wsp.harness import Lab, apply_edit
from debias_edit import letter_debias, letter_row_deltas, log_softmax, write_edit
from labkit import CACHE, PASS_SIZE, WORK, describe, save_json
from steer import add_biases, best_temperature, restore, solve_biases

A4, A6, S4 = (16, 17, 18, 19), (14, 15, 16, 17, 18, 19), (17, 18, 19, 20)
MID_LAYERS = tuple(range(10, 20))
MID_SCALE = 1.25
NO_SCALE = 1.0
CONFIGS = {
    "A4x2": (A4, 2.0, NO_SCALE), "A6x2": (A6, 2.0, NO_SCALE), "A6x3": (A6, 3.0, NO_SCALE),
    "S4x2": (S4, 2.0, NO_SCALE), "A4x2_mid": (A4, 2.0, MID_SCALE), "A4x1.4_mid": (A4, 1.4, MID_SCALE),
}
SOURCE_LAYER = 18
DIR_INDEX = 1  # rows of dirs_L*.npy: mild, mildperp, neuperp, pol
EXPLORE_BLOCK, CONFIRM_BLOCK, VERIFY_BLOCK = 2, 3, 1
TOP_K = 2
EDIT_NAME = "steer_mild_debias"
DEV_SIZE = 1101

lab = Lab()
model, tok = lab.model, lab.tok
print("tied embeddings:", model.get_output_embeddings().weight is model.get_input_embeddings().weight)
CACHE.mkdir(parents=True, exist_ok=True)
train = lab.examples("train")
direction = np.load(WORK / f"dirs_L{SOURCE_LAYER}.npy").astype(np.float64)[DIR_INDEX]
block = lambda k: list(range(k * PASS_SIZE, (k + 1) * PASS_SIZE))
labels_of = lambda k: np.array([e["label"] for e in train[k * PASS_SIZE:(k + 1) * PASS_SIZE]])
acc = lambda logp, labels: float((logp.argmax(1) == labels).mean())
results, store, deltas_of = [], {}, {}


def apply_config(name):
    layers, alpha, scale = CONFIGS[name]
    deltas, info = solve_biases(model, layers, alpha * direction)
    deltas_of[name] = deltas
    backups = add_biases(model, layers, deltas)
    if scale != NO_SCALE:
        with torch.no_grad():
            for i in MID_LAYERS:
                weight = model.model.layers[i].self_attn.o_proj.weight
                backups.append((weight, weight.detach().clone()))
                weight.mul_(scale)
    return backups, info


def run(name, k):
    backups, info = apply_config(name)
    try:
        r = lab.forward("train", block(k), note=f"{name} train block {k} (capture final)", capture_layers=(-1,))
    finally:
        restore(backups)
    store[(name, k)] = (r["logp"], r["hidden"][-1].astype(np.float64))
    np.save(CACHE / f"j3_{name}_b{k}.npy", r["logp"])
    print(info, flush=True)
    res = describe(f"{name} b{k}", r["logp"], r["examples"])
    bias = letter_debias(r["logp"])
    res.update(info, name=name, block=k, letter_bias=bias.tolist(), debiased_acc=acc(r["logp"] + bias, labels_of(k)))
    print(f"   self-debiased acc {res['debiased_acc']:.4f} bias {bias.round(2)}", flush=True)
    results.append(res)
    save_json("job3_results.json", results)
    return res["debiased_acc"]


explore = {name: run(name, EXPLORE_BLOCK) for name in CONFIGS}
top = sorted(explore, key=explore.get, reverse=True)[:TOP_K]
pooled_score, pooled = {}, {}
for name in top:
    run(name, CONFIRM_BLOCK)
    logp = np.concatenate([store[(name, k)][0] for k in (EXPLORE_BLOCK, CONFIRM_BLOCK)])
    hidden = np.concatenate([store[(name, k)][1] for k in (EXPLORE_BLOCK, CONFIRM_BLOCK)])
    labels = np.concatenate([labels_of(EXPLORE_BLOCK), labels_of(CONFIRM_BLOCK)])
    bias = letter_debias(logp)
    pooled[name] = (logp, hidden, labels, bias)
    pooled_score[name] = acc(logp + bias, labels)
print("explore", explore, "pooled", pooled_score, flush=True)
chosen = max(pooled_score, key=pooled_score.get)
logp, hidden, labels, bias = pooled[chosen]
mean_hidden = hidden.mean(0)
proj = hidden @ mean_hidden / (mean_hidden @ mean_hidden)
tau, tau_nll = best_temperature(log_softmax(logp + bias), labels)
print(f"chosen {chosen}; letter bias {bias.round(3)}; proj sd {proj.std():.4f}; logit scale {tau:.2f} (train nll {tau_nll:.4f})", flush=True)

layers, alpha, scale = CONFIGS[chosen]
attn_layers = MID_LAYERS if scale != NO_SCALE else ()
row_deltas = letter_row_deltas(bias, mean_hidden)
write_edit(config.EDITS_DIR, EDIT_NAME, layers, deltas_of[chosen], attn_layers, scale, row_deltas, tau,
           f"Steer the answer-position sentiment code toward mild (mean-difference direction at layer {SOURCE_LAYER}, "
           f"x{alpha:g}) through v_proj biases of layers {layers}; o_proj scale {scale:g} on layers {attn_layers}; "
           f"label-free letter debias written into the A-E embedding rows; final-norm gain x{tau:.2f}.")
before = {n: p.detach().clone() for n, p in model.named_parameters()
          if "v_proj.bias" in n or n.endswith("norm.weight") or "embed_tokens" in n or "lm_head" in n or (attn_layers and "o_proj" in n)}
apply_edit(model, tok, EDIT_NAME)
changed = int(sum((p != before[n]).sum().item() for n, p in model.named_parameters() if n in before))
total = int(sum(p.numel() for p in model.parameters()))
print("weights changed", changed, "of", total, flush=True)

r = lab.forward("train", block(VERIFY_BLOCK), note=f"{EDIT_NAME} verify on train block {VERIFY_BLOCK}")
verify = describe(f"VERIFY {EDIT_NAME} b{VERIFY_BLOCK}", r["logp"], r["examples"])
dev_logp, dev_ex = [], []
for start in range(0, DEV_SIZE, PASS_SIZE):
    r = lab.forward("dev", list(range(start, min(start + PASS_SIZE, DEV_SIZE))), note=f"{EDIT_NAME} dev from {start}")
    dev_logp.append(r["logp"])
    dev_ex += r["examples"]
dev_logp = np.concatenate(dev_logp)
np.save(CACHE / f"{EDIT_NAME}_dev_logp.npy", dev_logp)
final = describe(f"DEV {EDIT_NAME}", dev_logp, dev_ex)
final.update(chosen=chosen, tau=tau, letter_bias=bias.tolist(), weights_changed=changed, total_weights=total, verify=verify)
results.append(final)
save_json("job3_results.json", results)
