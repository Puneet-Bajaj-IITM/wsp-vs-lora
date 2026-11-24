"""Job 4: letter-balanced "very" direction from 3 fresh capture passes, screen on train A+B, write the final edit, full dev."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import editlib
from labutil import ARRAYS, NUM_CLASSES, PASS_SIZE, WORK, full_split, run_pass, shown_ids, true_sentiment
from wsp import config
from wsp.harness import Lab

TOP_K = 512
NUM_LETTERS = 5
CAPTURE_LAYERS = (18, 20)
CAPTURE_CHUNKS = (2, 3, 4)
SCREEN_CHUNKS = {"A": range(0, PASS_SIZE), "B": range(PASS_SIZE, 2 * PASS_SIZE)}
SWAP_BASELINE = {"A": 0.412109375, "B": 0.478515625}
MIN_GAIN = 0.005
MIN_CELL = 5
MIRROR = np.arange(NUM_CLASSES)[::-1]
VERY_OF = {0: 1, 4: 3}  # very class -> its plain neighbour (read order)
# (tag, layer, intensity scale, neutral scale)
VARIANTS = [
    ("L20_i2", 20, 2.0, 0.0),
    ("L20_i3.5", 20, 3.5, 0.0),
    ("L20_i5", 20, 5.0, 0.0),
    ("L18_i3.5", 18, 3.5, 0.0),
    ("L20_i3.5_n1", 20, 3.5, 1.0),
]
TEMPERATURE_GRID = np.round(np.arange(0.1, 1.01, 0.05), 2)
FINAL_NAME = "swap_steer_temp"
SWAP_TEMP_NAME = "swap_temp"

lab = Lab()
tok, model = lab.tok, lab.model
original_down = {l: model.model.layers[l].mlp.down_proj.weight.clone() for l in CAPTURE_LAYERS}
editlib.swap_rows(model, tok)

# ---- captures on fresh train chunks ----
hidden = {l: [] for l in CAPTURE_LAYERS}
pred_read, pred_letter, sent, label_letter = [], [], [], []
for chunk in CAPTURE_CHUNKS:
    r = run_pass(lab, "train", range(chunk * PASS_SIZE, (chunk + 1) * PASS_SIZE), f"swap capture chunk {chunk}",
                 f"swap_cap_chunk{chunk}", CAPTURE_LAYERS)
    for f in ARRAYS.glob(f"swap_cap_chunk{chunk}_h*.npy"):
        f.unlink()
    exs = r["examples"]
    letters = r["logp"].argmax(1)
    for l in CAPTURE_LAYERS:
        hidden[l].append(r["hidden"][l])
    pred_letter.append(letters)
    pred_read.append(MIRROR[shown_ids(exs)[np.arange(len(exs)), letters]])
    sent.append(true_sentiment(exs))
    label_letter.append(r["labels"])
hidden = {l: np.concatenate(v) for l, v in hidden.items()}
pred_read, pred_letter, sent, label_letter = map(np.concatenate, (pred_read, pred_letter, sent, label_letter))
print("capture n", len(sent), "pred class hist", np.bincount(pred_read, minlength=5), "pred letter hist", np.bincount(pred_letter, minlength=5))
print("very-pick rate by predicted letter", [round(float(np.isin(pred_read[pred_letter == k], [0, 4]).mean()), 3) for k in range(NUM_LETTERS)])

old_layers = json.loads((ARRAYS / "steer_layers.json").read_text())
old_pred = dict(zip(old_layers, np.load(ARRAYS / "steer_pred_means.npy").astype(np.float32)))
keys = dict(zip(old_layers, torch.tensor(np.load(ARRAYS / "steer_keys.npy").astype(np.float32))))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def balanced_difference(h, group_a, group_b, letter):
    """Mean over letters of (mean h | group_a, letter) - (mean h | group_b, letter): letter identity cancels."""
    diffs = [h[group_a & (letter == k)].mean(0) - h[group_b & (letter == k)].mean(0)
             for k in range(NUM_LETTERS) if (group_a & (letter == k)).sum() >= MIN_CELL and (group_b & (letter == k)).sum() >= MIN_CELL]
    return np.mean(diffs, axis=0), len(diffs)


directions = {}
for layer in CAPTURE_LAYERS:
    h = hidden[layer]
    parts = [balanced_difference(h, pred_read == very, pred_read == plain, pred_letter) for very, plain in VERY_OF.items()]
    intensity = np.mean([p[0] for p in parts], axis=0)
    neutral_parts = [balanced_difference(h, sent == 2, sent == plain, pred_letter) for plain in VERY_OF.values()]
    neutral = np.mean([p[0] for p in neutral_parts], axis=0)
    letter_means = np.stack([h[pred_letter == k].mean(0) for k in range(NUM_LETTERS)])
    letter_basis = np.linalg.svd(letter_means - letter_means.mean(0), full_matrices=False)[2][:NUM_LETTERS - 1]
    raw_norms = (np.linalg.norm(intensity), np.linalg.norm(neutral))
    letter_share = (np.linalg.norm(letter_basis @ unit(intensity)), np.linalg.norm(letter_basis @ unit(neutral)))
    intensity = intensity - letter_basis.T @ (letter_basis @ intensity)
    neutral = neutral - letter_basis.T @ (letter_basis @ neutral)
    old = 0.5 * ((old_pred[layer][0] - old_pred[layer][1]) + (old_pred[layer][4] - old_pred[layer][3]))
    directions[layer] = (intensity, neutral)
    print(f"layer {layer}: cells {[p[1] for p in parts]} raw norms {raw_norms} letter-subspace share {letter_share} "
          f"final norms {np.linalg.norm(intensity):.2f} {np.linalg.norm(neutral):.2f} cos(new,old intensity) {unit(intensity) @ unit(old):.3f} "
          f"old share in letter subspace {np.linalg.norm(letter_basis @ unit(old)):.3f} cos(int,neu) {unit(intensity) @ unit(neutral):.3f}", flush=True)


def steer_vector(layer, i_scale, n_scale):
    intensity, neutral = directions[layer]
    return torch.tensor((-i_scale * intensity + n_scale * neutral).astype(np.float32))


def apply_variant(layer, i_scale, n_scale):
    for l, saved in original_down.items():
        model.model.layers[l].mlp.down_proj.weight.copy_(saved)
    if layer is not None:
        editlib.rank_one_steer(model, layer, steer_vector(layer, i_scale, n_scale), keys[layer], TOP_K)


# ---- screen on train A and B ----
scores, logps = {}, {}
for tag, layer, i_scale, n_scale in VARIANTS:
    apply_variant(layer, i_scale, n_scale)
    results = {name: run_pass(lab, "train", idx, f"balanced steer {tag} on train {name}", f"s4_{tag}_{name}") for name, idx in SCREEN_CHUNKS.items()}
    scores[tag] = float(np.mean([r["summary"]["accuracy"] for r in results.values()]))
    logps[tag] = (np.concatenate([r["logp"] for r in results.values()]), np.concatenate([r["labels"] for r in results.values()]))
baseline = float(np.mean(list(SWAP_BASELINE.values())))
best = max(VARIANTS, key=lambda v: scores[v[0]])
use_steer = scores[best[0]] > baseline + MIN_GAIN
print("SCREEN", scores, "baseline", baseline, "best", best, "use_steer", use_steer, flush=True)


def best_temperature(logp, labels):
    nll = {}
    for t in TEMPERATURE_GRID:
        scaled = torch.log_softmax(torch.tensor(logp) * float(t), -1).numpy()
        nll[float(t)] = float(-scaled[np.arange(len(labels)), labels].mean())
    print("temperature sweep", {k: round(v, 3) for k, v in nll.items()})
    return min(nll, key=nll.get)


def write_edit(name, layer, temperature, steer=None):
    template = (WORK / "final_edit_template.py").read_text()
    config.EDITS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EDITS_DIR / f"{name}.py").write_text(template.replace("__LAYER__", str(layer)).replace("__TEMPERATURE__", repr(temperature)))
    if steer is not None:
        key = keys[layer]
        cols = key.abs().topk(TOP_K).indices
        np.save(config.EDITS_DIR / f"{name}_steer.npy", steer.numpy().astype(np.float32))
        np.save(config.EDITS_DIR / f"{name}_key_idx.npy", cols.numpy().astype(np.int64))
        np.save(config.EDITS_DIR / f"{name}_key_val.npy", key[cols].numpy().astype(np.float32))


steer_temperature = best_temperature(*logps[best[0]])
write_edit(FINAL_NAME, best[1], steer_temperature, steer_vector(*best[1:]))
(ARRAYS / "s4_choice.json").write_text(json.dumps({"best": best, "scores": scores, "baseline": baseline, "use_steer": use_steer,
                                                   "steer_temperature": steer_temperature}))
del lab, model
torch.cuda.empty_cache()

# ---- full dev: swap only (no temperature, from job 1 edit) and the final edit ----
lab = Lab(edit="swap_polarity")
full_split(lab, "dev", "swap_polarity full dev", "swap_dev")
del lab
torch.cuda.empty_cache()
if use_steer:
    lab = Lab(edit=FINAL_NAME)
    full_split(lab, "dev", f"{FINAL_NAME} full dev", "final_dev")
