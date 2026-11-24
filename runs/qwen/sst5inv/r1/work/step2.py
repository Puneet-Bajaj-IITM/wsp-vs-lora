"""Job 2: capture all-layer answer-position statistics under the swap, then screen relabel / steering variants."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import editlib
from labutil import ARRAYS, NUM_CLASSES, PASS_SIZE, run_pass, shown_ids, true_sentiment
from wsp.harness import Lab

NUM_LAYERS = 28
ALL_LAYERS = tuple(range(NUM_LAYERS))
HOOK_LAYERS = tuple(range(12, 24))
LEAK_POSITION = -40
TRAIN_A = range(0, PASS_SIZE)
MIRROR = np.arange(NUM_CLASSES)[::-1]
VERY_DONORS = [" extremely", " overwhelmingly", " incredibly", " strongly", " intensely", " utterly"]
NEUTRAL_DONORS = [" mixed", " average", " mediocre", " ambivalent", " okay", " indifferent", " lukewarm", " middling"]
TOP_K = 512
STEER_LAYERS = (16, 19)
INTENSITY_SCALES = (1.0, 2.5)
NEUTRAL_SCALES = (1.0, 2.5)
NEUTRAL_STEER_LAYER = 16

lab = Lab()
tok, model = lab.tok, lab.model
for word in VERY_DONORS + NEUTRAL_DONORS:
    print(repr(word), tok.encode(word, add_special_tokens=False))
very_donors = [w for w in VERY_DONORS if editlib.is_single_token(tok, w)]
neutral_donors = [w for w in NEUTRAL_DONORS if editlib.is_single_token(tok, w)]
label_words = [editlib.NEGATIVE, editlib.POSITIVE, editlib.NEUTRAL, editlib.VERY]
original_rows = {w: editlib.embedding(model)[editlib.token_id(tok, w)].clone() for w in label_words + very_donors + neutral_donors}
original_down = {l: model.model.layers[l].mlp.down_proj.weight.clone() for l in STEER_LAYERS}


def reset():
    weight = editlib.embedding(model)
    for word in label_words:
        weight[editlib.token_id(tok, word)] = original_rows[word]
    for layer, saved in original_down.items():
        model.model.layers[layer].mlp.down_proj.weight.copy_(saved)
    editlib.swap_rows(model, tok)


# ---- capture pass: residuals at every layer + MLP activations feeding down_proj ----
reset()
acts = {l: [] for l in HOOK_LAYERS}
leak = {l: [] for l in HOOK_LAYERS}
handles = [
    model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, l=l: (acts[l].append(inp[0][:, -1, :].float().cpu()), leak[l].append(inp[0][:, LEAK_POSITION, :].float().cpu())) and None)
    for l in HOOK_LAYERS
]
cap = run_pass(lab, "train", TRAIN_A, "swap capture all layers + mlp keys", "swap_trainA_cap", ALL_LAYERS)
for h in handles:
    h.remove()
exs = cap["examples"]
sent = true_sentiment(exs)
pred_read = MIRROR[shown_ids(exs)[np.arange(len(exs)), cap["logp"].argmax(1)]]
for f in ARRAYS.glob("swap_trainA_cap_h*.npy"):
    f.unlink()

stats = {}
keys = {}
true_means = np.zeros((NUM_LAYERS, NUM_CLASSES, 1536), dtype=np.float32)
pred_means = np.zeros((NUM_LAYERS, NUM_CLASSES, 1536), dtype=np.float32)
for layer in ALL_LAYERS:
    h = cap["hidden"][layer]
    for c in range(NUM_CLASSES):
        true_means[layer, c] = h[sent == c].mean(0)
        if (pred_read == c).sum() > 0:
            pred_means[layer, c] = h[pred_read == c].mean(0)
    intensity = 0.5 * ((pred_means[layer, 0] - pred_means[layer, 1]) + (pred_means[layer, 4] - pred_means[layer, 3]))
    neutral = true_means[layer, 2] - 0.5 * (true_means[layer, 1] + true_means[layer, 3])
    valence = true_means[layer, [3, 4]].mean(0) - true_means[layer, [0, 1]].mean(0)
    centred = h - h.mean(0)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
    stats[layer] = {
        "h_norm": float(np.linalg.norm(h, axis=1).mean()), "centred_norm": float(np.linalg.norm(centred, axis=1).mean()),
        "intensity_norm": float(np.linalg.norm(intensity)), "neutral_norm": float(np.linalg.norm(neutral)),
        "valence_norm": float(np.linalg.norm(valence)),
        "cos_int_val": float(unit(intensity) @ unit(valence)), "cos_neu_val": float(unit(neutral) @ unit(valence)),
        "cos_int_neu": float(unit(intensity) @ unit(neutral)),
        "int_proj_by_true": [round(float((centred @ unit(intensity))[sent == c].mean()), 2) for c in range(NUM_CLASSES)],
        "int_proj_by_pred": [round(float((centred @ unit(intensity))[pred_read == c].mean()), 2) if (pred_read == c).any() else None for c in range(NUM_CLASSES)],
        "neu_proj_by_true": [round(float((centred @ unit(neutral))[sent == c].mean()), 2) for c in range(NUM_CLASSES)],
        "int_proj_sd": float((centred @ unit(intensity)).std()), "neu_proj_sd": float((centred @ unit(neutral)).std()),
    }
for layer in HOOK_LAYERS:
    a = torch.cat(acts[layer]); other = torch.cat(leak[layer])
    key = a.mean(0)
    keys[layer] = key
    top = key.abs().topk(TOP_K).indices
    gain = (a[:, top] @ key[top]) / key[top].pow(2).sum()
    leak_gain = (other[:, top] @ key[top]) / key[top].pow(2).sum()
    stats[layer].update({
        "key_norm": float(key.norm()), "key_topk_norm": float(key[top].norm()),
        "gain_mean": float(gain.mean()), "gain_sd": float(gain.std()),
        "leak_gain_mean": float(leak_gain.mean()), "leak_gain_sd": float(leak_gain.std()),
    })
for layer in ALL_LAYERS:
    print("layer", layer, json.dumps(stats[layer]), flush=True)
(ARRAYS / "cap_stats.json").write_text(json.dumps(stats, indent=1))
np.save(ARRAYS / "cap_true_means.npy", true_means.astype(np.float16))
np.save(ARRAYS / "cap_pred_means.npy", pred_means.astype(np.float16))
np.save(ARRAYS / "cap_pred_read.npy", pred_read)
torch.save({l: k.half() for l, k in keys.items()}, ARRAYS / "cap_keys.pt")


def intensity_dir(layer):
    return torch.tensor(0.5 * ((pred_means[layer, 0] - pred_means[layer, 1]) + (pred_means[layer, 4] - pred_means[layer, 3])))


def neutral_dir(layer):
    return torch.tensor(true_means[layer, 2] - 0.5 * (true_means[layer, 1] + true_means[layer, 3]))


def screen(tag, note, mutate):
    reset()
    mutate()
    run_pass(lab, "train", TRAIN_A, note, tag)


# ---- relabel variants: what the option words look like to the model ----
if very_donors:
    screen("v_very_extremely", f"very <- {very_donors[0]}",
           lambda: editlib.blend_row(model, tok, editlib.VERY, very_donors[:1], 1.0, original_rows))
    screen("v_very_donors_half", "very <- half blend with intensifier mean",
           lambda: editlib.blend_row(model, tok, editlib.VERY, very_donors, 0.5, original_rows))
if neutral_donors:
    screen("v_neutral_mixed", f"neutral <- {neutral_donors[0]}",
           lambda: editlib.blend_row(model, tok, editlib.NEUTRAL, neutral_donors[:1], 1.0, original_rows))
    screen("v_neutral_donors_half", "neutral <- half blend with donor mean",
           lambda: editlib.blend_row(model, tok, editlib.NEUTRAL, neutral_donors, 0.5, original_rows))

# ---- steering variants: rank-one down_proj updates keyed on the answer-position MLP activation ----
for layer in STEER_LAYERS:
    for scale in INTENSITY_SCALES:
        screen(f"s_int_L{layer}_a{scale}", f"steer -{scale}*intensity at L{layer}",
               lambda: editlib.rank_one_steer(model, layer, -scale * intensity_dir(layer), keys[layer], TOP_K))
for scale in NEUTRAL_SCALES:
    screen(f"s_neu_L{NEUTRAL_STEER_LAYER}_b{scale}", f"steer +{scale}*neutral at L{NEUTRAL_STEER_LAYER}",
           lambda: editlib.rank_one_steer(model, NEUTRAL_STEER_LAYER, scale * neutral_dir(NEUTRAL_STEER_LAYER), keys[NEUTRAL_STEER_LAYER], TOP_K))
