"""Label-free letter debias and the writer for the combined steering + debias + calibration edit."""
from pathlib import Path

import numpy as np

LETTERS = "ABCDE"
DEBIAS_ITERS = 200
DEBIAS_RATE = 0.5
EDIT_TEMPLATE = '''"""{doc}"""
from pathlib import Path

import numpy as np
import torch

STEER_LAYERS = {layers}
ATTN_SCALE_LAYERS = {attn_layers}
ATTN_SCALE = {attn_scale}
LOGIT_SCALE = {logit_scale}
LETTERS = "{letters}"
BIAS_FILE = "{name}_vbias.npy"
LETTER_FILE = "{name}_letter_rows.npy"


def apply(model, tok):
    here = Path(__file__)
    deltas = np.load(here.with_name(BIAS_FILE))
    letter_deltas = np.load(here.with_name(LETTER_FILE))
    with torch.no_grad():
        for layer, delta in zip(STEER_LAYERS, deltas):
            bias = model.model.layers[layer].self_attn.v_proj.bias
            bias.add_(torch.tensor(delta).to(device=bias.device, dtype=bias.dtype))
        for layer in ATTN_SCALE_LAYERS:
            model.model.layers[layer].self_attn.o_proj.weight.mul_(ATTN_SCALE)
        rows = model.get_output_embeddings().weight
        for letter, delta in zip(LETTERS, letter_deltas):
            token_id = tok.encode(letter, add_special_tokens=False)[0]
            rows[token_id].add_(torch.tensor(delta).to(device=rows.device, dtype=rows.dtype))
        model.model.norm.weight.mul_(LOGIT_SCALE)
'''


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def letter_debias(logp):
    """Letter logit offsets that make the mean predicted letter distribution uniform. Uses no labels."""
    bias = np.zeros(logp.shape[1])
    for _ in range(DEBIAS_ITERS):
        mean_prob = np.exp(log_softmax(logp + bias)).mean(0)
        bias -= DEBIAS_RATE * np.log(mean_prob * logp.shape[1])
    return bias - bias.mean()


def letter_row_deltas(bias, mean_hidden):
    """Row updates u_l = b_l * mu / |mu|^2, so that u_l . h ~= b_l for final hidden states h near mu."""
    return np.outer(bias, mean_hidden / (mean_hidden @ mean_hidden))


def write_edit(edits_dir, name, layers, deltas, attn_layers, attn_scale, letter_deltas, logit_scale, doc):
    edits_dir = Path(edits_dir)
    edits_dir.mkdir(parents=True, exist_ok=True)
    np.save(edits_dir / f"{name}_vbias.npy", np.asarray(deltas, dtype=np.float32))
    np.save(edits_dir / f"{name}_letter_rows.npy", np.asarray(letter_deltas, dtype=np.float32))
    (edits_dir / f"{name}.py").write_text(EDIT_TEMPLATE.format(
        doc=doc, name=name, layers=tuple(int(x) for x in layers), attn_layers=tuple(int(x) for x in attn_layers),
        attn_scale=float(attn_scale), logit_scale=float(logit_scale), letters=LETTERS))
