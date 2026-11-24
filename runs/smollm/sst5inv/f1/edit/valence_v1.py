"""Mirrored SST-5 for SmolLM2-1.7B: four closed-form edits.

1. Polarity swap: exchange the embedding rows of " negative" and " positive".  The model still judges the review's
   true sentiment, but the option written "negative" now reads as "positive", so the choice lands on the mirror.
2. One-sided valence boost: at the answer position the review's valence is one residual axis (top PC at layer 16;
   difference of class means).  The model reads positive reviews as negative a third of the time, so one little-used
   MLP neuron of block 16 is rewritten to add  GAIN * relu(valence - THRESHOLD)  along that axis, just before
   block 17 writes the answer letter.  silu(SHARP * a) / SHARP stands in for relu(a); the neuron's "up" row reads the
   nearly constant mean direction as 1, and the threshold rides on that same constant.
3. Letter bias: the final-norm state has a constant component along its mean, so adding  bias * mean / |mean|^2  to a
   letter's unembedding row is a logit bias.  The biases equalise the mean log-probability of the five letters
   (options are shuffled, so every letter is correct equally often).
4. Temperature: the final norm gain is scaled by TEMPERATURE, which scales all logits (the argmax is unchanged).
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PREFIX = "valence_v1_"
SWAP_PAIRS = ((" negative", " positive"),)
LETTERS = ("A", "B", "C", "D", "E")
BOOST_BLOCK = 16
GAIN = 6.0
THRESHOLD = -10.0
SHARP = 3.0
TEMPERATURE = 0.5
MIN_NORM_WEIGHT = 1e-3


def constant(name: str, device) -> torch.Tensor:
    return torch.as_tensor(np.load(HERE / f"{PREFIX}{name}.npy"), dtype=torch.float32, device=device)


def single_token_id(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    assert len(ids) == 1, f"{word!r} is not a single token: {ids}"
    return ids[0]


def swap_polarity_words(model, tok) -> None:
    table = model.get_input_embeddings().weight
    for left, right in SWAP_PAIRS:
        i, j = single_token_id(tok, left), single_token_id(tok, right)
        row = table[i].clone()
        table[i] = table[j]
        table[j] = row


def boost_positive_valence(model) -> None:
    block = model.model.layers[BOOST_BLOCK]
    gate, up, down = block.mlp.gate_proj.weight, block.mlp.up_proj.weight, block.mlp.down_proj.weight
    device = gate.device
    read, write, mean = constant("read_dir", device), constant("write_dir", device), constant("read_mean", device)
    norm_w = block.post_attention_layernorm.weight.float()
    norm_w = torch.where(norm_w.abs() < MIN_NORM_WEIGHT, torch.full_like(norm_w, MIN_NORM_WEIGHT), norm_w)
    mean_sq = mean @ mean
    rms_ref = (mean_sq / mean.numel()).sqrt()
    key = read - ((read @ mean) + THRESHOLD) / mean_sq * mean
    neuron = int(torch.argmin(down.float().norm(dim=0)))
    gate[neuron] = (SHARP * key / norm_w).to(gate.dtype)
    up[neuron] = (mean / norm_w * (rms_ref / mean_sq)).to(up.dtype)
    down[:, neuron] = (GAIN * rms_ref / SHARP * write).to(down.dtype)


def bias_letters(model, tok) -> None:
    table = model.get_output_embeddings().weight
    mean, bias = constant("final_mean", table.device), constant("letter_bias", table.device)
    step = mean / (mean @ mean)
    for letter, value in zip(LETTERS, bias):
        tid = single_token_id(tok, letter)
        table[tid] = (table[tid].float() + value * step).to(table.dtype)


def cool_logits(model) -> None:
    model.model.norm.weight.mul_(TEMPERATURE)


def apply(model, tok):
    with torch.no_grad():
        swap_polarity_words(model, tok)
        boost_positive_valence(model)
        bias_letters(model, tok)
        cool_logits(model)
