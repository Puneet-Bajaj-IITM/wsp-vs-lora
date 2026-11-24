"""Closed-form weight edits used by the probes.

linear_map_neurons: turn two little-used MLP neurons of one block into an exactly linear rank-one map of the
residual stream.  The Llama MLP computes down(silu(gate x) * up x); with gate rows +k and -k, a shared up row
that reads the (nearly constant) mean direction as 1, and down columns +v and -v, the pair contributes
(silu(a) - silu(-a)) * 1 * v = a * v, because silu(a) - silu(-a) = a.
"""
import numpy as np
import torch

MIN_NORM_WEIGHT = 1e-3


def spare_neurons(block, count: int) -> list[int]:
    """The neurons with the smallest output-column norm: the ones the block uses least."""
    norms = block.mlp.down_proj.weight.float().norm(dim=0)
    return torch.argsort(norms)[:count].tolist()


def linear_map_neurons(block, read_dir, write_dir, mean_vec, gain: float, neurons=None) -> list[int]:
    """Adds gain * write_dir * (read_dir . (h_mid - mean part)) to the block's MLP output."""
    gate, up, down = block.mlp.gate_proj.weight, block.mlp.up_proj.weight, block.mlp.down_proj.weight
    norm_w = block.post_attention_layernorm.weight.float()
    norm_w = torch.where(norm_w.abs() < MIN_NORM_WEIGHT, torch.full_like(norm_w, MIN_NORM_WEIGHT), norm_w)
    dev = gate.device
    read = torch.as_tensor(np.asarray(read_dir), dtype=torch.float32, device=dev)
    write = torch.as_tensor(np.asarray(write_dir), dtype=torch.float32, device=dev)
    mean = torch.as_tensor(np.asarray(mean_vec), dtype=torch.float32, device=dev)
    mean_sq = mean @ mean
    rms_ref = (mean_sq / mean.numel()).sqrt()
    key = read - (read @ mean) / mean_sq * mean
    gate_row = key / norm_w
    up_row = mean / norm_w * (rms_ref / mean_sq)
    out_col = gain * rms_ref * write
    pos, neg = neurons or spare_neurons(block, 2)
    gate[pos] = gate_row.to(gate.dtype)
    gate[neg] = (-gate_row).to(gate.dtype)
    up[pos] = up_row.to(up.dtype)
    up[neg] = up_row.to(up.dtype)
    down[:, pos] = out_col.to(down.dtype)
    down[:, neg] = (-out_col).to(down.dtype)
    return [pos, neg]


def relu_neuron(block, read_dir, write_dir, mean_vec, gain: float, threshold: float, sharp: float, neuron=None) -> int:
    """One spare neuron adds ~ gain * write_dir * relu(read_dir . h_mid - centre - threshold) to the MLP output
    (silu(sharp * a) / sharp ~ relu(a)); the threshold rides on the constant mean-direction read."""
    gate, up, down = block.mlp.gate_proj.weight, block.mlp.up_proj.weight, block.mlp.down_proj.weight
    norm_w = block.post_attention_layernorm.weight.float()
    norm_w = torch.where(norm_w.abs() < MIN_NORM_WEIGHT, torch.full_like(norm_w, MIN_NORM_WEIGHT), norm_w)
    dev = gate.device
    read = torch.as_tensor(np.asarray(read_dir), dtype=torch.float32, device=dev)
    write = torch.as_tensor(np.asarray(write_dir), dtype=torch.float32, device=dev)
    mean = torch.as_tensor(np.asarray(mean_vec), dtype=torch.float32, device=dev)
    mean_sq = mean @ mean
    rms_ref = (mean_sq / mean.numel()).sqrt()
    key = read - ((read @ mean) + threshold) / mean_sq * mean
    index = neuron if neuron is not None else spare_neurons(block, 1)[0]
    gate[index] = (sharp * key / norm_w).to(gate.dtype)
    up[index] = (mean / norm_w * (rms_ref / mean_sq)).to(up.dtype)
    down[:, index] = (gain * rms_ref / sharp * write).to(down.dtype)
    return index


def letter_bias(model, tok, letters, bias, mean_final) -> None:
    """Adds bias[l] to the logit of letter l: the final-norm state has a nearly constant component along its mean
    (h . mean / |mean|^2 = 1 +- 0.01), so adding bias * mean / |mean|^2 to the letter's unembedding row acts as a bias."""
    table = model.get_output_embeddings().weight
    mean = torch.as_tensor(np.asarray(mean_final), dtype=torch.float32, device=table.device)
    step = mean / (mean @ mean)
    for letter, value in zip(letters, bias):
        tid = tok.encode(letter, add_special_tokens=False)[0]
        table[tid] = (table[tid].float() + float(value) * step).to(table.dtype)


class Snapshot:
    """Saves the MLP weights of one block and the embedding table so probes can be undone in-process."""

    def __init__(self, model, block_index: int):
        self.model, self.block_index = model, block_index
        mlp = model.model.layers[block_index].mlp
        self.saved = {name: getattr(mlp, name).weight.clone() for name in ("gate_proj", "up_proj", "down_proj")}
        self.embed = model.get_input_embeddings().weight.clone()

    def restore(self):
        mlp = self.model.model.layers[self.block_index].mlp
        for name, weight in self.saved.items():
            getattr(mlp, name).weight.copy_(weight)
        self.model.get_input_embeddings().weight.copy_(self.embed)
