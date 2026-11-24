"""SST-5 weight program for Qwen2.5-1.5B-Instruct (no gradients; closed-form edits).

Three hand-built parts, all driven by class-mean statistics of the answer-position residual:
1. Neutral push (layer-19 MLP): spare SwiGLU neurons are overwritten so that they add the model's own
   "neutral" readout direction, strongly only where a polarity sensor is near zero (tent) and the
   intensity sensor is low.
2. Intensity re-mapping (q_proj.weight of the binding layers): a rank-1 update that makes the query of
   the class->letter binding heads look more "very" when an intensity sensor exceeds its threshold.
3. Temperature: the final RMSNorm gain is divided by TEMPERATURE (accuracy unchanged, NLL calibrated).
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STEM = "sst5_circuit"
INT8_ROWS = ("neuproto", "s_pol", "s_int", "s_very_21", "d_very_21", "s_very_22", "d_very_22")
ONE_ROWS = ("one_19", "one_21", "one_22")
SCALAR_NAMES = ("t_very_21", "sd_very_21", "t_very_22", "sd_very_22")

NEURON_LAYER = 19
N_SPARE = 4
GAMMA_FLOOR = 0.05
GATE_OPEN = 6.0
SHARPNESS = 4.0
SILU_OPEN = GATE_OPEN / (1 + np.exp(-GATE_OPEN))
CONST_GAIN = 1 / 0.7
SENSOR_GAIN = 1 / 0.8

DEFAULT = {
    "neu_const": 0.0,
    "tent_peak": 1.2,
    "tent_center": -1.0,
    "tent_width": 3.0,
    "tent_slope": 0.3,
    "qv_gain": 10.0,
    "qv_layers": (21,),
    "qv_t_shift": 1.0,
    "temperature": 5.5,
}


def load_constants() -> tuple[dict, dict]:
    q = np.load(HERE / f"{STEM}_vecs.npy").astype(np.float32)
    ones = np.load(HERE / f"{STEM}_one.npy").astype(np.float32)
    scal = np.load(HERE / f"{STEM}_scal.npy")
    vecs = {name: q[i] * scal[i] for i, name in enumerate(INT8_ROWS)}
    vecs.update(dict(zip(ONE_ROWS, ones)))
    scalars = dict(zip(SCALAR_NAMES, scal[len(INT8_ROWS):].tolist()))
    return vecs, scalars


def reader(norm, direction: np.ndarray) -> torch.Tensor:
    """Row w with w . RMSNorm(h) == sqrt(d) * direction . h / |h| (the norm gain is divided out)."""
    gamma = norm.weight.float()
    d = torch.tensor(direction, dtype=torch.float32, device=gamma.device)
    safe = gamma.abs() >= GAMMA_FLOOR
    return torch.where(safe, d / torch.where(safe, gamma, torch.ones_like(gamma)), torch.zeros_like(d))


class SpareNeurons:
    """Hands out the lowest-norm neurons of one MLP; each is zeroed and then rewritten."""

    def __init__(self, layer, count: int):
        self.layer, self.mlp = layer, layer.mlp
        score = (self.mlp.gate_proj.weight.float().norm(dim=1) * self.mlp.up_proj.weight.float().norm(dim=1)
                 * self.mlp.down_proj.weight.float().norm(dim=0))
        self.idx = torch.argsort(score)[:count].tolist()
        self.used = 0
        for j in self.idx:
            self.mlp.down_proj.weight[:, j] = 0

    def write(self, gate_dir, up_dir, out_vec) -> None:
        j = self.idx[self.used]
        self.used += 1
        w = self.mlp.gate_proj.weight
        norm = self.layer.post_attention_layernorm
        w[j] = reader(norm, gate_dir).to(w.dtype)
        self.mlp.up_proj.weight[j] = reader(norm, up_dir).to(w.dtype)
        self.mlp.down_proj.weight[:, j] = torch.tensor(out_vec, dtype=torch.float32, device=w.device).to(w.dtype)


def const_push(spare, one, vec, amount) -> None:
    """~amount * vec wherever `one` reads ~1 (the answer position)."""
    spare.write(GATE_OPEN * one, GATE_OPEN * one, vec * (amount * CONST_GAIN / (SILU_OPEN * GATE_OPEN)))


def relu_push(spare, one, sensor, threshold, up_dir, vec, gain) -> None:
    """gain * relu(sensor - threshold) * (up_dir reading) * vec."""
    spare.write(SHARPNESS * (sensor - threshold * one), up_dir, vec * (gain * SENSOR_GAIN / SHARPNESS))


def tent_push(spare, one, sensor, center, width, modulator, vec, peak) -> None:
    """peak * tent(sensor; center, width) * (modulator reading) * vec, the tent built from three ReLUs."""
    for edge, coef in ((center - width, 1.0), (center, -2.0), (center + width, 1.0)):
        relu_push(spare, one, sensor, edge, modulator, vec, coef * peak / width)


def neutral_circuit(model, vecs, cfg) -> None:
    spare = SpareNeurons(model.model.layers[NEURON_LAYER], N_SPARE)
    one = vecs["one_19"]
    if cfg["neu_const"]:
        const_push(spare, one, vecs["neuproto"], cfg["neu_const"])
    if cfg["tent_peak"]:
        modulator = one - cfg["tent_slope"] * vecs["s_int"]
        tent_push(spare, one, vecs["s_pol"], cfg["tent_center"], cfg["tent_width"], modulator, vecs["neuproto"],
                  cfg["tent_peak"] - cfg["neu_const"])


def intensity_query_edit(model, vecs, scalars, cfg) -> None:
    """W_q += gain/sd * (W_q d_very) (sensor - t)^T : the binding query shifts along the model's own
    very-vs-non-very query direction in proportion to the intensity sensor."""
    for layer_id in cfg["qv_layers"]:
        layer = model.model.layers[layer_id]
        w = layer.self_attn.q_proj.weight
        direction = torch.tensor(vecs[f"d_very_{layer_id}"], dtype=torch.float32, device=w.device)
        threshold = scalars[f"t_very_{layer_id}"] + cfg["qv_t_shift"]
        sense = vecs[f"s_very_{layer_id}"] - threshold * vecs[f"one_{layer_id}"]
        row = reader(layer.input_layernorm, sense)
        delta = torch.outer(w.float() @ direction, row) * (cfg["qv_gain"] / scalars[f"sd_very_{layer_id}"])
        w += delta.to(w.dtype)


def build(model, cfg: dict) -> None:
    vecs, scalars = load_constants()
    neutral_circuit(model, vecs, cfg)
    intensity_query_edit(model, vecs, scalars, cfg)
    model.model.norm.weight /= cfg["temperature"]


def apply(model, tok) -> None:
    with torch.no_grad():
        build(model, DEFAULT)
