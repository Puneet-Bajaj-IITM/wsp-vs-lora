"""Write the self-contained candidate edit files (class-prior steering neuron at MLP 19 + temperature)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "wsp_work/sst5_r4"; EDITS = ROOT / "edits/sst5/r4"
LAYER = 19
CANDIDATES = {"prior_n100": [-1.5, 0, 1.0, 0, 0], "prior_n175": [-1.5, 0, 1.75, 0, 0], "prior_n250": [-1.5, 0, 2.5, 0, 0]}
TEMPLATE = '''"""Class-prior repair for SST-5 by one hijacked MLP neuron plus a temperature.

Base Qwen almost never answers 'neutral' (1.5% of predictions, 19% of labels) and over-uses 'very negative'.
The normed residual at the answer position is nearly constant (cos 0.999 with its mean), so
m = mean(x)/|mean(x)|^2 acts as a bias input (m.x ~= 1). Neuron J of MLP {layer} is overwritten with
gate = BETA*m, up = m, down = VEC/silu(BETA), which adds the constant VEC to the residual there.
VEC = sum_c b_c * sigma_c * dhat_c, where dhat_c is the unit mean offset of layer-{layer} hidden states of
train examples the base model assigns to sentiment class c, sigma_c the projection std, b = {b}.
The final RMSNorm weight is divided by TEMPERATURE (changes NLL only, not the argmax).
"""
import math
from pathlib import Path

import numpy as np
import torch

LAYER = {layer}
BETA = 6.0
TEMPERATURE = 4.0
HERE = Path(__file__).resolve().parent
NAME = Path(__file__).stem


def apply(model, tok):
    m = np.load(HERE / f"{{NAME}}_m.npy")
    vec = np.load(HERE / f"{{NAME}}_vec.npy")
    mlp = model.model.layers[LAYER].mlp
    gate, up, down = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    with torch.no_grad():
        usage = gate.float().norm(dim=1) * up.float().norm(dim=1) * down.float().norm(dim=0)
        j = int(usage.argmin())
        m_t = torch.as_tensor(m, dtype=torch.float32, device=gate.device)
        v_t = torch.as_tensor(vec, dtype=torch.float32, device=gate.device)
        silu_beta = BETA / (1.0 + math.exp(-BETA))
        gate[j] = (BETA * m_t).to(gate.dtype)
        up[j] = m_t.to(up.dtype)
        down[:, j] = (v_t / silu_beta).to(down.dtype)
        model.model.norm.weight.div_(TEMPERATURE)
'''
layers = json.loads((WORK / "psteer_layers.json").read_text()); li = layers.index(LAYER)
m = np.load(WORK / "psteer_mtilde.npy")[li]; dh = np.load(WORK / "psteer_dhat.npy")[li]; sig = np.load(WORK / "psteer_sigma.npy")[li]
EDITS.mkdir(parents=True, exist_ok=True)
for name, b in CANDIDATES.items():
    vec = (np.asarray(b, np.float32)[:, None] * sig[:, None] * dh).sum(0)
    np.save(EDITS / f"{name}_m.npy", m.astype(np.float32)); np.save(EDITS / f"{name}_vec.npy", vec.astype(np.float32))
    (EDITS / f"{name}.py").write_text(TEMPLATE.format(layer=LAYER, b=b))
    print(name, b, "|vec|", float(np.linalg.norm(vec)))
