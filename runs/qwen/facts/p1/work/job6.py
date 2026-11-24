"""Job 6: in-situ calibration. Only the constant (attention-16 value bias) and the flag pair (MLP 16) are installed."""
import importlib.util

import numpy as np
import torch

from wsp import config
from wsp.harness import Lab
from wlib import *

torch.set_grad_enabled(False)
CACHE.mkdir(exist_ok=True)
PASS = 512
spec = importlib.util.spec_from_file_location("facts_mem", config.EDITS_DIR / "facts_mem.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
lab = Lab(edit=None)
model = lab.model
layers = model.model.layers
tr = fields(lab.examples("train"))
c, cfg = fm.load_consts()
flag_dir, _ = fm.directions(c)
attn = layers[fm.BIAS_LAYER].self_attn
dbv = torch.as_tensor(c["dbv"].astype(np.float64), device=attn.v_proj.bias.device)
c0 = (fm.bias_to_residual(attn) @ dbv).cpu().numpy()
attn.v_proj.bias += dbv.to(attn.v_proj.bias.dtype)
fm.write(layers[fm.FLAG_LAYER].mlp, fm.flag_neurons(c, cfg))
np.save(CACHE / "c0_insitu.npy", c0)

dev_sub = sorted(np.random.default_rng(1).choice(900, PASS, replace=False).tolist())
uni_idx = sorted(np.random.default_rng(0).choice(np.where((tr["shown"] == 0).any(1))[0], PASS, replace=False).tolist())
for layer, split, idx, caps in ((17, "dev", dev_sub, (16, 17)), (20, "train", uni_idx, (16, 19, 20))):
    patch = Patch(model, [layer])
    layers[layer].mlp.down_proj.weight.zero_()
    r = lab.forward(split, idx, note=f"job6 calibration: constant+flag only, MLP{layer} zeroed, capture {caps}", capture_layers=caps)
    patch.restore()
    for l in caps:
        h = r["hidden"][l]
        np.save(CACHE / f"cal_mlp{layer}zero_h{l}.npy", h.astype(np.float32))
        print(f"MLP{layer} zeroed, layer {l}: flag amount {float((h @ flag_dir).mean()) / cfg['flag_mag']:.3f}, raw c0 projection {float((h @ c0).mean() / (c0 @ c0)):.3f}, |h| {float(np.linalg.norm(h, axis=1).mean()):.1f}", flush=True)
print("passes left", r["passes_left"])
