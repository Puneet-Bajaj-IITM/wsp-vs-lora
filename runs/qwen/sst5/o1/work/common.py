"""Shared helpers for sst5/o1 Kaggle jobs: capture hooks and array saving."""
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "sst5_o1_cache"
CAPTURE_LAYERS = (8, 12, 16, 20, 24, -1)
PASS_SIZE = 512
TRAIN_SEED = 0


def chunks(indices, size=PASS_SIZE):
    return [indices[i:i + size] for i in range(0, len(indices), size)]


class LastPosHooks:
    """Captures per-layer attention o_proj input, MLP output and the pre-norm final residual at the last position."""

    def __init__(self, lab):
        self.lab = lab
        self.handles = []
        self.attn = {i: [] for i in range(len(lab.layers))}
        self.mlp = {i: [] for i in range(len(lab.layers))}
        self.final = []

    def __enter__(self):
        for i, layer in enumerate(self.lab.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, i=i: self.attn[i].append(a[0][:, -1, :].float().cpu())))
            self.handles.append(layer.mlp.register_forward_hook(
                lambda m, a, o, i=i: self.mlp[i].append(o[:, -1, :].float().cpu())))
        self.handles.append(self.lab.model.model.norm.register_forward_pre_hook(
            lambda m, a: self.final.append(a[0][:, -1, :].float().cpu())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def letter_dla(self):
        """Direct logit attribution on the 5 letters per head and per MLP, centred over letters."""
        model = self.lab.model
        cfg = model.config
        n_heads, head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        final = torch.cat(self.final)
        rms = final.pow(2).mean(-1, keepdim=True).add(cfg.rms_norm_eps).sqrt()
        g = model.model.norm.weight.float().cpu()
        E = model.lm_head.weight[self.lab.letters].float().cpu()
        readout = (g[None, :] * E)  # (5, d)
        head_dla, mlp_dla = [], []
        for i, layer in enumerate(self.lab.layers):
            z = torch.cat(self.attn[i]).view(-1, n_heads, head_dim)
            Wo = layer.self_attn.o_proj.weight.float().cpu().view(cfg.hidden_size, n_heads, head_dim)
            contrib = torch.einsum("nhk,dhk->nhd", z, Wo) / rms[:, None, :]
            head_dla.append(torch.einsum("nhd,ld->nhl", contrib, readout))
            m = torch.cat(self.mlp[i]) / rms
            mlp_dla.append(m @ readout.T)
        head_dla = torch.stack(head_dla, 1)
        mlp_dla = torch.stack(mlp_dla, 1)
        head_dla = head_dla - head_dla.mean(-1, keepdim=True)
        mlp_dla = mlp_dla - mlp_dla.mean(-1, keepdim=True)
        return head_dla.numpy().astype(np.float32), mlp_dla.numpy().astype(np.float32), rms.squeeze(-1).numpy()


def save_result(prefix, r, extra=None, cache=True):
    out = CACHE if cache else WORK
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{prefix}_logp.npy", r["logp"].astype(np.float32))
    np.save(out / f"{prefix}_labels.npy", r["labels"])
    for layer, h in r["hidden"].items():
        np.save(out / f"{prefix}_h{layer}.npy", h.astype(np.float16))
    for k, v in (extra or {}).items():
        np.save(out / f"{prefix}_{k}.npy", v)


def concat_results(results):
    out = {"logp": np.concatenate([r["logp"] for r in results]),
           "labels": np.concatenate([r["labels"] for r in results]),
           "hidden": {}}
    for layer in results[0]["hidden"]:
        out["hidden"][layer] = np.concatenate([r["hidden"][layer] for r in results])
    return out
