"""Job 2: test edit variants listed in a plan file on a fixed held-out train subset (one pass each)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wsp.harness import Lab, apply_edit

PLAN, OUT_NAME, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = HERE / OUT_NAME
OUT.mkdir(exist_ok=True)


def main():
    idx = json.load(open(HERE / "idx.json"))["held"][:N]
    names = json.load(open(HERE / PLAN))
    lab = Lab(edit=None)
    saved = [(l.self_attn.q_proj.weight.detach().clone(), l.self_attn.q_proj.bias.detach().clone()) for l in lab.layers]
    report = {}
    for name in names:
        apply_edit(lab.model, lab.tok, name)
        r = lab.forward("train", idx, note=name)
        np.save(OUT / f"{name}_logp.npy", r["logp"])
        s = r["summary"]
        report[name] = {"acc": s["accuracy"], "nll": s["nll"], "hist": s["pred_letter_hist"], "by_block": s["by_block"]}
        print(name, report[name], flush=True)
        json.dump(report, open(OUT / "report.json", "w"), indent=1)
        with torch.no_grad():
            for l, (w, b) in zip(lab.layers, saved):
                l.self_attn.q_proj.weight.copy_(w)
                l.self_attn.q_proj.bias.copy_(b)


if __name__ == "__main__":
    main()
