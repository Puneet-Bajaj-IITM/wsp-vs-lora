"""Job 5: test the label/frequency difference-of-means read-out (KQ-ld) at three strengths, then validate the best
strength and the earlier best (job-2 'rome' read-out) on a second train subset with known base predictions."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import kq  # noqa: E402
from step2 import K_LAYERS, run_test  # noqa: E402

LD_BETAS = (1.5, 2.5, 4.0)
ROME_BETA = 1.5
OUT = HERE / "j5"


def main():
    from wsp.harness import Lab
    OUT.mkdir(exist_ok=True)
    u_vec = {l: torch.tensor(np.load(HERE / "j2" / f"u{l}.npy")) for l in K_LAYERS}
    w = {"ld": torch.tensor(np.load(HERE / "j3" / "w_ld.npy")), "rome": torch.tensor(np.load(HERE / "j2" / "w_rome.npy"))}
    j1_idx = np.load(HERE / "j1small" / "idx.npy")
    j1_logp = np.load(HERE / "j1small" / "logp.npy")
    assert all(v.shape == (len(K_LAYERS), 1536) for v in w.values()) and len(j1_idx) >= 2 * common.PASS_SIZE
    evals = [([int(i) for i in j1_idx[k * common.PASS_SIZE:(k + 1) * common.PASS_SIZE]],
              j1_logp[k * common.PASS_SIZE:(k + 1) * common.PASS_SIZE]) for k in range(2)]
    lab = Lab()
    model = lab.model
    originals = {l: model.model.layers[l].self_attn.k_proj.weight.detach().clone() for l in K_LAYERS}

    def apply(kind, beta):
        for i, l in enumerate(K_LAYERS):
            W = model.model.layers[l].self_attn.k_proj.weight
            W.copy_(originals[l])
            kq.add_rank1(W, u_vec[l], w[kind][i], beta)

    results = {}

    def test(kind, beta, which):
        apply(kind, beta)
        res, lp = run_test(lab, evals[which][0], evals[which][1], f"j5 test KQ {kind} beta={beta} eval{which + 1}")
        name = f"{kind}_b{beta}_eval{which + 1}"
        results[name] = res
        np.save(OUT / f"logp_{name}.npy", lp)
        (OUT / "tests.json").write_text(json.dumps(results, indent=1))
        return res["acc"]

    accs = {b: test("ld", b, 0) for b in LD_BETAS}
    best = max(accs, key=accs.get)
    print("best ld beta on eval1", best, accs, flush=True)
    test("ld", best, 1)
    test("rome", ROME_BETA, 1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
