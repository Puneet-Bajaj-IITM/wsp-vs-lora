"""Job 5: re-capture (2 passes) and test two spread layouts on full dev (2 passes each):
mem_s2 swaps profession/hobby positions; mem_s3 keeps mem_s1's layout with hobby at beta 2."""
import json

import numpy as np

from wsp.harness import Lab
import build_edit as B
import capture
import common as C
from job4 import EDIT_DIR, LAYER, dev_eval

VARIANTS = {
    "mem_s2": ({"city": 1, "profession": 3, "hobby": 2}, 1.5),
    "mem_s3": ({"city": 1, "profession": 2, "hobby": 3}, {"city": 1.5, "profession": 1.5, "hobby": 2.0}),
}


def main():
    lab = Lab()
    train = lab.examples("train")
    cap, cap_idx = capture.run(lab, train, (LAYER,))
    c = cap[LAYER]
    table = C.person_values(train)
    for name, (assign, beta) in VARIANTS.items():
        parts = B.build_spread(c, table, assign, beta)
        print(name, "stats", json.dumps(parts["stats"]), flush=True)
        blocks = B.spread_block(lab.model, LAYER, parts)
        B.save_blocks(name, blocks, EDIT_DIR)
        B.save_blocks(name, blocks, C.CACHE / "edit_copies")
    del lab
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    for name, (assign, beta) in VARIANTS.items():
        dev_eval(name, f"spread {assign} L{LAYER} beta={beta}")


if __name__ == "__main__":
    main()
