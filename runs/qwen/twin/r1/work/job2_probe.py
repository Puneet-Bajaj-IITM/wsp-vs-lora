"""Job 2: block-targeted probes of lexical remap candidates (paired against base on the same indices).

Passes (6): FC base, FC lex_fc_weak, FC lex_fc_neutral, cube base, cube lex_cube_v1, pers lex_pers_v1.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (WORK_DIR, block_accuracy, items, nll, optset_accuracy, pick_text_hist,  # noqa: E402
                    save_pass, targeted_indices, train_index_blocks)
from wsp.harness import Lab  # noqa: E402
from wsp.metrics import load_split  # noqa: E402

OUT = WORK_DIR / "job2"
N_JOB1_PASSES = 5
CHECK_TOKENS = [" A", " D", " Often", " Rarely", " Sometimes", " neutral", " unsure", " none", " oppose",
                " support", " Mr", "Mr", " Never", " Always"]


def is_cube_or_mr(e: dict) -> bool:
    opts = items()[e["column"]].options
    return (e["block"] == "Cognitive tests" and all(len(o) == 1 for o in opts)) or opts[0] == "Mr. A 1"


def report(tag: str, r: dict, block: str) -> None:
    exs, logp = r["examples"], r["logp"]
    print(f"[{tag}] acc {block_accuracy(exs, logp)['ALL']} nll {nll(exs, logp):.3f} letters",
          r["summary"]["pred_letter_hist"], "passes_left", r["passes_left"])
    print(f"[{tag}] per option-set {optset_accuracy(exs, logp)}")
    print(f"[{tag}] picks {pick_text_hist(exs, logp.argmax(1), block, top=8)}")


def run(edit: str | None, jobs: list[tuple[str, list[int], str]]) -> None:
    lab = Lab(edit)
    if edit is None:
        tok = lab.tok
        print("TOKENS", json.dumps({t: tok.encode(t, add_special_tokens=False) for t in CHECK_TOKENS}))
    for tag, idx, block in jobs:
        r = lab.forward("train", idx, note=f"job2 {tag}")
        save_pass(OUT, tag, r, idx)
        report(tag, r, block)
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    used = set(sum(train_index_blocks(len(train), N_JOB1_PASSES), []))
    fc_idx = targeted_indices(train, lambda e: e["block"] == "False consensus", exclude=used)
    cube_idx = targeted_indices(train, is_cube_or_mr, exclude=used)
    pers_idx = targeted_indices(train, lambda e: e["block"] == "Personality", exclude=used)
    for name, idx in (("fc", fc_idx), ("cube", cube_idx), ("pers", pers_idx)):
        np.save(OUT / f"{name}_indices.npy", np.array(idx))
        print(name, "targeted", len(idx))
    run(None, [("fc_base", fc_idx, "False consensus"), ("cube_base", cube_idx, "Cognitive tests")])
    run("lex_fc_weak", [("fc_weak", fc_idx, "False consensus")])
    run("lex_fc_neutral", [("fc_neutral", fc_idx, "False consensus")])
    run("lex_cube_v1", [("cube_swap", cube_idx, "Cognitive tests")])
    run("lex_pers_v1", [("pers_lex", pers_idx, "Personality")])


if __name__ == "__main__":
    main()
