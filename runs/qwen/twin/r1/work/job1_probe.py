"""Job 1: base-model captures on train (for closed-form derivations) + one lexical-swap probe.

Passes: 5 x 512 train (base, capture layers 20/27/-1), 1 x 512 train (lexswap_v1). Total 6.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (LIKERT, WORK_DIR, block_accuracy, nll, option_texts, pick_text_hist,  # noqa: E402
                    save_pass, train_index_blocks)
from wsp.harness import Lab  # noqa: E402
from wsp.scoring import letter_token_ids  # noqa: E402

OUT = WORK_DIR / "job1"
N_BASE_PASSES = 5
CAPTURE = (20, 27, -1)
PROBE_TOKENS = [" strongly", " little", " a", " Agree", " Disagree", " Neither", " Strongly", " Somewhat",
                " support", " oppose", "A", "B", "C", "D", "E", " slightly", " somewhat", " moderately"]
LAST_LAYER = 27


def dump_weights(lab: Lab) -> None:
    model, tok = lab.model, lab.tok
    letters = letter_token_ids(tok)
    emb = model.get_input_embeddings().weight
    tied = emb.data_ptr() == model.lm_head.weight.data_ptr()
    layer = model.model.layers[LAST_LAYER]
    np.save(OUT / "W_U_letters.npy", emb[letters].float().cpu().numpy())
    np.save(OUT / "final_norm_g.npy", model.model.norm.weight.float().cpu().numpy())
    np.save(OUT / "L27_o_proj.npy", layer.self_attn.o_proj.weight.float().cpu().numpy())
    np.save(OUT / "L27_v_bias.npy", layer.self_attn.v_proj.bias.float().cpu().numpy())
    np.save(OUT / "L27_norm_w.npy", layer.post_attention_layernorm.weight.float().cpu().numpy())
    tokens = {}
    for text in PROBE_TOKENS:
        ids = tok.encode(text, add_special_tokens=False)
        tokens[text] = {"ids": ids, "norm": float(emb[ids[0]].float().norm()) if len(ids) == 1 else None}
    row_norms = emb.float().norm(dim=1)
    meta = {
        "tied": bool(tied), "letters": letters, "tokens": tokens,
        "emb_row_norm_mean": float(row_norms.mean()), "emb_row_norm_median": float(row_norms.median()),
        "cfg": {k: getattr(model.config, k) for k in ("num_attention_heads", "num_key_value_heads", "hidden_size",
                                                      "intermediate_size", "num_hidden_layers", "rms_norm_eps")},
        "sample_tokenization": tok.tokenize("A. Agree strongly\nB. Agree a little\nC. Strongly oppose"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("META", json.dumps(meta)[:1500])


def check_capture(result: dict) -> None:
    w = np.load(OUT / "W_U_letters.npy")
    logits = result["hidden"][-1] @ w.T
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    print("capture(-1) reproduces logp? max abs diff", float(np.abs(logp - result["logp"]).max()),
          "| h27 rms mean", float(np.sqrt((result["hidden"][27] ** 2).mean(1)).mean()),
          "| h(-1) norm mean", float(np.linalg.norm(result["hidden"][-1], axis=1).mean()))


def report(tag: str, result: dict) -> None:
    exs = result["examples"]
    print(f"[{tag}] acc/block {block_accuracy(exs, result['logp'])} nll {nll(exs, result['logp']):.3f}",
          "letters", result["summary"]["pred_letter_hist"], "passes_left", result["passes_left"])
    print(f"[{tag}] Personality picks {pick_text_hist(exs, result['logp'].argmax(1), 'Personality')}")
    print(f"[{tag}] FalseConsensus picks {pick_text_hist(exs, result['logp'].argmax(1), 'False consensus')}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = Lab(None)
    blocks = train_index_blocks(len(lab.examples("train")), N_BASE_PASSES)
    dump_weights(lab)
    for i, idx in enumerate(blocks):
        r = lab.forward("train", idx, note=f"job1 base capture {i}", capture_layers=CAPTURE)
        save_pass(OUT, f"base{i}", r, idx)
        if i == 0:
            check_capture(r)
        report(f"base{i}", r)
    del lab
    gc.collect()
    torch.cuda.empty_cache()
    lab = Lab("lexswap_v1")
    r = lab.forward("train", blocks[0], note="job1 lexswap_v1 on block0")
    save_pass(OUT, "lexswap_v1", r, blocks[0])
    report("lexswap_v1", r)


if __name__ == "__main__":
    main()
