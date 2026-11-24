"""Job 1: dump layer-0 weights + prompt-vocab embeddings, and run embedding-injection probes."""
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab
from wsp.scoring import render_prompts

OUT = Path(__file__).parent / "job1"
OUT.mkdir(exist_ok=True)
SAMPLE_SEED = 11
LAST_PASSES = 3
GLOBAL_N = 384
PASS_N = 512


def dump_weights(lab: Lab) -> None:
    tok, model = lab.tok, lab.model
    train, dev = lab.examples("train"), lab.examples("dev")
    ids_train = [tok(p)["input_ids"] for p in render_prompts(tok, train)]
    ids_dev = [tok(p)["input_ids"] for p in render_prompts(tok, dev)]
    vocab = sorted({i for seq in ids_train + ids_dev for i in seq})
    emb = model.get_input_embeddings().weight
    layer = model.model.layers[0]
    attn, mlp = layer.self_attn, layer.mlp
    gate, up, down = mlp.gate_proj.weight.float(), mlp.up_proj.weight.float(), mlp.down_proj.weight.float()
    arrays = {
        "vocab": np.array(vocab), "emb_vocab": emb[vocab].cpu().numpy(),
        "emb_norm_all": emb.float().norm(dim=1).cpu().numpy(),
        "emb_mean": emb.float().mean(0).cpu().numpy(),
        "q_w": attn.q_proj.weight.cpu().numpy(), "q_b": attn.q_proj.bias.cpu().numpy(),
        "k_w": attn.k_proj.weight.cpu().numpy(), "k_b": attn.k_proj.bias.cpu().numpy(),
        "v_w": attn.v_proj.weight.cpu().numpy(), "v_b": attn.v_proj.bias.cpu().numpy(),
        "o_w": attn.o_proj.weight.cpu().numpy(),
        "ln1": layer.input_layernorm.weight.cpu().numpy(), "ln2": layer.post_attention_layernorm.weight.cpu().numpy(),
        "gate_norm": gate.norm(dim=1).cpu().numpy(), "up_norm": up.norm(dim=1).cpu().numpy(),
        "down_norm": down.norm(dim=0).cpu().numpy(),
    }
    np.savez(OUT / "weights.npz", **arrays)
    (OUT / "ids.json").write_text(json.dumps({"train": ids_train[:30], "dev": ids_dev[:30]}))
    cfg = model.config
    info = {"heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads, "eps": cfg.rms_norm_eps,
            "o_bias": attn.o_proj.bias is not None, "act": cfg.hidden_act, "cfg": str(cfg.to_dict()),
            "params": sum(p.numel() for p in model.parameters()),
            "tied": model.get_input_embeddings().weight is model.get_output_embeddings().weight}
    (OUT / "info.json").write_text(json.dumps(info))
    print("info", info, "vocab", len(vocab), flush=True)


def run(edit: str, indices: list[int], tag: str) -> None:
    lab = Lab(edit=edit)
    logps = []
    for start in range(0, len(indices), PASS_N):
        r = lab.forward("train", indices[start:start + PASS_N], note=f"job1 {tag}")
        logps.append(r["logp"])
        print(tag, r["summary"], "left", r["passes_left"], flush=True)
    np.save(OUT / f"logp_{tag}.npy", np.concatenate(logps))
    np.save(OUT / f"idx_{tag}.npy", np.array(indices))
    del lab
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    lab = Lab(edit=None)
    dump_weights(lab)
    n_train = len(lab.examples("train"))
    del lab
    gc.collect()
    torch.cuda.empty_cache()
    order = list(range(n_train))
    random.Random(SAMPLE_SEED).shuffle(order)
    run("inj_last", order[:LAST_PASSES * PASS_N], "inj_last")
    for edit in ("inj_opt1", "inj_opt4", "inj_optcolon2"):
        run(edit, order[:GLOBAL_N], edit)


if __name__ == "__main__":
    main()
