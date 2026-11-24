"""Job 1: base model on 1024 train examples, residual captured at every layer; weight read-outs."""
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

OUT_DIR = Path(__file__).resolve().parent / "job1_out"
NUM_LAYERS = 28
FINAL_NORM = -1
CAPTURE_LAYERS = tuple(range(NUM_LAYERS)) + (FINAL_NORM,)
PASS_SIZE = 512
NUM_PASSES = 2
CLASS_WORDS = (" very", " negative", " neutral", " positive", "very", "negative", "neutral", "positive")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = Lab(edit=None)
    logps, hiddens = [], []
    for p in range(NUM_PASSES):
        idx = list(range(p * PASS_SIZE, (p + 1) * PASS_SIZE))
        r = lab.forward("train", idx, note=f"job1 base train {idx[0]}-{idx[-1]} all-layer capture",
                        capture_layers=CAPTURE_LAYERS)
        print(r["summary"], r["passes_left"], flush=True)
        logps.append(r["logp"])
        hiddens.append(np.stack([r["hidden"][layer] for layer in CAPTURE_LAYERS], 1).astype(np.float16))
    np.save(OUT_DIR / "logp.npy", np.concatenate(logps))
    np.save(OUT_DIR / "hidden.npy", np.concatenate(hiddens))

    model, tok = lab.model, lab.tok
    unembed = model.lm_head.weight.float().cpu()
    np.save(OUT_DIR / "letter_unembed.npy", unembed[lab.letters.cpu()].numpy())
    np.save(OUT_DIR / "final_norm_weight.npy", model.model.norm.weight.float().cpu().numpy())
    word_ids = [tok.encode(w, add_special_tokens=False) for w in CLASS_WORDS]
    print("class word ids", dict(zip(CLASS_WORDS, word_ids)))
    print("letter ids", lab.letters.tolist())
    sample = tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                                     tokenize=False, add_generation_prompt=True)
    print("template tail tokens", tok.convert_ids_to_tokens(tok.encode(sample))[-6:])
    for i, layer in enumerate(lab.layers):
        attn = layer.self_attn
        print(i, "b_v", round(attn.v_proj.bias.float().norm().item(), 3),
              "o_proj", tuple(attn.o_proj.weight.shape), "o_bias", attn.o_proj.bias is not None,
              "v_proj", tuple(attn.v_proj.weight.shape),
              "ln1", round(layer.input_layernorm.weight.float().mean().item(), 3))
    print("config heads", model.config.num_attention_heads, model.config.num_key_value_heads,
          "rms eps", model.config.rms_norm_eps)


if __name__ == "__main__":
    main()
