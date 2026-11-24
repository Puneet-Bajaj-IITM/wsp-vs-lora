"""Logic test of the circuit on a RANDOM-INIT 3-layer Qwen2-shaped mock (not the study model)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edits" / "facts" / "r2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _hj_build  # noqa: E402
from _hj_config import HijackConfig  # noqa: E402
from diagnose import diagnose  # noqa: E402

LOCAL_TOKENIZER = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MOCK_LAYERS = 3
N_EXAMPLES = 96
BATCH = 32
SEED = 0


def main(half_weights: bool):
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER)
    tok.padding_side = "left"
    conf = Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=MOCK_LAYERS,
                       num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(conf).eval()
    with torch.no_grad():
        _hj_build.install(model, tok, HijackConfig())
        if half_weights:
            for p in model.parameters():
                p.copy_(p.half().float())
    rows = [json.loads(l) for l in (ROOT / "data/processed/facts/train.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(SEED)
    exs = [rows[i] for i in rng.choice(len(rows), N_EXAMPLES, replace=False)]
    prompts = [tok.apply_chat_template([{"role": "system", "content": e["system"]}, {"role": "user", "content": e["user_message"]}],
                                       tokenize=False, add_generation_prompt=True) for e in exs]
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCDE"]
    logits, hidden = [], {0: [], 1: [], 2: []}
    with torch.no_grad():
        for s in range(0, len(prompts), BATCH):
            batch = tok(prompts[s:s + BATCH], return_tensors="pt", padding=True)
            out = model(**batch, output_hidden_states=True)
            logits.append(out.logits[:, -1, letters])
            for k in hidden:
                hidden[k].append(out.hidden_states[k + 1][:, -1])
    logits = torch.cat(logits)
    labels = torch.tensor([e["label"] for e in exs])
    logp = torch.log_softmax(logits, -1)
    print("half" if half_weights else "float", "acc", float((logits.argmax(1) == labels).float().mean()),
          "nll", float(-logp[torch.arange(len(labels)), labels].mean()))
    print(json.dumps(diagnose({k: torch.cat(v).numpy() for k, v in hidden.items()}, exs, _hj_build.LAST, 1), indent=1))


if __name__ == "__main__":
    main(half_weights="half" in sys.argv)
