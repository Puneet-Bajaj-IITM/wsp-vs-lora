"""Local code test of the probe on a TINY RANDOM network (not the study model, no study weights)."""
import json, sys, types
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "wsp_work/twin_p1")
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from wsp.scoring import render_prompts
import probe as P

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct"); tok.padding_side = "left"
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=12, num_key_value_heads=2, rope_theta=1000000.0)
model = Qwen2ForCausalLM(cfg).eval()
exs = [json.loads(l) for l in open("data/processed/train.jsonl")][:6]
lab = types.SimpleNamespace(model=model, tok=tok, letters=torch.tensor([32, 33, 34, 35, 36]))
pr = P.Probe(lab)
prompts = render_prompts(tok, exs)
tr = [P.option_positions(tok, p) for p in prompts]
pr.reset([t[0] for t in tr], [t[1] for t in tr], [t[2] for t in tr]); pr.attach()
with torch.no_grad():
    for s in range(0, 6, 4):
        batch = tok(prompts[s:s + 4], return_tensors="pt", padding=True)
        model(**batch, output_hidden_states=True, logits_to_keep=1)
pr.detach()
print("errors", pr.errors); print("validation", pr.validation)
for k in ("att", "kt", "dla", "x", "q", "att_content_max", "xr", "kx"):
    print(k, pr.stacked(k).shape)
print(pr.stacked("att")[0, 0, 0])
