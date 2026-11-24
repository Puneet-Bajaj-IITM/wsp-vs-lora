"""Local smoke test of the instrumentation on a TINY RANDOM model (never the real network)."""
import glob, os, sys, json
sys.path.insert(0, 'src'); sys.path.insert(0, 'wsp_work/twin_r4')
import numpy as np, torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
import wsp.harness as H, wsp.scoring as S
p = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/*/'))[0]
def fake_load(*a, **k):
    tok = AutoTokenizer.from_pretrained(p); tok.padding_side = "left"
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=96, intermediate_size=128, num_hidden_layers=28 if os.environ.get('FULL') else 3,
                      num_attention_heads=12, num_key_value_heads=2, max_position_embeddings=4096)
    torch.manual_seed(0)
    return tok, Qwen2ForCausalLM(cfg).eval()
H.load_model = fake_load
H._charge = lambda *a, **k: None
H.passes_used = lambda: 0
import common
common.PASS_SIZE = 24
import importlib
mod = importlib.import_module(sys.argv[1])
for k, v in json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}").items():
    if isinstance(v, dict): v = {int(a): tuple(b) for a, b in v.items()}
    setattr(mod, k, v)
with torch.no_grad(): mod.main()
