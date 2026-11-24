"""Job 1: tokenisation facts, base measurement, and the polarity-token swap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from labutil import PASS_SIZE, full_split, run_pass
from wsp.harness import Lab
from wsp.scoring import render_prompts

CAPTURE = (8, 12, 16, 20, 24, 27, -1)
TRAIN_A = range(0, PASS_SIZE)
PROBE_WORDS = [" negative", " positive", " neutral", " very", "negative", "positive", " mixed", " good", " bad",
               " great", " terrible", " okay", " Negative", " Positive"]

lab = Lab()
tok = lab.tok
ex = lab.examples("train")[0]
prompt = render_prompts(tok, [ex])[0]
ids = tok.encode(prompt)
print([tok.decode([i]) for i in ids[-60:]])
for w in PROBE_WORDS:
    print(repr(w), tok.encode(w, add_special_tokens=False))
emb = lab.model.get_input_embeddings().weight.float()
print("tied", lab.model.get_output_embeddings().weight is lab.model.get_input_embeddings().weight)
single = [w for w in PROBE_WORDS if len(tok.encode(w, add_special_tokens=False)) == 1]
rows = torch.stack([emb[tok.encode(w, add_special_tokens=False)[0]] for w in single])
print("norms", dict(zip(single, [round(float(x), 3) for x in rows.norm(dim=1)])))
unit = rows / rows.norm(dim=1, keepdim=True)
print("cos\n", (unit @ unit.T).cpu().numpy().round(2))
print([n for n, p in lab.model.named_parameters() if "layers.0." in n], [tuple(p.shape) for n, p in lab.model.named_parameters() if "layers.0." in n])

run_pass(lab, "train", TRAIN_A, "base on train A with captures", "base_trainA", CAPTURE)
full_split(lab, "dev", "base full dev", "base_dev")
del lab
torch.cuda.empty_cache()
lab = Lab(edit="swap_polarity")
run_pass(lab, "train", TRAIN_A, "swap_polarity on train A with captures", "swap_trainA", CAPTURE)
