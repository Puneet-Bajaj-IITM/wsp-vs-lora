"""Step 1: architecture + tokenisation facts, base probe, polarity-swap probe, swap on full dev."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAIN_PROBE, full_dev, report, save  # noqa: E402

from wsp.harness import Lab  # noqa: E402
from wsp.scoring import render_prompts  # noqa: E402

CAPTURE = (8, 12, 16, 20, 23, -1)
WORDS = (" negative", " positive", " neutral", " very", "negative", "positive", "neutral", "very")
SWAP_GAIN_MIN = 0.05

lab = Lab(edit=None)
cfg = lab.model.config
print("CONFIG", cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.vocab_size,
      "tied", cfg.tie_word_embeddings,
      lab.model.get_input_embeddings().weight is lab.model.get_output_embeddings().weight, flush=True)
for word in WORDS:
    print("TOK", repr(word), lab.tok.encode(word, add_special_tokens=False))
prompt = render_prompts(lab.tok, lab.examples("train")[:1])[0]
print(repr(prompt))
print([lab.tok.decode([i]) for i in lab.tok.encode(prompt, add_special_tokens=False)], flush=True)

base = lab.forward("train", TRAIN_PROBE, note="base train probe", capture_layers=CAPTURE)
save("base_train", base)
report("base_train", base)

del lab
lab = Lab(edit="swap_v1")
swap = lab.forward("train", TRAIN_PROBE, note="swap_v1 train probe", capture_layers=CAPTURE)
save("swap_train", swap)
report("swap_train", swap)

if swap["summary"]["accuracy"] > base["summary"]["accuracy"] + SWAP_GAIN_MIN:
    full_dev(lab, "swap_v1", "swap_v1")
