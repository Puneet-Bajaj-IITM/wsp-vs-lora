import json, os, sys, collections
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
from wsp.scoring import render_prompts
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
J = "wsp_work/facts_r1/job2/"
logp = np.load(J + "logp_diag_train.npy"); idx = np.load(J + "idx_diag_train.npy")
exs = [json.loads(l) for l in open("data/processed/facts/train.jsonl")]
exs = [exs[i] for i in idx]
lens = np.array([len(tok(p)["input_ids"]) for p in render_prompts(tok, exs)])
finite = np.isfinite(logp).all(1)
longest = np.zeros(len(exs), bool)
for s in range(0, len(exs), 16):
    longest[s:s + 16] = lens[s:s + 16] == lens[s:s + 16].max()
print("finite & longest", int((finite & longest).sum()), "finite & not longest", int((finite & ~longest).sum()),
      "nonfinite & longest", int((~finite & longest).sum()), "nonfinite & padded", int((~finite & ~longest).sum()))
print("pad token", tok.pad_token, tok.pad_token_id, "len range", lens.min(), lens.max())
print("finite logp sample", logp[finite][:5].round(2), "labels", [e["label"] for e, f in zip(exs, finite) if f][:5])
