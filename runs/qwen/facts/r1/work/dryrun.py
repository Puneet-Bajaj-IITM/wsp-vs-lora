"""Weight-level checks of the circuit design (no prompts are run through any network)."""
import os, sys
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
sys.path.insert(0, "edits/facts/r1")
import _lk_build as B, _lk_design
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
W = np.load("wsp_work/facts_r1/job1/weights.npz")
design = _lk_design.load(tok)
vocab = list(W["vocab"]); row = {t: i for i, t in enumerate(vocab)}
rng = np.random.default_rng(0)
w = {"emb_vocab": W["emb_vocab"].astype(float), "emb_letters": W["emb_vocab"].astype(float)[[row[t] for t in design.letter_tokens]],
     "ln1": W["ln1"].astype(float), "ln2": W["ln2"].astype(float), "ln1_next": W["ln1"].astype(float),
     "ln_final": 3 + rng.random(1536)}
b = B.build(w, design); d = b.diag; p = b.patches
print({k: float(d[k]) for k in ("beta", "flag_rms", "u_norm", "fill_norm", "gate_max", "down_max")})
print("edited emb norms pct", np.percentile(d["emb_edit_norm"], [0, 50, 100]).round(2))
xh = d["xh"]; a = p["l0_attn"]
k63 = xh @ a["k_w"][63]
tagged = np.array([t in design.color for t in vocab])
print("k63 tagged", k63[tagged].min(), k63[tagged].max(), "untagged max abs", np.abs(k63[~tagged]).max())
val = xh @ a["v_w"].T
col = np.array([design.color.get(t, -1) for t in vocab])
onehot = np.zeros((len(vocab), design.num_colors)); onehot[tagged, col[tagged]] = 1
print("colour code err", np.abs(val[:, :design.num_colors] - onehot).max())
slot = design.num_colors + 1
tg = np.zeros((len(vocab), 8)); 
for i, t in enumerate(design.letter_tokens + design.probe_tokens): tg[row[t], i] = 1
print("dual err", np.abs(val[:, slot:slot + 8] - tg).max(), "dual norms", np.linalg.norm(d["duals"], axis=1).round(3))
# positional profile of BACK2
th = d["theta"]; q = d["q_back"]; kb = d["k_bias"]
qc = q[:64] + 1j * q[64:]; kc = kb[:64] + 1j * kb[64:]
prof = np.array([np.real(qc * np.conj(kc) * np.exp(1j * delta * th)).sum() / np.sqrt(128) for delta in range(0, 120)])
print("back2 profile", prof[:8].round(1), "best other", np.delete(prof, 2).max().round(1), "at", int(np.argmax(np.where(np.arange(120) == 2, -1e9, prof))))
# MLP logic check, composed from designed messages (weights only)
sizes = d["sizes"]; dirs = d["dirs_all"]; ncol = sizes[0]
gcol, gconst, gval = dirs[:ncol], dirs[ncol], dirs[ncol + 1:ncol + 1 + sizes[2]]
ell = dirs[ncol + 1 + sizes[2]:ncol + 6 + sizes[2]]; flag = dirs[-1]
m = p["l0_mlp"]; emb_ids, emb_rows = p["emb_rows"]; erow = {t: i for i, t in enumerate(emb_ids)}
shared = [design.color[_lk_design.single(tok, s)] for s in _lk_design.SHARED_PIECES]
worst_fire, worst_quiet, flags = 1e9, -1e9, []
for person in range(300):
    for extra in ((0, 0), (1, 0), (0, 1), (1, 1)):
        counts = design.counts[person].copy(); counts[0] = 1
        for c, x in zip(shared, extra): counts[c] += x
        bag = counts / counts.sum()
        for v in range(30):
            e = emb_rows[erow[design.value_tokens[v]]]
            x = e + B.Z_PIN * d["z_dir"] + B.MU * (bag @ gcol + gconst) + B.LETTER_AMP * ell[0]
            xn = w["ln2"] * x / np.sqrt((x ** 2).mean())
            z = m["gate"] @ xn; upv = m["up"] @ xn
            should = np.zeros(900, bool)
            for a_i in range(3):
                if design.fact_value_index[person, a_i] == v: should[person * 3 + a_i] = True
            if should.any():
                worst_fire = min(worst_fire, z[should].min())
                act = z * upv / (1 + np.exp(-z))
                flags.append(float((m["down"].T @ act) @ flag))
            worst_quiet = max(worst_quiet, z[~should].max())
print("min firing z", worst_fire, "max non-firing z", worst_quiet, "flag amp range", min(flags), max(flags))
print("write norms", np.linalg.norm(d["write"], axis=1).round(1), "logit table\n", (d["readout"] @ d["write"].T * np.sqrt(1536) / np.linalg.norm(d["write"], axis=1)).round(2))
