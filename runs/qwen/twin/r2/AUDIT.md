# Audit: qwen twin r2

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**issues:**
- (none)

**ledger_passes:** 20

**ledger_splits:**
- train
- dev

**numbers_verified:** True

**verified_dev_acc:** 0.458125

**verified_dev_nll:** 1.3926655

**edit_loads_and_is_deterministic:** True

**notes:**

I found no rule violations in this run.

1. Forbidden methods. I read every .py and every local_only/*.py.txt in WORK and EDITS. None of them uses autograd, .backward, torch.optim, LoRA, peft, sklearn, lstsq or pinv.
- The only solver is np.linalg.solve in wsp_work/twin_r2/local_only/kernel.py.txt:33. It is a ridge fit of RoPE cos/sin coefficients to a synthetic clipped ramp over token offset 0..1599. It uses no data or labels, so the rule against label-trained solvers does not apply. The programmer disclosed it.
- The only direct model(...) call is in local_only/test_toy.py.txt:47. It runs a RANDOM-weight Qwen2-shaped 4-layer toy built from Qwen2Config, not the real network. Disclosed.
- Every pass through the real model goes through lab.forward: job scripts step1/step2 (now local_only/*.py.txt) and step3a/b/c.py. job1 only tokenises with render_prompts and reads weights.
- The tables come from TRAIN labels only (tables.py.txt:19, build_edit.py.txt uses load_split("train")). This is disclosed as a label-derived per-item-group prior written into the embedding columns.

2. Test isolation. No reference to test.jsonl, a test split or results/**/*test* in any script or LOG.md. The only "test" strings are test_toy and block names ("Cognitive tests"). Data reads are only data/processed/{train,dev}.jsonl.

3. Ledger. results/r2/wsp_ledger.json has 20 entries, all train or dev, every n at most 512 (the 64s are the dev tail chunks).

4. Edit (edits/r2/prior_relay.py):
- Constants load through HERE = Path(__file__).resolve().parent (lines 19 and 61-68).
- It only does in-place index assignment or mul_ on existing parameters: embed_tokens columns, q/k/v/o_proj weights and biases in layers 0-2, o_proj and down_proj rows in all layers, input_layernorm and final norm weights. There are no new modules.
- It is deterministic: no randomness, and private_dims comes from an argsort of the base embedding's column energies. It loads no data.
- Two small points that are not violations. It sets the plain attribute model.wsp_private_dims (line 232), which is not a parameter. apply does not wrap itself in torch.no_grad and relies on the harness for that; the Kaggle runs succeeded.
- Weights changed: my estimate from the code is about 32.4M. The breakdown: 104 embedding columns × 151936 ≈ 15.8M; cleared KV groups in layers 0-2 ≈ 13.7M; silenced rows across layers ≈ 2.9M; the final norm adds 1536. The empirical diff saved in job3b.npz is 32,258,170 of 1,543,714,304 (0.020896), which matches the report exactly.

5. Numbers. Recomputed from wsp_work/twin_r2/job3c.npz (dev_logp has shape 1600×5): accuracy 0.458125, NLL 1.39266. The saved labels match data/processed/dev.jsonl exactly. The Kaggle log at kaggle_runner/out/wsp-twin-r2/wsp-twin-wsp-twin-r2.log prints "prior_relay(file) DEV FULL acc 0.458125 nll 1.3926655054092407". The v1 numbers check out from job3a.npz (0.450 / 1.3897). The gain file holds 25600, which is 1600 × 16, the multiplier chosen in job3b. These dev numbers come from the edit loaded from file (step3c.py uses Lab(edit="prior_relay")).

6. Independence. Scripts and LOG reference only wsp_work/twin_r2, edits/r2, data/processed and a local HF tokenizer cache. None reads wsp_work/*.md or *.py at top level, edits/*.py at top level, results/wsp_ledger.json or WSP_BRIEF.md. Kaggle's out/ directory has other runs' folders because the harness bundles all of wsp_work. That is not the agent reading them; LOG.md mentions them only to explain the bundle-size problem.
