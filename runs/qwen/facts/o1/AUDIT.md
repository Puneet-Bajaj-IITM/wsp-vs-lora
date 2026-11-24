# Audit: qwen facts o1

Final message of the auditor agent, recovered from its transcript.

Audit verdict for facts/o1, edit mem_s2

compliant: true
ledger_passes: 44
numbers_verified: true
verified_dev_acc: 0.7300 (n=900; by block city 0.900, profession 0.663, hobby 0.627)
verified_dev_nll: 1.1633

issues: none that break the rules. Minor points:
- The base dev figures (0.194 / 3.634) cannot be checked from this run. The ledger has no base dev pass, so those numbers come from outside the run.
- apply() does not open torch.no_grad() itself. The harness calls it under no_grad (src/wsp/harness.py:54-55), so it works as used.
- LOG.md:47 says the "dev/test templates" cue the attribute before the name. That comes from the public builder source (src/wsp/tasks/facts.py:24-33, SPLIT_TEMPLATES). Neither the test split nor test.jsonl was read.

Checks:
1. Forbidden methods: none found. I read all 15 .py files in wsp_work/facts_o1/ and both in edits/facts/o1/, and grepped for backward, autograd, optim, peft, LoRA, sklearn, lstsq, solve, pinv, requires_grad and direct model() calls: no hits.
   - Every network pass goes through lab.forward: job1.py:61,102; capture.py:73; job2.py:40,89; job3.py:45,56; job4.py:36,47.
   - Hooks are registered just before a charged forward and removed after it (common.py:72-103).
   - Construction is closed-form (build_edit.py):
     - keys are centered class means projected off the mean and scaled by their squared norm (:40-47);
     - gates are BETA_G*(k - TAU*c) (:132);
     - down columns are beta times the centered option-token mean of each person's true value (:134), divided by a fixed SiLU scale.
   - attrdir.py:10-23 is a difference of class means (used only for mem_g1).
   - Neuron choice uses the lowest weight-norm product (:50-53).
   - Picking the best oracle by max accuracy on train (job2.py:73, job3.py:83-84) is selection, not fitting.
   - The per-person table from train labels (common.py:115-120 into build_edit.py:134) is written into the weights and is disclosed in LOG.md.
2. Test isolation: clean. No script mentions test.jsonl or results/*test*. Data comes only from lab.examples("train"/"dev") and load_split("train").
3. Ledger (results/facts/o1/wsp_ledger.json): 44 entries (34 train, 10 dev), all at most 512 examples (largest 512; the dev halves are 512+388). Within the budget of 50.
4. Edit integrity:
   - mem_s2.py:1-11 loads _memlib.py relative to __file__.
   - _memlib.py:9-17 loads mem_s2_mem.pt relative to __file__ and does in-place index assignment into the existing layer-8 gate_proj, up_proj and down_proj weights. No architecture change.
   - It is deterministic: stored constants only, no randomness.
   - The .pt holds one block: layer 8, 900 unique neuron indices (4..8959), gate 900x1536, up 900x1536, down 1536x900, fp16, all finite. That is 3 x 900 x 1536 = 4,147,200 weights, which matches the claim (10% of the layer-8 MLP, about 0.27% of the model).
   - The .pt is byte-identical (md5 f17e43e2...) to the in-job copy in wsp_work/facts_o1_cache/j5/edit_copies/, and mem_s2.py is identical too, so the shipped file is the one that was measured.
5. Numbers: I recomputed from wsp_work/facts_o1/dev_logp_mem_s2.npy (900x5, rows sum to 1) against the labels in data/processed/facts/dev.jsonl with numpy on CPU. Result: acc 0.7300, NLL 1.1633, which matches dev_mem_s2.json and the report. The other candidates recompute as:

| Edit | Dev acc | Dev NLL |
|---|---|---|
| mem_s1 | 0.713 | 1.442 |
| mem_g1 | 0.654 | 1.394 |
| mem_s3 | 0.690 | 1.534 |
| mem_v1 | 0.508 | 1.966 |

6. Independence: clean. Paths are only WORK, WORK/../facts_o1_cache (common.py:13-14) and edits/facts/o1 (job2.py:23, job3.py:18, job4.py:23). Nothing points at facts_r*, other runs or other tasks.

notes on dev use:
- 10 of 44 passes were dev: 5 candidate edits (v1, s1, g1, s2, s3), each on full dev, and the best was picked by dev accuracy.
- Dev also shaped the design. The local per-block, per-fallback diagnosis of mem_s1 on dev (LOG.md "Local diagnosis of mem_s1 on dev") led to the s2 layout (hobby moved to offset -2) and to the s3 variant.
- The reported 0.730 is therefore mildly optimistic from selection. mem_s2 beats mem_s1 by 1.7 points, inside the bootstrap 95% CI of [0.699, 0.758].
- All key and value capture and all oracle tuning used train only.
