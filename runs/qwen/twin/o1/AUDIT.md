# Audit: qwen twin o1

Final message of the auditor agent, recovered from its transcript.

Verdict for twin/o1 (edit `optprior`)

- compliant: true
- issues: none that break a rule. There are minor notes below.
- ledger_passes: 24 (16 train, 8 dev; largest n is 512)
- numbers_verified: true
- verified_dev_acc: 0.316875
- verified_dev_nll: 1.52756

**1. Forbidden methods: none found.** I read and grepped every .py in wsp_work/twin_o1 and edits/o1 for backward, autograd, optim, peft/lora, sklearn, lstsq, solve, pinv, logistic, requires_grad, `.model(` and score_letters. There are no hits apart from docstrings that say "no solver".
- Every network pass goes through lab.forward: step1.py:23, step2.py:36,81,108 and step3.py:36,60,83.
- The hooks in optstats.py:50-79 only record activations inside charged lab.forward passes. They return None, so they do not change outputs.
- The weight statistics are simple: mean direction, a b-weighted difference of means, a diff-of-means down column, and a std-match (step2.py:44-64). They are not a label-fitted solver.
- analyze3.py:36 runs an SVD/PCA cross-covariance readout. It was explored offline, rejected as close to least squares (LOG.md:30-31), and not used in the edit.

**2. Test isolation: clean.** No script references a test split. The only "test" matches are the block name "Cognitive tests" in LOG.md. Data is read only from data/processed/{train,dev}.jsonl. load_items reads only raw/question_catalog.json (catalog.py:48-49, config.py:17).

**3. Ledger: OK.** results/o1/wsp_ledger.json has 24 entries, split train/dev only, every n at most 512. The entries line up with the job-1, job-2 and sweep plan in the LOG, including the early-break logic at step2.py:87.

**4. Edit integrity: OK.**
- edits/o1/optprior.py is byte-identical to edit_template.py.
- It loads its params relative to `__file__` (line 7).
- It changes only existing tensors, in place: index-assignment of gate row, up row and down column (lines 16-18), and `norm.weight.mul_` (line 19). It has no randomness.
- optprior_params.pt holds one injection: layer 20, neuron 7, with three float32 vectors of size 1536, and norm_scale 0.2 (1/5).
- That gives 3 × 1536 + 1536 = 6144 weights, which matches the report.
- The neuron was chosen as the argmin of |down|·|up| (step2.py:60-61).

**5. Numbers: confirmed.** I recomputed with numpy on CPU from wsp_work/twin_o1_cache/arrays2/dev_logp_optprior.npy (1600×5, rows normalised) against data/processed/dev.jsonl: acc 0.316875, NLL 1.52756. Base from arrays1/dev_logp.npy is 0.29375 / 2.67397. The cached dev labels match the jsonl.

**6. Independence: OK.** No script or LOG line refers to other runs' paths (top-level wsp_work/ or edits/ files, r2, r3, results/wsp_ledger.json, results/r2, results/r3, WSP_BRIEF.md). The only "r2" hits are the variable name in analyze3.py:41. All paths point at twin_o1, twin_o1_cache or edits/o1, the last through config.EDITS_DIR.

**Notes (borderline, not violations):**
- **Disclosed label table:** b(text) is built from train label counts (make_offsets.py:26, offsets.json) and is folded into the up row as a b-weighted difference of means (step2.py:51-54). The programmer says some texts are item-specific, so it partly acts as a per-item prior.
- **In-sample selection:** layer and alpha were picked on train set B, which was also one of the two sets used to compute w and d (step2.py:44-90). This is train-only, so no leak to dev or test. Only T (norm scale) was picked by a 1-D NLL grid on set B's logp (step2.py:93-94).
- **Offline dev use:** the offline scripts (analyze1-3, make_offsets.py:31-33) printed dev metrics while design choices were being made, for example the MIN_SHOWN/text restriction. The brief does not forbid using dev this way.
- **Unused code:** step3.py and injection.py never ran; the ledger has no job-3 entries.
- **Small LOG rounding:** L12 a2 is logged as .313, while sweep.json has 0.3125.
