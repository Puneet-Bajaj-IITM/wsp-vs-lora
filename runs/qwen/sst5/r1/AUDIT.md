# Audit: qwen sst5 r1

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**issues:**
- Minor, not a violation: wsp_work/sst5_r1/LOG.md:4 reads results/sst5/eval_base_dev.json. This is the task-level base dev result shared by all runs of sst5, not another run's folder, and it is not a test file.
- Minor, transparency: 18 finished scripts (job1-4, an1-8, an_job2, build_vecs_job2-4, wlib, circuits) exist only inside wsp_work/sst5_r1/old_scripts.tgz, not as plain .py files in WORK. I extracted and read all of them; none contains a forbidden method, a test reference or a path into another run.
- Cosmetic: edits/sst5/r1/sst5_circuit.py loads unused layer-22 constants (s_very_22, d_very_22, one_22), because DEFAULT qv_layers=(21,). They have no effect.

**ledger_passes:** 49

**ledger_splits:**
- train
- dev

**numbers_verified:** True

**verified_dev_acc:** 0.4741

**verified_dev_nll:** 1.2906

**edit_loads_and_is_deterministic:** True

**notes:**

The run is compliant, and the reported numbers match the saved outputs.

1. Forbidden methods: none found. I read every .py in wsp_work/sst5_r1 and edits/sst5/r1, plus the 18 scripts in old_scripts.tgz (extracted to the scratchpad). A grep for backward, optim, sklearn, lstsq, solve, pinv, peft, lora, LogisticRegression, requires_grad, model(, score_letters and generate matched nothing. Every forward computation goes through lab.forward:
   - trial.py:41 builds a variant, runs lab.forward, then restores the snapshot.
   - The archived job2/3/4 do the same through wlib.NeuronWriter.
   - Archived job1.py:41-47 only renders and tokenizes a prompt; it runs nothing through the model.
   The numpy statistics used are class means, soft predicted-class means, an SVD of class means, nearest-centroid diagnostics, percentile threshold scans and grids over a dose-response simulation. These are the scalar fits the programmer already disclosed; none is a gradient or a regression/classifier solver. Archived an3.py builds a Gaussian nearest-centroid score, offline and as a diagnostic only, matching borderline item 5.

2. Test isolation: no reference to test.jsonl, a "test" split or results/**/*test* in any script or in LOG.md. Data reads are data/processed/sst5/train.jsonl (make_stats.py:18, archived an1-4) and dev.jsonl (an10.py:9, for dev labels only).

3. Ledger (results/sst5/r1/wsp_ledger.json): 49 entries, which is within the limit of 50.
   - 46 are train passes of n=512.
   - 3 are dev passes: 512, 512 and 77 (base).
   - 3 are dev passes: 512, 512 and 77 (edit).
   No n exceeds 512, and the dev passes are only the job-7 final measurements.

4. Edit (edits/sst5/r1/sst5_circuit.py):
   - apply() runs under torch.no_grad() and mutates existing tensors in place: L19 gate/up rows and down columns (lines 72, 79-81), L21 q_proj.weight via `w += delta` (line 122), and model.norm.weight via `/=` (line 129).
   - It is deterministic: no randomness, and spare neurons are picked by argsort of weight norms. Constants load from HERE = Path(__file__).resolve().parent (lines 16, 45-47).
   - The constants come from train captures only (build_edit_consts.py).
   - Expected count from the code: 3 rewritten neurons give gate+up 2x3x1536 = 9,216; 4 down columns give 6,144; the rank-1 L21 q_proj update touches up to 2,359,296; the final norm adds 1,536.
   - The job-7 Kaggle log (kaggle_runner/out/wsp-sst5-r1/wsp-twin-wsp-sst5-r1.log) measured gate 4608 + up 4608 + down 6143 + q_proj 2,321,498 + norm 1536 = 2,338,393 of 1,543,714,304 parameters (0.1515%). This matches the report.

5. Numbers: I recomputed from wsp_work/sst5_r1/job7_dev.npz against the data/processed/sst5/dev.jsonl labels, on CPU with numpy.
   - Base: acc 0.4233, NLL 3.4362.
   - Edit: acc 0.4741, NLL 1.2906.
   - Paired: 143 items fixed, 87 broken.
   All match the report and the log line "FULL DEV sst5_circuit: acc 0.4741 nll 1.2906 n 1101".

6. Independence: no script or LOG path points into another task's or another run's edits/, wsp_work/ or results/ folder. The Kaggle out bundle does contain other tasks' folders (sst5inv_r1, twin_r2, facts_r1), but that comes from the shared bundler, and nothing in this run's code reads them.
