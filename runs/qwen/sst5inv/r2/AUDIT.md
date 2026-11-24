# Audit: qwen sst5inv r2

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**issues:**
- (none)

**ledger_passes:** 24

**ledger_splits:**
- train
- dev

**numbers_verified:** True

**edit_loads_and_is_deterministic:** True

**verified_dev_acc:** 0.4423

**verified_dev_nll:** 1.3045

**notes:**

No rule violations found. Paths below are relative to the project root.

1. Forbidden methods: I read all 6 .py files in WORK (step1, step2, analyze1, analyze_bias, analyze_hidden, edit_template) and both files in EDITS. None of them uses autograd, .backward, torch.optim, LoRA/peft, sklearn, lstsq, linalg.solve or pinv. The only forward passes are lab.forward calls (step1.py:38,42,46,53 and step2.py:37). There is no direct model(...) call, no score_letters and no generate. step1.py:49-55 scales the ' neutral' embedding row on lab.model in place between passes and then restores it; that is arithmetic on weights only, so it is allowed. The analysis scripts use numpy statistics only: confusion counts, a temperature grid, offset counts, diff-of-means and AUC.

2. Test isolation: grep finds no reference to test.jsonl, a test split or results/ in any WORK or EDITS script or in LOG.md.

3. Ledger: results/sst5inv/r2/wsp_ledger.json has 24 entries, within the 50 limit. Splits are train and dev only. The largest n is 512; the dev tails are 77. Job 1 used 7 passes and job 2 used 17, matching the LOG.

4. Edit: in edits/sst5inv/r2/swap_temp.py, apply() works under torch.no_grad() and changes two existing tensors in place:
- It swaps the input-embedding rows of ' negative' and ' positive' (lines 31-33): 2 x 1536 = 3072 weights.
- It multiplies model.model.norm.weight by 0.2 (line 36): 1536 weights.
- That is 4608 weights in total, matching the report. NEUTRAL_WORD is None, so the replacement branch never runs.
- Token ids are found through the tokenizer. There is no randomness and no data or constant files, so __file__-relative loading does not apply. It is deterministic.
- One point for whoever writes this up: the embeddings are tied, so the swap also swaps the matching lm_head rows. Only the letters A-E are scored, so the metric is unaffected.

5. Numbers: I recomputed from wsp_work/sst5inv_r2/job2/logp_swap_temp_dev.npy (1101x5) and the dev labels with numpy: acc 0.4423, NLL 1.3045, exact match. The other two dev runs also match the LOG and the report:
- swap_v1 (job1/logp_dev.npy): acc 0.4432, NLL 3.2507.
- the 'ambiguous' variant: acc 0.4332, NLL 1.3669.

6. Independence: the scripts read only data/processed/sst5inv/{train,dev}.jsonl, their own job1/job2 folders and wsp_work/sst5inv_r2_cache/ (this run's own unbundled cache, analyze_hidden.py:9). Nothing reads another task's or run's edits/, wsp_work/ or results/ folder.

Borderline items, all disclosed in the report and not violations:
- The 0.2 temperature is one global scalar picked by a train-NLL grid.
- The neutral-word candidate was picked on train with a pre-set margin and then rejected on dev.
- The final choice of swap_temp over swap_v1 was made after seeing both dev results.
- The step2 candidate edit files (swap_temp_neu_*.py, made from edit_template.py) were deleted and are not in EDITS, so I could not inspect them. Their generator template is clean.
