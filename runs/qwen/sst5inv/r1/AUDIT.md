# Audit: qwen sst5inv r1

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**issues:**
- (none)

**ledger_passes:** 44

**ledger_splits:**
- train
- dev

**numbers_verified:** True

**edit_loads_and_is_deterministic:** True

**verified_dev_acc:** 0.4814

**verified_dev_nll:** 1.2827

**notes:**

I found no rule violations, and every reported number matches the saved files.

1. Forbidden methods: none. I grepped all 9 .py files in wsp_work/sst5inv_r1 and both files in edits/sst5inv/r1. There is no backward, optim, peft or LoRA, sklearn, lstsq, solve, pinv, score_letters, generate or direct model(...) call. The only network forward is lab.forward in labutil.py:37.
   - The step scripts only use statistics: means, differences of means, and one SVD of 5 letter means (step4.py:86) used to project the letter subspace off the direction.
   - The temperature is chosen by a 19-point 1-D grid over logged train log-probs (step4.py:123-129), which is not a solver.
   - Borderline, but disclosed in the report: step2.py:51-58 registers forward pre-hooks on mlp.down_proj. They only read activations during a ledgered lab.forward pass (ledger note "swap capture all layers + mlp keys").
   - step1.py uses render_prompts and tok.encode only to tokenise.
2. Test isolation: grepping for "test" across WORK, EDITS and LOG.md finds nothing. analyze1.py and analyze2.py read only data/processed/sst5inv/train.jsonl and dev.jsonl.
3. Ledger: 44 entries (50 or fewer). Splits are train and dev only, every n is 512 or less, and the largest is 512.
   - Train: 35 passes of 512.
   - Dev: 9 passes, in sets of 512+512+77 for base, swap_polarity and swap_steer_temp.
4. Edit: swap_steer_temp.py is deterministic and has no randomness. It runs under torch.no_grad, mutates only existing parameters in place, and loads its 3 .npy files relative to __file__ (HERE / f"{STEM}_*.npy").
   - Constants: steer has shape (1536,), key_idx is (512,) with 512 unique values, and key_val is (512,).
   - Counted from the code, it changes 2×1536 + 1536×512 + 1536 = 791,040 weights (0.0512% of 1,543,714,304), which matches the report. The 2×1536 are the embedding rows of tokens 8225 and 6785, the 1536×512 are layers[20].mlp.down_proj[:, cols], and the 1536 are model.norm.weight scaled by 0.25.
   - The local edit files have the same MD5 hashes as the copies in kaggle_runner/out/wsp-sst5inv-r1/out/edits/sst5inv/r1/, so the dev numbers come from exactly these files.
5. Numbers: I recomputed them with numpy from wsp_work/sst5inv_r1/local_archive/job4_arrays.npz (final_dev_p0-2 logp and idx; 1101 unique indices) and the dev.jsonl labels.
   - Final edit: acc 0.48138, NLL 1.28273. These match final_dev_full.json in kaggle_runner/out.
   - Swap only: 0.44323 / 3.2504.
   - Base (job1_job2_arrays.npz): 0.04995 / 9.3956.
   - All match the report.
   - The variant and temperature were chosen in step4.py from train chunks A and B only. Dev was measured only after the choice was made.
6. Independence: no script or the LOG refers to another task, run, or edits/ or results/ folder. The only data paths are data/processed/sst5inv/{train,dev}.jsonl. kaggle_runner/out contains other wsp_work folders (for example wsp_work/job3), but they come from the bundler's sync, not from anything the programmer's scripts read.
