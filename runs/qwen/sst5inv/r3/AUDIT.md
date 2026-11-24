# Audit: qwen sst5inv r3

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**issues:**
- (none)

**ledger_passes:** 18

**ledger_splits:**
- train
- dev

**numbers_verified:** True

**edit_loads_and_is_deterministic:** True

**verified_dev_acc:** 0.4841

**verified_dev_nll:** 1.2698

**notes:**

I found no rule violations in this run.

1. Forbidden methods: none. I read every .py in wsp_work/sst5inv_r3/ (analyze1, common, edit_writer, job1, job2, make_directions) and edits/sst5inv/r3/. There is no autograd, .backward, torch.optim, LoRA/peft, sklearn, lstsq or linalg.solve. All model runs go through lab.forward: job1.py:18, common.py:38 and job2.py:50. There are no direct model(...), score_letters or generate calls.
- The one solver is torch.linalg.pinv at _steerlib.py:48. It inverts the stacked o_proj-derived maps, which are weights only, to express a fixed target vector as v_proj bias shifts. It is not fitted to labels.
- The target is a neutral-vs-rest difference of class means (make_directions.py:17), which the rules allow.
- Three values were chosen against train labels by grid, as the report says: alpha (best train accuracy, job2.py:56), the layer group, and tau (minimum train NLL, job2.py:22-27). This is hyperparameter selection, and it was disclosed.

2. Test isolation: clean. The only data files opened are data/processed/sst5inv/train.jsonl (analyze1.py:12, make_directions.py:12). Nothing in the WORK scripts, the edits or LOG.md refers to test.jsonl, a test split or results/**/*test*.

3. Ledger: 18 entries, within the limit of 50, which matches passes_used=18. Splits are train (9 passes, n=512 each) and dev (9 passes: 3 full-dev sweeps of 512+512+77). Every n is 512 or less.

4. Edit: final_steer.py loads _steerlib.py through Path(__file__).resolve().parent, and _steerlib.py loads neutral_dir_h20.npy through HERE / direction_file. Both are relative to __file__.
- apply only mutates existing parameter tensors in place:
  - swaps rows 8225 and 6785 of the tied embedding (_steerlib.py:21-25);
  - adds to v_proj.bias of layers 17-20 (_steerlib.py:53);
  - multiplies model.model.norm.weight by 0.25 (_steerlib.py:61).
- apply has no torch.no_grad of its own, but the harness wraps the call in one (harness.py:54-55), so this is fine.
- There is no randomness and no data loading; the pinv runs in double precision on CPU.
- Weight count from the code: 2×1536 (swap) + 4×256 (v biases, GQA has 2 kv heads × 128) + 1536 (norm gain) = 5632. This matches the report. The 3.6e-6 fraction holds if the model has about 1.54B parameters, which the programmer did not count.

5. Numbers: I recomputed from the saved logp arrays against the labels in data/processed/sst5inv/dev.jsonl, using numpy on CPU.

| edit | file | acc | NLL |
|---|---|---|---|
| final_steer | wsp_work/sst5inv_r3/j2_final_steer_dev_logp.npy (1101×5) | 0.484105 | 1.269785 |
| swap_temp | j2_swap_temp_dev_logp.npy | 0.4423 | 1.3045 |
| swap_polarity | j1_dev_logp.npy | 0.4432 | 3.2507 |

final_steer matches the reported 0.4841 / 1.2698 and j2_report.json. The other two match their LOG.md entries. The rows are normalised log-probabilities.

6. Independence: no script and no LOG entry references another task's or run's folders. The scripts only use their own WORK, edits/sst5inv/r3 and wsp_work/sst5inv_r3_cache. The programmer says they listed other runs' file names and sizes under edits/ to diagnose the bundle-size failure. That left no trace in the scripts, and since it was only a listing to fix a push error, I don't count it as a breach.

Minor, not violations: alpha was selected on the same 512 train examples the direction came from, and three edits were measured on full dev before the best was reported. Both were disclosed.
