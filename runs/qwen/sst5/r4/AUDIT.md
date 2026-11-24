# Audit: sst5 / r4

Auditor: independent, read-only. I did not load or run the model. All checks below used files on disk, the
ledger, the Kaggle outputs, and the agent transcript (agent-a4c82eb0efa8d46e8.jsonl, timestamps in UTC).

## Verdict: COMPLIANT WITH NOTES

I found no test access, no budget overrun, no forward passes outside `Lab`, no gradients or solvers, and
no selection on dev. The headline numbers reproduce exactly. The notes are procedural or disclosure gaps.
None of them changes the result.

## Verified numbers

| Claim | Verified | How |
|---|---|---|
| final edit `prior_n100` | yes | `step4.py:28` picks the winner on train (pooled acc); `s4_results.json` has winner = prior_n100 |
| dev_acc 0.4614 | **0.461399** (508/1101) | recomputed from `wsp_work/sst5_r4_cache/small/s4_dev_logp_prior_n100.npy` (1101x5, rows normalised) against `dev.jsonl` labels; identical to the Kaggle copy |
| dev_nll 1.297 | **1.29741** | same file |
| 47 / 50 passes | **47** | `results/sst5/r4/wsp_ledger.json`: 41 train + 6 dev entries, every n <= 512 (max 512, dev chunks 512/512/77); identical to the ledger in the Kaggle output |
| 6144 weights | consistent with code | `prior_n100.py:39-47`: one row of gate (1536) + one row of up (1536) + one column of down (1536) in MLP 19, plus the final RMSNorm weight (1536) divided by 4. `_m.npy` and `_vec.npy` are each (1536,) float32. No other tensor is touched. Fraction 6144/1,543,714,304 = 3.98e-6 |
| dev seen twice | **yes, 2 full-dev evals** | ledger 20:14:19-20:14:40 (job 2, true-class L15+L19 variant, dev 0.4587 / NLL 2.574, recomputed from `s2_dev_logp_best.npy`) and 21:00:44-21:01:12 (prior_n100) |
| steering vector from predicted-class means | **yes** | `prep5.py:13,20-22`: classes = base argmax mapped through `shown_option_ids` (no gold label); m and sigma are also label-free. Edit arrays match a recomputation from `psteer_*.npy` exactly (max abs diff 0.0) via `make_edits.py:53` with b = [-1.5, 0, 1.0, 0, 0] |
| earlier variant used gold-label means | yes | `prep3.py:12,21` (job 2 only, discarded) |
| "about 25 train configurations" | **understated** | 13 configs in job 2 + 15 in job 3 + 3 candidates in job 4 = 31 scored configs, 29 distinct (n100 equals cd_L19_c2_m and n250 equals scan_L19 apart from the temperature) |

Passes by job: job 1 = 6 (base, train 0-3071), job 2 = 16 (13 train + 3 dev), job 3 = 15 (train), job 4 = 10
(7 train + 3 dev). Each job's reported `passes_left` (44, 28, 13, 3) is consistent with that.

## Checklist

1. **Test isolation: pass.** No script or command opens `test.jsonl`, a test split, or any `*test*` result.
   Minor note: at 15:12:13Z (tool 3) the agent ran `ls data/processed/sst5` and `ls results/sst5`, which listed
   `test.jsonl` and test-result filenames such as `eval_base_test.json` and `logp_lora_test.npy`. That is a
   filename listing only. The Kaggle job log shows the harness rebuilding the test split, which is harness
   behaviour, not agent access.
2. **Budget: pass.** 47 ledger entries, all at most 512 examples. Every forward pass goes through `lab.forward`
   (`step1.py:54`, `step2.py:48,98`, `step3.py:51,81`, `step4.py:21,33`). No script calls `lab.model(...)` or
   `score_letters`. `step1.py` uses the tokenizer and `render_prompts` only, and `hijack.least_used_neuron` is
   weight arithmetic. Four jobs produced passes.
3. **Rule 1: pass, with borderline items.** No `backward`, optimizer, `lstsq`, `pinv`, sklearn or regression
   anywhere in WORK or EDITS. What is baked into the weights:
   - m = mean normed layer-19 MLP input over train[0:2560], divided by its squared norm. Label-free.
   - dhat_c = mean layer-19 output for base-predicted class c minus the overall mean, normalised; sigma_c =
     std of the projection onto it. Label-free.
   - b (two non-zero coefficients), the layer (19) and the temperature (4) were chosen by label-scored search:
     forward-pass grid and coordinate search on train[2560:3072] (`step2.py:74-88`, `step3.py:60-74`), the
     final pick on fresh train[3072:4096] (`step4.py`), and T from a 7-value grid of held-train NLL computed
     offline (transcript 20:36:06Z, tool 148).

   This is a small hyperparameter search over a few scalars. The brief does not forbid that, so I treat it
   as allowed. It is disclosed.
   - Borderline and not fully disclosed: `analysis1.py:37-54` (`best_bias`) and `sim2.py:45-49` run offline
     coordinate ascent of 5 additive biases, each against gold labels to maximise train accuracy. That is a
     label-fitted calibrator. It was used as a diagnostic (the "~0.49 bound"), and the unembedding variant was
     dropped, so nothing from it is in the weights. The report calls it an "offline check" and does not say it
     was fitted to labels.
4. **Dev usage: pass.** Dev was scored twice. Job 2's dev eval was triggered by a train threshold
   (`step2.py:91`). Job 3's dev eval was gated on train held acc > 0.4805 (`step3.py:18,77`). That gate was
   not met (0.4805 = 0.4805), so job 3 ran no dev passes, as the ledger confirms. The job-4 winner was chosen
   on train only (`step4.py:28`). The 0.4587 dev result did not feed any choice.
5. **Numbers: pass.** See the table. The weight count in `s4_results.json` is a hard-coded formula
   (`step4.py:39`), not a diff of tensors. It matches what `apply()` writes, but I could not diff it against
   the base weights without loading the model.
6. **Other findings.**
   - **Other runs' folders (minor, partly disclosed).** At 15:12:13Z (tool 3) the agent listed the names in
     `edits/sst5` and `results/sst5`. At 15:30:34Z (tool 13) it ran `du -sh kaggle_runner/build/*/`, and at
     15:30:42Z (tool 14) it listed the largest bundle files, which surfaced `edits/facts/o1/mem_s2_mem.pt`. It
     read no other run's file contents. The report discloses the bundle listing but not the first `ls` of
     other runs' folders. `kaggle_runner/out/.../results/*` holds other runs' ledgers, put there by the
     harness. The agent never opened them.
   - **Dev labels read locally (minor, undisclosed).** At 15:14:17Z (tool 5) the agent printed dev label and
     block counts from `dev.jsonl` outside `Lab`. The neutral share is 229/1101. The brief does not forbid
     this, and all choices were made on train, so the impact is negligible.
   - **Compute protocol deviations (procedural).** The brief says never to use run_in_background or sleep
     loops of your own. The agent used `until/for ... sleep` polling loops to watch push and fetch output
     files that the tool had auto-backgrounded (tools 12-74 from 15:27Z to 18:17Z; 134-135; 142-144; 150-156),
     and polled `kaggle kernels status` in a loop (20:05:29Z). At 18:23:04Z (tool 86) it tried
     `(nohup push.sh step1.py &)`. On macOS, nohup failed ("can't detach from console"), and
     `/tmp/claude-501/r4/push.log` holds only that error, so this did not cause a second push. None of this
     affects the budget.
   - **Coordinator re-push of job 1 at about 19:52Z: no double charge.** The agent's kernel v1 was pushed at
     18:24Z and stayed QUEUED until at least 19:49Z. The coordinator deleted it and pushed an identical build.
     The ledger has exactly one set of 6 base passes (19:54:09-19:54:58), with notes matching `step1.py`, and
     job 2 started from 6 used. `sync_ledgers.py` only replaces a ledger with a longer one, so a re-push
     cannot double-charge. Not verifiable: whether the deleted v1 kernel ran any passes before it was deleted.
     That would mean uncharged passes. The coordinator says it was still queued. The first RUNNING status
     (poll at 19:51:21Z) and the ledger timestamps fit the re-pushed kernel. The Kaggle log on disk is job 4's
     only, because fetch overwrites it.
   - Two unused sibling edits (`prior_n175`, `prior_n250`) remain in EDITS. They were scored on train only
     and are disclosed.

## Could not verify
- That the deleted original job-1 kernel never executed (Kaggle-side state is gone).
- The actual tensor diff count (it would need the model loaded). The count follows from `apply()` code.
- Kaggle-side outputs of jobs 1-3 beyond what was copied into WORK and cache. The per-config train logp files
  and `s2_results.json` / `s3_results.json` agree with the transcript.
