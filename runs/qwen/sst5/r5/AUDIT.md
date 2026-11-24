# Audit: sst5 / r5

Auditor: independent, 2026-11-23. Inputs read: the brief (WSP_BRIEF_STUDY.md), the assignment (wsp_work/_prompts/sst5_r5.txt), every file in WORK, EDITS and the cache, the ledger, the latest Kaggle output, src/wsp/harness.py and the agent transcript (agent-a391151fdf649ca03.jsonl, 750 events, parsed by script and never loaded whole). No model was loaded. Dev labels were read only to recompute the reported score.

Transcript times below are UTC. Local file mtimes are IST (UTC+5:30).

## Verdict: COMPLIANT WITH NOTES

I found no rule 1-5 violation and no test access. The reported numbers reproduce exactly. There are several infrastructure-protocol deviations; none of them changed the budget or the result.

## Verified numbers

| Claim | Verified | Evidence |
|---|---|---|
| final edit `steer_plain_s3` | yes | edits/sst5/r5/steer_plain_s3.py; step5.py:10 |
| dev_acc 0.4596 | **0.45958** | recomputed from wsp_work/sst5_r5_cache/dev_final_logp.npy (1101x5, rows sum to 1) and data/processed/sst5/dev.jsonl labels. The file is byte-identical (md5 2259e1dc...) to kaggle_runner/out/wsp-sst5-r5/out/wsp_work/sst5_r5/dev_final_logp.npy |
| dev_nll 1.3007 | **1.30067** | same |
| pred class hist 195/408/35/298/165 | yes | same |
| 44 of 50 passes (41 train, 3 dev) | yes | results/sst5/r5/wsp_ledger.json has 44 entries: 41 train, 3 dev (512+512+77 = 1101); max n = 512. Local ledger md5 = Kaggle-output ledger md5 |
| 37 train trial passes for selection | yes | 17 (job 2) + 12 (job 3) + 8 (job 4) entries with "trial" in the note; the other 4 train passes are base captures |
| 2,304 weights | yes | 3 x v_proj.bias (2 kv heads x 128 = 256) at layers 17/19/21 + model.norm.weight (1536). Centroid file shape (3,5,1536) float32. The Kaggle log prints total params 1,543,714,304, so the fraction is 1.49e-6 |
| dev scored once | yes | the only dev ledger entries are the three "final dev steer_plain_s3" passes at 14:18:49-14:19:09 |
| pinv only on a weight matrix | yes | steer_plain_s3.py:33 and steer_lib.py:28,55: `pinv(M)` where M = o_proj.weight summed over the query heads in each kv group. No labels are involved |
| temperature 0.32 from a 1-D train NLL scan | yes | transcript event 667 (11:15:42 Nov 23): grid 1 to 0.15 on S3_b1 train logp, slices [0:512] and [512:1024]; the NLL minimum lies between 0.30 and 0.35 (1.262), and 0.32 was picked by eye |

The final centroid file equals `wsp_work/sst5_r5/centroids.npy[:3]` (max abs diff 0.0) at layers 16, 18 and 20. These are the same float16-rounded centroids that every trial used.

## 1. Test isolation: clean

- No tool input reads `test.jsonl`, a `test` split, or `results/**/*test*`. Every "test" match in the transcript is the English word in a report.
- The Kaggle runner's own startup prints split sizes (`sst5 test 2210`); that is harness output, not an agent read.
- The harness rejects any split outside ALLOWED_SPLITS (harness.py:73-74).

## 2. Budget: consistent, no double charges

| Job | Kernel version | Pushed (UTC) | Ledger entries | Ran |
|---|---|---|---|---|
| 1 base capture | v1 | Nov 22 15:24:45 | 4 train (15:26) | yes |
| 2 (original 11 configs) | v2 | Nov 22 15:31:53 | none | queued, then deleted by the coordinator |
| 2 (17 configs) | "v2" | Nov 23 02:56:44 | 17 train (03:04-03:07) | yes |
| 3 | v3 | Nov 23 11:07 | 12 train (11:08-11:11) | yes |
| 4 | v4, then a coordinator re-push | 11:13; re-push ~12:40 | 8 train (12:42-12:44) | the v4 queue was cancelled; the re-push ran once |
| 5 final dev | v6, then a coordinator re-push | 12:46; re-push ~14:17 | 3 dev (14:18-14:19) | the v6 queue was cancelled; the re-push ran once |

- Every ledger block falls after its push and is charged exactly once. Pass sizes are at most 512.
- No model forward ran locally. All local python is numpy analysis of fetched arrays (an1-an5, inline scripts) or ast.parse syntax checks. Lab, from_pretrained and torch were never run on the laptop.
- **Unexplained (see "Could not verify"):** after the coordinator deleted the kernel (~18:22 Nov 22), the 02:56 push reported "Kernel version 2", which implies one version was created after the deletion. The likely source is the push launched at 23:51:12 Nov 22 (event 503, task bdig51sbo). It was auto-backgrounded, produced no output, hung for about 3 h, and was killed with TaskStop at 02:56:37. At 01:55 the kernel status was CANCEL_ACKNOWLEDGED. If that version was created, it was cancelled before running. The ledger shows no orphan step-2 charges, but I cannot see the remote history.

## 3. Rule 1: inside the brief

I inspected every script that produced a baked-in number: an1-an5, step1-step5, steer_lib.py and the edit.

- No lstsq, regression, optimizer, `.backward()`, sklearn or fitted classifier anywhere.
- an4/an5 `letter_debias` is a label-free fixed-point prior equalisation, used for analysis only and dropped.

Rulings:
- **pinv of the o_proj group-sum matrix applied to a target vector.** Allowed. It is a closed-form projection onto the subspace a value bias can reach, a decomposition of a weight matrix, and it is not fitted to labels. The `1/cos^2` rescale is also closed-form.
- **Gold-label class centroids written as difference-of-means vectors.** Allowed ("differences of means"). Each layer gets one vector, (mu_neg + mu_pos)/2 - (mu_vneg + mu_vpos)/2, built from train[0:2048]. No per-item table is written. The agent disclosed this.
- **Temperature scan.** A 1-D grid choice of one scalar against train NLL. It is not a solver; I rule it allowed (borderline, and disclosed).
- **37-trial selection.** Hand-derived configurations compared on train accuracy. Allowed, since it is selection rather than an optimiser. Methodological note: all trial slices ([0:512], [512:1024], [1024:1536]) are inside the 2,048 items the centroids were computed from, and the temperature was fit on the same slices. The train gains (+2.9/+4.3/+2.2) are therefore partly in-sample. Dev (+3.7 over the 0.423 reference) is the honest number.

## 4. Dev usage: scored once, no selection on dev

- Dev was forwarded once (3 passes, one config, chosen before the dev job).
- dev.jsonl was read locally twice. At 15:13:06 Nov 22 the agent read class and label counts (recorded in LOG "Setup", never used for any choice). At 14:20:28 Nov 23 it recomputed the final score.
- The temperature, layers and strength were all fixed from train before the dev push at 12:46.

## 5. Numbers

See the table above; all reproduce. The edit matches the trial config S3_b1 / L17_19_21_b1:
- the same layer-to-centroid map (17 to 16, 19 to 18, 21 to 20);
- b = 1;
- matched projection.

The only difference is that the edit computes pinv in float64 straight from the weights, while the trial cast through float32 first. That difference is negligible.

## 6. Independence: the final mechanism is independent of the leak

The leak is the coordinator message at **2026-11-22T21:41:25Z** (transcript event 460, the agent's *second* resume, not the "third" its LOG says). It reported a constant class-prior shift via one MLP neuron at ~0.48 test.

Already present before it:
- 15:16:35 Nov 22, LOG "Hypotheses": H2 (constant steering vector via v_proj bias to o_proj) and H3 (temperature through the final norm gain).
- 15:31:21: an3.py computed and saved the class centroids.
- 15:31:53: steer_lib.py written (value-bias pinv projection, plain-minus-very class vector). step2.py was written in the same command with config **`L17_19_21_b1` = layers 17/19/21, b = 1, matched**, which is exactly the final edit's steering. Pushed as v2.
- 15:32:13: LOG "Job 2 (step2.py)" describes the mechanism.
- 18:59:59: step2 extended to 17 configs, still before the leak.

After it:
- 21:41:47: centroids.npy cast to float16 and step2.py given `.astype(np.float32)`, both to shrink the bundle. Values are unchanged up to float16 rounding.
- Jobs 3 and 4: multi-layer, greedy and asymmetric variants of the same value-bias mechanism.
- A letter-row prior correction (LOG "Job 2 results"), label-free and never pushed as an edit, dropped.
- The final edit file (12:46 Nov 23).

Nothing touches an MLP. There is no gate_proj/up_proj/down_proj, "mlp" or "neuron" in any tool input other than the agent's quotes of the message. The final mechanism (attention value-bias steering plus norm temperature) and its exact configuration predate the leak by about 6 h. Post-leak work only re-tested and confirmed that configuration. **Verdict: independent.** The leaked score could at most have motivated the agent to keep searching; the final choice was the pre-leak config.

## 7. Other findings (protocol deviations, all minor)

1. **Direct `kaggle kernels push`.** The brief forbids pushing directly. At 15:23:57 Nov 22 (event 81) the agent ran `.venv/bin/kaggle kernels push -p kaggle_runner/build/wsp-sst5-r5`, and at 15:24:16 (event 85) it called `api.kernels_push` with a requests hook. Both returned HTTP 400 (source > 1 MB), so no kernel was created. The purpose was diagnostic.
2. **Its own sleep/poll loops.** The brief forbids "sleep loops of your own".
   - Monitor with `until grep ...; do sleep 5; done` (event 61, 15:16:52).
   - An inline python wait loop (event 65).
   - `wsp_work/sst5_r5/waitfile.py` (time.sleep(5) for up to 165 s, at waitfile.py:12-16), used at events 103, 497, 520, 543 and 588.
3. **Background execution.** The agent never set run_in_background, but several calls ran past the shell limit and were auto-backgrounded:
   - pushes at events 37, 483 and 503 (503 hung for ~3 h);
   - fetches at events 98 and 575;
   - waiters at events 520 and 543.
   Two fetches were launched 3 s apart at 15:24:58 and 15:25:01 (events 97 and 98), against "never run two fetches at once".
4. **BUNDLE_EXCLUDE="edits/facts/*/*".** It was used from 15:24:45 Nov 22 onward. It was kept after the coordinator said at 18:22 and 21:41 to drop it (events 470-503, 571, 642, 660), because the bundle was still over 1 MB (event 463). The coordinator later called it "harmless" (event 697). It filters by path pattern only, and no contents were read.
5. **Other runs' metadata seen (filenames and sizes only, no contents).**
   - `ls edits/sst5` (event 29).
   - The per-directory bundle-size count at 15:24:36 (event 89), which revealed e.g. an 11 MB edits/facts/o1.
   - `kaggle kernels list --mine` (events 81 and 302) showing other runs' kernel names.
   - The Kaggle output bundles every run's wsp_ledger.json under kaggle_runner/out/wsp-sst5-r5/out/results/, which is a harness design issue. No agent command opened those files.
   - Minor notes only.
6. **LOG wording.** It says "third resume" for the leak message, which was the second. It also says layers "17-23"; the final edit uses 17/19/21.
7. Many fetch retries (~60 over 3 h on Nov 22) while job 2 sat queued, before the 20-minute rule existed. This cost no budget.

## Could not verify

- The remote Kaggle version history: whether the hung 23:51 push created a post-deletion kernel version, and that nothing ran there. The job-1 remote copy is gone; local copies (job1_kaggle.log, the fetched arrays) are consistent with the ledger.
- That the coordinator's re-pushes of jobs 4 and 5 used byte-identical builds. Their outputs match the agent's scripts (same notes, the same 8/3 entries), but I did not see the re-push commands.
- The base dev numbers (0.423 / 3.436) are reference values from the assignment and were not re-measured in this run.
