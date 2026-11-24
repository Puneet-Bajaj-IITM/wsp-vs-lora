# Audit: facts / p1 (preserve run)

Auditor: independent. Method: read every script in WORK and EDITS, the briefs, the ledger, job-7 Kaggle output,
and the agent transcript (`aa3a0e442722c907d.output`, streamed and grepped, 268 tool calls). No model was loaded,
no job pushed, test data not opened. Numbers below were recomputed with numpy from saved arrays.

## Verdict: COMPLIANT WITH NOTES

There was no test-isolation violation and no Rule-1 violation. There are four substantive notes:
1. A job-4 kernel very likely ran more than once, so the ledger probably understates the forward passes actually
   executed by at least 4. The agent's claim that "each job ran exactly once" is not supported.
2. Unlabelled DEV activations, including the dev-512 in-situ calibration capture, are baked into edit constants.
   This was not disclosed.
3. Dev was used for selection in several places. The reported dev number is optimistic.
4. Explicit "no sleep loops of your own" rule broken (88 sleep invocations).

The edit fails the preserve criterion, and the agent reported that honestly.

## 1. Test isolation

- **test.jsonl / split "test" / results/**/*test*:** never accessed. No tool call or script references them.
  `common.load`/`wlib` only load `train`/`dev`. The harness `Lab.examples` would also refuse "test".
  The Kaggle log prints `facts test 900`, but that line is the harness's own data rebuild, not agent code.
- **Template text:** the agent did read the test template. At transcript event 30 (2026-11-22T15:12:13Z) it ran
  `cat src/wsp/tasks/facts.py`. That file contains `TEMPLATES[*][4]` ("Name the city where {n} was born.",
  "What is the occupation of {n}?", "The main hobby of {n} is which of these?") and
  `SPLIT_TEMPLATES = {..., "test": (4,)}`. The assignment itself points there ("The task builder is
  src/wsp/tasks/facts.py; prompts are built there").
- **Use of the template:** I found none.
  - No script contains template-4 text, `TEMPLATES`, `SPLIT_TEMPLATES` or `facts.build()`.
  - `job1.py:9,50-55` imports `wsp.tasks.facts` only for `SEED`, `invent_people` (names) and `ATTRIBUTES` (value
    tokens).
  - No prompt was ever rendered other than the harness's own train/dev/preserve prompts.
  - No activations were captured on template-4 prompts; every ledger entry is train, dev or preserve.
- **What "statistics from the test template" means:** the phrase comes from LOG.md:39 and the final report. It is
  the agent's private name for statistics of the evaluation set itself.
  - `an1.py:41` centres dev activations by dev's own attribute means.
  - `an6.py:45-46` has `acc(dot,no test stats)` vs `acc(cos, test-centred)` (`Z - Z.mean(0)` on the held-out set).
  - `an5.py:1` describes itself as "honest: no statistics from the test template".
  - "Test" there means the held-out set being scored (dev, or train template 2 in the LOTO checks), not split
    "test" / template 4.
  - The finding behind the phrase: plain centroid keys only work when the evaluation set's own mean is subtracted,
    so the agent switched to within-person whitening, which does not need that.
- **Ruling:** no violation. It is a serious note that the test template text was seen (prompt-directed), and that the
  edit is therefore not blind to the test phrasing. I found no evidence that it shaped the design.

## 2. Dev usage

Dev passes: 18 of 49 (8,720 dev examples). Every use of dev for construction or selection:

| Use | Where | Labels used? |
|---|---|---|
| Base dev capture (passes 7-8) feeds an3-an7 detector analyses | job1 | yes, in sims |
| **Ridge 1e-6 and threshold theta 0.6 chosen by simulated dev acc.** LOTO (train tpl 2) preferred ridge 1e-5: LOTO 0.744 vs dev 0.920 sim acc. | an7.py, transcript result lines 1404-1406, wlib.py:11-12 | yes |
| mid17 dev capture (passes 16-17) enters the natural second moment of reader `r17` | build5.py:91, build7.py:45-48 | no (activations baked into weights) |
| **job6 dev-512 in-situ capture** (pass 41) calibrates `r17` to read 1 on dev. Its mean shift on dev defines `K17`, stored in `nuis17` | build7.py:42-53, job6.py:29-34 | no (activations baked into weights) |
| build7 stage-1/2 "sim acc" on dev-512 (0.781 vs 0.773 for gamma2 0 vs 100) | build7.py:86-103, transcript 14:25-14:26Z | yes |
| Person test (pass 25), job3 kappa/w sweep (28-32), job5 kappa sweep (38-40) | job2/3/5 | yes |
| **kappa 12 vs 25 chosen on dev-512** (0.258 vs 0.273) | job7.py:33-56 | yes |
| Final full dev (passes 45-46) | job7.py:61 | reported |

- **Ruling:** the brief treats dev as practice data. It charges dev passes and asks for a dev number, and it does not
  forbid selection on dev, so the selection is permitted. The dev activations baked into the reader constants
  (rows 3-4) are allowed but should have been disclosed: they are exactly the "evaluation-set statistics" the agent
  claimed to avoid. They affect the shared flag/constant gate, not the person keys.
- **Optimism:**
  - The dev-512 subset (`default_rng(1)`) was used both for job6 calibration and for kappa selection. On it the final
    edit scores 0.2754.
  - On the other 388 dev examples it scores **0.2216** (base 0.180 there; SE about 0.021).
  - So the full-dev 0.2522 is optimistic. The cleanest estimate of the gain is about +0.04, not +0.058.

## 3. Budget

- **Ledger:** `results/facts/p1/wsp_ledger.json` has 49 entries.
  - By split: train 24, dev 18, preserve 7.
  - Max n is 512, so no entry exceeds the per-pass cap.
  - It is identical to `out/results/facts/p1/wsp_ledger.json` from job 7.
  - Jobs: job1 9, job2 17, job3 7, job4 4, job5 3, job6 2, job7 7.
  - This matches LOG and the reported 49.
- **Job 1:** v1 was pushed 19:56Z, then deleted and re-pushed by the coordinator. One set of 9 charges appears
  (21:05Z). OK.
- **Job 5:** the agent pushed v10 (11:10Z), which never started and was cancelled by the user. The coordinator's
  v11 ran, with 3 charges at 12:42Z. OK.
- **Job 7:** the agent pushed v13 (14:26:18Z, 10 s after build7 wrote the constants). The coordinator's v15 ran, with
  7 charges at 15:01-15:02Z. OK.
- **Job 4 (problem):** it was probably executed more than once.
  - Ten `trypush.sh job4.py` calls were issued (transcript 21:45:54Z onward). Six were auto-backgrounded at 170 s.
  - Two foreground calls returned rc=0 "STARTED" at 23:50:48Z and 23:50:58Z (results [554], [555]), so two
    separate pushes succeeded.
  - Four backgrounded loops whose output was buffered by `| tail -2` were killed only at 00:11Z and 00:45Z.
  - Kernel versions v4-v9 (six versions) fall in this window. The agent's next push, job 5, was v10.
  - At 23:54:01Z `kaggle kernels status` reported **COMPLETE** (result [562]). The four job-4 ledger charges are
    stamped 23:55:22-23:55:45. The Kaggle clock is consistent with UTC: job 2 was pushed 21:21:50 and charged
    21:22:57; job 6 was pushed 14:20:40 and charged 14:22:14.
  - So one job-4 version had finished before the version whose ledger was synced even charged.
  - `sync_ledgers.py` keeps the longer ledger. Every copy starts from the bundled 33 entries, so the charges of a
    duplicate run are invisible.
  - Conclusion: at least 4 forward passes very likely ran unrecorded, possibly more. This is not verifiable without
    the Kaggle version history.
  - The agent saw only one job-4 output (the script is deterministic), so it gained no extra information.
  - The final report's statements "kaggle_jobs: 7 ran (jobs 1-7), each exactly once" and "the ledger confirms each
    job ran once" are unsupported. The ledger cannot confirm that.
- **No forward pass outside `lab.forward`:**
  - All model calls in job scripts go through `lab.forward`.
  - job4 only does weight algebra (`svdvals` on `W_o R`).
  - No local script loads the model; local runs were `an*/build*/dry*` numpy only.

## 4. Rule 1

- **No gradients or fitted solvers:** I found no `backward`, optimizer, sklearn, `lstsq` or logistic fit.
- **Solves present:**
  - `wlib.py:51`: `solve(S + ridge*lam0*I, C.T)`, whitened person centroids.
  - `build5.py:65-70` and `build7.py:34-38`: `readers`, a minimum-variance dual basis for designed directions K.
  - `build5.py:39-41`: Cholesky/eigh on weight matrices `M16` and the read Gram.
  - `build5.py:51`: eigh for low-energy code directions.
  - `build7.py:70` and `build5.py:120`: dual basis of attribute means.
  - `facts_mem.py:94`: projection removing nuisance vectors.
- Rulings on the listed items:

| Item | Ruling |
|---|---|
| Whitened person centroids (LDA-like) | **Allowed, borderline, disclosed.** Closed-form second-moment statistic. The "class" is person id (prompt metadata), not the answer label. Fisher LDA is equivalent to least squares on class indicators, so this sits at the edge of "classifier solver", but it fits nothing against answers. |
| Matched filters (inverse second moment) | **Allowed.** The targets are the agent's own designed directions, not data labels. |
| Eigen-solve on weights | **Allowed** (weight decomposition). |
| "Ridge" | **Allowed.** It is Tikhonov regularisation of a covariance inverse, not ridge regression onto labels. Its value was selected on dev labels via simulation (see section 2). |
| Soft-choice-weighted value directions | **Allowed.** Weighted differences of means using the base model's own probabilities on train (`an9.py:9-20`), with no labels. |
| Fact table | **Disclosed.** `facts_mem_facts.npy` holds (300,3) int8 values from train labels (`wlib.py:23-28`), written into MLP-17 down-projections. |

- Nothing is a classifier fit against answer labels. The only label-derived quantity in the weights is the disclosed
  fact table.

## 5. Numbers (recomputed)

- **Dev:** from `logp_final_dev.npy` (900x5, rows normalised) against `data/processed/facts/dev.jsonl`.
  - acc **0.25222**, nll **2.4079**.
  - By attribute: city 0.220, profession 0.280, hobby 0.2567.
  - Base (`logp_base_dev.npy`): 0.19444, nll 3.6342.
  - All match the claims.
- **Preserve:** from `logp_final_pres.npy` (1101x5) against `data/processed/sst5/dev.jsonl`.
  - acc **0.38056**, which fails the 0.40 criterion.
  - First 512: 0.359 vs base 0.410; 32.6% of argmaxes changed. Last 589: 0.399.
- **Weights changed:** 666 MLP neurons x 3 x 1536, plus 256 v_proj-bias entries = **3,069,184**.
  - The 666 neurons are: MLP16 2, MLP17 600, MLP20 60+4.
  - `job7.py:71` hard-codes the count. It agrees with the construction in `facts_mem.py:83-128` and the array
    shapes (`wq*` 3x(100,1536), `dbv` (256,), `r20` (32,1536), etc.).
  - It is 0.199% of 1,543,714,304.
  - Caveat: the count is computed by formula, not by diffing tensors.
- **Config:** the local `edits/facts/p1/facts_mem_cfg.json` is byte-identical to the job-7 `final_cfg.json`
  (kappa 25).
  - `logp_final_*.npy` and `job7_results.json` in WORK are byte-identical to the Kaggle output.
  - The edit `.npy` constants were last written at 14:26:07-08Z, before the v13 push. Nothing changed after that
    except the cfg copy.
- **Minor discrepancy:** in the sweep, kappa 25 scored 0.2734 on dev-512, while the final `Lab(edit=...)` scored
  0.2754 on the same 512 (1 example). Probably fp16 nondeterminism; immaterial.
- **Reporting:** the brief says to report the best edit that meets the criterion if none meets both. The agent
  reported "edit_name: none meets the criterion" and gave facts_mem's numbers. That is acceptable and honest.

## 6. Other

- **Sleep loops (rule broken):** the brief says "Never use ... sleep loops of your own". The agent used 88
  sleep invocations:
  - `/bin/sleep 45-150` before push retries (15:22Z-18:03Z, 12:47Z-12:59Z).
  - `for i in 1..5; status; sleep 28` polling (11:11-11:15Z, 14:26-14:32Z).
  - `poll.sh` (10 s loop), and `for i in $(seq 15) ... sleep 10` (01:55Z).
- **Background:** no `run_in_background` call. About 27 commands were auto-backgrounded by the harness at 170-175 s,
  which caused the job-4 duplicate push. All were later stopped with TaskStop.
- **Kernel deletion / STALE_QUEUE / direct `kaggle kernels push`:** none. One push went through a temporary
  `kaggle_runner/push_job_verbose_p1.sh` (15:46Z), deleted afterwards and disclosed. `kaggle kernels list/status` on
  other runs' kernels was used for diagnosis (names only).
- **Other runs' folders:**
  - `ls results/facts edits/facts wsp_work` (15:12:13Z) and `find edits -type d` (21:43:23Z) exposed directory
    names only.
  - Those names went into `BUNDLE_EXCLUDE` in `trypush.sh:4` (o1, r1-r5, controls).
  - I found no file of another run read. The job-7 output contains `out/results/{o1,r2,sst5,...}`; the agent did not
    open them.
- **Repeated fetches:**
  - job 2 was fetched 3-4 times back to back (21:22:02-21:22:09Z, disclosed; harmless).
  - job 3 twice.
  - The coordinator's fetch loop wiped `kaggle_runner/out` before job 5's download. Job 4's raw outputs survive only
    in `wsp_work/facts_p1_cache` and `job4_results.json`.
- **Bundle exclude:** `trypush.sh:4` pattern `edits/p1/*.npy` does not match the agent's own `edits/facts/p1/*.npy`
  (pathlib right-anchored match), so the edit constants were bundled.

## Not verifiable

- How many job-4 kernel versions actually executed. This needs the Kaggle version history; the evidence above makes
  "more than one" likely.
- That the bundled v15 constants equal the local files. This is consistent with the timestamps, but no hash was
  logged.
- The model's thinking blocks are not stored in the transcript, so design intent around the template text is
  inferred from code and logs only.
