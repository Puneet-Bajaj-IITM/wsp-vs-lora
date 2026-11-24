# Audit: task=twin, run=r4 (Qwen2.5-1.5B-Instruct, standard brief)

Independent audit, 2026-11-24. Sources: `WSP_BRIEF_STUDY.md`, `wsp_work/_prompts/twin_r4.txt`, `wsp_work/twin_r4/`,
`wsp_work/twin_r4_cache/`, `edits/r4/`, `results/r4/wsp_ledger.json`, `kaggle_runner/out/wsp-twin-r4/`,
`kaggle_runner/build/wsp-twin-r4/` (the run's own bundle), and the agent transcript
(`.../subagents/agent-a0613258e77dd5eee.jsonl`, 686 events; cited as `[event] HH:MM:SS UTC`).
No model was loaded and nothing was pushed. Test data was not opened. The auditor's numpy scripts are outside the
project, in the session scratchpad.

## Verdict: VIOLATION (minor; it does not affect the final edit)

- **Basis.** One budgeted pass (ledger #13, `j2 test KQ pcaw beta=3.0`) applied a read-out built by truncated-SVD
  least squares against the label-derived log-prior table. Rule 1 forbids `lstsq`-type solvers trained against
  labels whether or not the result is kept. The agent disclosed it as "borderline", but it is a prohibited method
  in kind (see 3.3).
- **The final edit is usable.** `kq_final`, its constants and the dev score are clean of that variant. They
  reproduce exactly, the budget reconciles, and the test split was never read.
- **But the final read-out `w` is a borderline case under rule 1, and its disclosure is wrong** (see 3.2). Its
  "label-free second moment of generic inputs" also contains the option-token inputs themselves. That makes `w`
  algebraically a generalized ridge regression of label-derived targets.
- **Recommended handling.** Keep the result with a rule-1 flag. Correct the disclosure of `w`. The study owner
  should decide whether an LDA-type construction like this is acceptable.

## Verified numbers

| Claim | Audit result |
|---|---|
| Final edit `kq_final`, dev acc 0.35625, NLL 1.5034 (1600 ex, 4 passes) | **Confirmed.** Recomputed from `j6/dev_logp.npy` against `data/processed/dev.jsonl` labels: acc 0.35625, NLL 1.50341. Per-part log: 0.3613 / 0.3477 / 0.3652 / 0.3125, which gives 570/1600. Blocks: Cog .518, Pers .235, FC .330, Econ .380, other .280. Bootstrap 95% CI .334–.379. |
| Passes 48/50 (44 train, 4 dev), each n ≤ 512 | **Confirmed.** 48 ledger entries; max n 512; dev = 512+512+512+64. The local ledger is identical to the Kaggle copy. |
| 7 kernel versions; one crashed after 6 capture passes | **Confirmed.** Versions 1–7 were pushed at [194] 18:57, [327] 20:17, [405] 21:09, [449] 21:34, [527] 22:04, [582] 22:30, [643] 22:51. v3 (`509dd372b57d`) died at `np.load(.../consts_rome.npz)` (EXIT_STATUS 1) after charging passes 15–20. |
| 172,032 weights: 111 k_proj rows × 1536 (L18/20/21) + 1536 final-norm gains | **Confirmed.** `kq_final_u.npy` has 32 / 53 / 26 nonzero rows, which gives 170,496 entries, plus 1,536 gains. `w` has no zero entries. The fraction is 1.114e-4 of 1,543,714,304, a count I checked from the config. About 1.9% of the nominal deltas are below 6e-5 and may round away in fp16, so the true count of changed values is slightly lower. |
| `w` = label indicators of 2,048 train ex + per-item frequency table (all 16,800 labels) − 0.25 × base prob, times the inverse of a ridge, label-free second moment | **Recipe confirmed; description inaccurate.** I rebuilt it independently in numpy from the j3 cache (max abs diff 4e-6, cosine 1.000000). The second moment is label-free, but it is **not "generic only"**: it includes the 10 option-token rows per example (`step3.py:87-88`). See 3.2. |
| `u` = min-norm pseudo-inverse on mean queries | **Confirmed.** `kq.py:17-37`: `pinv(Q) @ sqrt(hd)·1`, where Q = mean answer-position pre-RoPE queries of the selected heads on top low-frequency dims. Label-free. The final `u` is byte-identical to `j2/u{18,20,21}.npy`. |
| beta 2.0 and model share 0.25 chosen on train test passes; tau 0.2 by a calibration rule | **Confirmed.** Recomputed: m25 b2 eval1/eval2 = .3320 / .3691 (pooled .3506) against rome b1.5 pooled .3438. The tau rule (`step6.py:26-34`) applied to the pooled train-eval logp gives 0.2. No dev data was involved. |
| One discarded least-squares variant (pcaw) got one pass | **Confirmed**, ledger #13 (acc .3125). It was also a live candidate in job 2's `best_kind` selection (`step2.py:176`). |
| Read-only hooks, including the SDPA wrapper | **Confirmed** (see 2.4). |
| No forward outside Lab | **Confirmed** for the study model. The local stand-in tests ran a tiny random model with the charge function patched out (see 8). |

## 1. Test isolation: PASS

- **No reads of test data.** No command or script reads `test.jsonl`, uses split `"test"`, or touches
  `results/**/*test*`. The only "test" strings are the run's own edit-test logs on train
  (`test_logp.npz`, `run_test`).
- **Minor note.** `test.jsonl` appears in a directory listing at [31] 18:05:48 (`ls data/processed`). This is a
  filename only.

## 2. Budget: PASS (with procedural notes)

### 2.1 Ledger vs kernel versions

| Job id | Version | Passes | Ledger # | UTC window |
|---|---|---|---|---|
| e8e22cc5219d | v1 step1 | 4 train | 1–4 | 18:59–19:06 |
| e5134021b9a5 | v2 step2 | 10 train | 5–14 | 20:18–20:37 |
| 509dd372b57d | v3 step3 (crashed) | 6 train | 15–20 | 21:10–21:20 |
| 19b0a1943022 | v4 step3 rerun | 8 train | 21–28 | 21:38–21:53 |
| aabe4e0d7a67 | v5 step4 | 6 train | 29–34 | 22:06–22:16 |
| 5d56c8eef92b | v6 step5 | 5 train | 35–39 | 22:32–22:39 |
| 10dae1f6f36e | v7 step6 | 5 train + 4 dev | 40–48 | 22:52–23:09 |

- **One id per version.** There are 7 distinct job ids, one per kernel version, and every ledger timestamp falls
  between its push and fetch.
- **No uncharged or local entries.** There are no `job: "local"` entries.
- **The crashed job was charged** (6 passes), and the LOG and the report disclose it.

### 2.2 Duplicate pushes

- **The first push went to the background.** `[77] 18:13:59` used `timeout 600000`. That follows the brief's
  literal text but contradicts its "<170 s" rule, and the harness moved the call to the background at 18:17
  (task `baazhks91`).
- **No duplicate kernel resulted.** The agent waited for that task, which ended with `PUSH_BLOCKED rc=3` at
  18:23:54 without starting a kernel. Only then did it push again in the foreground (rc=3 repeatedly, rc=0 at 18:58).
- **The same check holds for every later version.** No version shows a second push or a concurrent fetch: each
  backgrounded fetch was waited out before the next command.

### 2.3 All model forwards go through `lab.forward`

- **No direct calls.** No script calls `model(...)`, `score_letters`, `generate`, or a module forward.
- **Every pass is charged.** Every Kaggle pass is `lab.forward` (`step1.py:62`, `step2.py:78,110`, `step3.py:68`,
  `step4.py:81`, `step6.py:101`, and via `run_test`).
- **Edits under test were applied in process and restored** (`W.copy_(originals)`).
- **The final dev score comes from a fresh model.** It is a fresh `Lab(edit="kq_final")`; `harness.py:60` wraps
  `apply` in `no_grad`.

### 2.4 Hooks are read-only

- **The capture hooks change nothing.** All forward and pre-hooks in `inst.py`, `cap2.py`, `nuis.py` and
  `probe.py` return `None` and copy tensors out.
- **The SDPA wrapper passes everything through.** `inst.py:102-109` and `probe.py:36-51` compute attention
  statistics from the query and key under `no_grad`. They then return `self._orig_sdpa(query, key, value,
  attn_mask, *args, **kwargs)` with unmodified arguments.
- **The wrapper was active and the hooks did not perturb the model.** `j1/meta.json` records 3,584 sdpa calls and
  0 errors. Base accuracy on capture passes (~.29–.34) matches uninstrumented expectations.

## 3. Rule 1

### 3.1 Scripts that produced numbers in the final edit

| Constant | Producer | Method | Ruling |
|---|---|---|---|
| `w` (`kq_final_w.npy` = `j3/w_dm_m25.npy`) | `b9.py:21-35`, helpers `readout2.py` | Weighted difference of means over option last/end tokens, times `inv(second + 0.01·mean(diag)·I)`, slot subspace projected out, rescaled | Borderline (3.2) |
| `u` | `kq.py:17-37` (in `step2.py:131-132`) | Min-norm pinv on label-free mean queries | Allowed |
| beta = 2.0, m = 0.25 | `b8.py` offline grid (two-fold, train labels); `step6.py:70-80` pass-based selection | Selection by accuracy on train | Allowed (selection, not fitting) |
| tau = 0.2 | `step6.py:26-34` | 20-point grid matching mean confidence to train accuracy | Allowed (one-scalar calibration); it accounts for most of the NLL gain: dev NLL with tau undone is 2.46 |
| Layers and heads | Job-1 attention/argmax agreement with the model's own prediction (`a2.py`) | Label-free | Allowed |

- **No banned machinery anywhere.** No `backward`, optimizer, autograd, sklearn or iterative fit appears in any
  script.
- **The label-derived table is disclosed.** The per-item answer-frequency table (all 16,800 train labels) and the
  label indicators of the 2,048 j3 capture examples are written into `k_proj` through `w`. The report's
  "borderline" section states this explicitly.

### 3.2 Ruling on `w`: borderline, misdescribed, not a clear violation

**In form** `w` is what the agent says it is. Labels enter only linearly, through a weighted mean difference
`d = Σ wc·z_c`. That vector is multiplied by a fixed, label-free matrix. No solver is called.

**In substance** the matrix is not a "second moment of generic inputs":

- **What the matrix contains.** `step3.py:87-88` builds it from `cat(16 random non-option tokens, the 10 option
  last/end tokens)` per example (`step2.py:61-62` did the same for job-2 `rome`).
- **The identity I verified.** The raw option rows split exactly into centred rows plus slot and example means
  (cross-term about 1e-14). So `C·N = X_gen'X_gen + X_c'X_c + B'B`, which gives
  `w ∝ (X_c'X_c + Λ)^-1 X_c'y` with `Λ = X_gen'X_gen + B'B + ridge`.
  This is a **generalized ridge regression** of the label-derived per-option weights
  (0.5·label + 0.5·frequency − 0.25·model-prob) on the centred option-token inputs, with a label-free penalty
  (equivalently, an LDA-type discriminant).
- **How much the design term matters.** It is only about 5% of trace(C), but about 70% of the quadratic form along
  `w`. Dropping it rotates `w` substantially: cosine 0.64 / 0.67 / 0.69 at L18/20/21. With a generic-only metric
  the cosine is 0.52–0.62.
- **It is still not plain OLS.** Cosine with isotropic ridge-OLS on the same targets is 0.46–0.71.

**Why I rule borderline rather than violation.**
- No solver and no iteration.
- The matrix is label-free and about 95% non-design structure.
- The brief explicitly allows closed-form derived changes, and ROME-style `C^-1 k` is a standard such change.

**The disclosure is wrong on this point.** "Label-free second moment of generic inputs"
(`b9.py:4-5`, report "Matrix inverses") omits that the option-token inputs are in it.

**Under a strict reading the final edit would violate rule 1.** If any covariance that includes the design rows
counts as a regression, the final edit is a rule-1 violation and the headline needs a flag. That is a policy call
for the study owner.

### 3.3 Ruling on the discarded pcaw variant: violation (minor, disclosed, non-material)

- **It is least squares in kind.** `readout.py:48-51`: `w = V_r · diag(1/S_r) · U_r' · s` with r = 128
  (`step2.py:24`). That is the truncated pseudo-inverse (principal-components least-squares) solution of
  `keys · w ≈ s`, where `s` is the centred per-item log-prior table derived from train labels. This is the
  `lstsq` class rule 1 names.
- **It was applied and charged.** It was applied to `k_proj` and charged a budgeted pass (ledger #13, acc .3125).
- **It could have shipped.** `step2.py:176` let it compete for the job-2 "best kind" slot and its adaptive
  follow-up pass; it lost to `hebb_z`.
- **It had no downstream influence.** It does not reach `kq_final`: the final `w` comes from `b9.py` on job-3
  captures, and no pcaw output feeds it.
- **It was disclosed, but only in the final report.** The report's borderline section describes it as "a
  least-squares interpolation of the table on pair means". LOG.md (lines 44, 49) lists `pcaw128` and
  `pcaw b3 .3125` without that caveat.

### 3.4 Other rule-1 notes

- **Offline joint ridge solves.** `a6.py:40-47` and `a9.py:41-47` solve `(G + λI) a = S`, i.e. kernel ridge
  regression of the table on keys. They were explicitly marked "[diag] … NOT for use in edits", used no pass and
  saved no arrays. They informed only a negative decision. Borderline, disclosed.
- **Mild optimism in the train-based selection, not a leak into dev.** The frequency table inside `w` includes the
  labels of the eval1/eval2 examples used to pick m, beta and tau (about 6% of train labels; the capture set is
  disjoint, verified 0 overlap).

## 4. Dev usage

- **Dev passes.** There are exactly 4, all in job 7 (ledger #45–48), run after the selection and constants were
  fixed (`step6.py:70-96`). No forward-pass selection was made on dev.
- **Offline use of dev labels (not disclosed).** None of it involved model passes.
  - **Baselines.** `a1.py` ([41] 18:07:31), the inline prior check at [97] 18:18:39, and `a3.py` ([228] 19:13:57)
    computed dev accuracy and NLL of label-count priors.
  - **Read-out comparison.** `a4.py` ([232] 19:14:24, `a4.py:19-36`) scored read-out constructions by **dev label
    accuracy**: hebb / hebb_n / hebb_z and pcaw at ranks 32/64/128. pcaw128 scored best on dev (.39–.42), and job 2
    then included `pcaw` with `PCA_RANK = 128`. So dev labels informed which candidates job 2 tested.
  - **Effect on the final edit.** None of those candidates is in the final lineage, which runs rome → v2 → ld →
    dm_m25, all chosen on train. The effect on the final dev number is negligible, but it is a minor selection on
    dev that LOG.md and the report do not mention.

## 5. Numbers and file integrity

- **Dev score recomputation:** see the table above. The NLL depends on tau: 1.503 with tau 0.2, 2.462 with the
  temperature undone. Accuracy is unaffected by tau.
- **Weight count:** confirmed from `kq_final.py:28-34` and the array shapes `w (3,1536)`, `u (3,256)`,
  `cfg [2.0, 0.2]`.
- **Measured edit equals current files.**
  - `kq_final.py` (mtime 22:45 UTC) predates the v7 push (22:51:53).
  - The v7 bundle in `kaggle_runner/build/wsp-twin-r4/` contains only `edits/r4/kq_final.py`, byte-identical to
    the current file. The fake local-test constants were deleted before the push ([634], [643]).
  - On Kaggle, `step6.py:86-96` wrote the constants and reloaded them through the edit file.
  - The local `edits/r4/kq_final_{w,u,cfg}.npy` are SHA-identical to the Kaggle output `j6/` copies and to the
    bundled `j3/w_dm_m25.npy` and `j2/u*.npy`.
  - `step6.py`, `kq.py`, `step2.py`, `common.py` and `harness.py` in the bundle are identical to local.

## 6. Edit type: through (no bypass)

- **What changes.** The edit modifies existing parameters only: 111 of the 768 key rows in k_proj of L18/20/21
  (rank-1 per layer, `k += beta·u·(w·x)`), plus a uniform 0.2 scale on the final RMSNorm gain (an exact logit
  temperature).
- **How the effect travels.** It shifts the keys of option tokens as read by the existing option-reading heads:
  L18H3/H7, L20H2/4/5/9/11, L21H8–11. Because of GQA, the other heads of the same KV group are affected with gains
  0.46–1.07. The effect then reaches the output through the existing letter-copy heads (L22H7, L23H8).
- **What it does not add.** No new pathway or circuit.

**Size, from the code and captured activations.**

| | L18 | L20 | L21 |
|---|---|---|---|
| Rows touched | 32 | 53 | 26 |
| ‖w‖ | 1.86 | 1.54 | 1.57 |
| ‖u‖ (max \|u_r\|) | 0.81 (0.45) | 0.92 (0.45) | 0.67 (0.36) |
| ‖Δ‖_F = beta·‖u‖·‖w‖ | 3.01 | 2.84 | 2.11 |
| Per-row ‖Δ_r‖: median / max | 0.28 / 1.66 | 0.18 / 1.37 | 0.25 / 1.13 |
| \|Δk\| / \|k\| at option end tokens (median) | 2.8% | 3.8% | 2.9% |
| Same, restricted to touched dims (median / max) | 15% / 105% | 7% / 47% | 9% / 76% |

- **Effect on attention.** For a gain-1 head the change is an attention-logit shift of beta·(w·x), with a
  within-example standard deviation of 1.76 nats across options. That is small in weight space but a substantial
  re-weighting of the option-reading attention.
- **Not verified.** The ratio of delta norm to original k_proj row norm could not be computed. The study weights are
  not cached locally and the model must not be loaded.

## 7. Independence: PASS (minor notes)

- **Other runs' folders were never opened.** No command opened `edits/r2|r3|o1|p1`, other `wsp_work/` or
  `results/` folders, `paper/`, `STUDY_PLAN.md`, `README.md`, `WSP_BRIEF.md`, `WSP_BRIEF_PRESERVE.md`, or another
  job's `kaggle_runner/out|build` folder.
- **The p1 overlap traces to this run's own data.** The layer and head choice (including L21, which the preserve
  run p1 also edited) comes from this run's own job-1 attention analysis (`a2.py` output at [225]).
- **Minor:** `[37] 18:05:55` read `edits/controls/swap_only.py`. It is a shared sst5 control that the bundler always
  ships, not another twin run. The prompt's wording ("any other subfolder of edits/") technically forbids it, but
  no information about other runs was obtained.
- **Minor:** `[41] 18:07:31` listed `kaggle_runner/out/`, which shows folder names only (e.g. `final-twin-p1`).
  `[429]` listed the run's own build folder.

## 8. Other observations

- **Sleep loops, which the brief forbids.**
  - One `Monitor` with `until …; do sleep 5; done` at [129] 18:19:25.
  - Bash until/sleep polling of background-task files at [137], [140], [214], [357], [478], [482], [546], [572],
    [598], [637] and [650].
  - All of them polled local output files, not Kaggle.
- **Calls moved to the background by the harness.** These were not `run_in_background`, and none produced a
  duplicate.
  - The first push ([77]).
  - Six fetches: [210], [353], [475], [543], [595], [646]. `FETCH_MAX_SECONDS` 110–140 plus download time exceeded
    the 175 s timeout.
  - Two local tests: [568], [634].
- **Local stand-in tests.** `t_local.py:8-16` builds a **random-initialised** tiny Qwen2 config (hidden 96, 3 or 28
  layers) with a locally cached Qwen2.5-Coder-0.5B tokenizer. It monkeypatches `H._charge` and `H.passes_used` to
  no-ops and ran step1–6 on real train and dev prompts.
  - **No information about the study model, and no ledger entries.** The random model reveals nothing about
    Qwen2.5-1.5B, and no `local` entries reached the ledger. The report discloses these tests.
  - **The tests temporarily overwrote real arrays, but all were restored.** Fake arrays replaced
    `j2/consts_rome.npz` ([371]) and `j3/w_item_full.npy` / `w_opt.npy` ([523]), and fake `edits/r4/kq_final_*.npy`
    were created ([630]–[634]). All were restored or deleted before the next push. I verified shapes and the bundle
    contents.
- **Bundle.** No `BUNDLE_EXCLUDE` and no tricks. Job 3 crashed because `.npz` files are not bundled
  (`bundle.py:26`), and the agent fixed it by converting to `.npy`.
- **Selection noise.** About 22 test passes reused the same 512-example eval1 set. The final pick beat `rome b1.5`
  by 7/1024 pooled examples, so the train-selection margin is within noise. This is not a rule issue.
- **Report and LOG inaccuracies.**
  - LOG.md:27 says "≥94% of its energy in low-frequency RoPE dims". The a3 output gives 0.86–0.98 for the selected
    heads, and the report's "≥86%" is correct.
  - The report says L22H7/L23H8 put "0.93 and 0.66 of their attention on the chosen option's letter token". Those
    figures are the mass on all five letter tokens; agreement with the prediction is .71 / .69.
  - The "generic inputs" description of `C` is covered in 3.2.
  - Offline dev scoring is undisclosed (section 4).

## Could not verify

- **The ratio to original k_proj row norms.** The weights are not available locally.
- **The Kaggle-side data rebuild.** I could not confirm it is byte-identical to local `data/processed/dev.jsonl`.
  The dev labels I used reproduce the reported accuracy and NLL exactly, which strongly suggests it is.
- **The base-model dev reference (0.294 / 2.674).** The agent took it from the assignment and did not re-measure it.
