# Independent audit: twin / p1 (preserve run)

Auditor: an independent agent. I read the files only. I loaded no model and pushed no job.
Transcript: `agent-a40f65ad6262495de.jsonl` (987 events). Below, "ev N" means JSONL line N; all times are UTC.

## Verdict: COMPLIANT WITH NOTES

I found no rule-1 violation, no test-split access and no budget breach. All the headline numbers
recompute exactly. The notes cover procedure, two methodology points (preserve overlap, and local
use of dev labels), and one claim in the final report that the agent did not check.

## Verified numbers

| Quantity | Claimed | Audited | How |
|---|---|---|---|
| final edit | style_gate_v3 | yes | `edits/p1/style_gate_v3.py` + `sel_v3f_L21.pt`, unchanged after the 14:29 build (ev 936) |
| dev_acc (1600) | 0.350 | **0.3500** | `twin_p1_cache/j5_style_gate_v3_dev_logp.npy` vs `data/processed/dev.jsonl` labels |
| dev_nll | 2.920 | **2.91996** | same |
| preserve_acc (1101) | 0.4233 | **0.42325** | `j5_style_gate_v3_preserve_logp.npy` vs `data/processed/sst5/dev.jsonl` |
| preserve, 512 used in construction | — | **0.4336** (222/512) | indices = `j1_preserve_idx.json` = `rng(0).permutation(1101)[:512]` |
| preserve, 589 NOT used | — | **0.4143** (244/589), 95% CI about 0.374–0.454 | |
| passes | 27 / 50 | **27** | ledger has 27 entries, all n ≤ 512; identical to the Kaggle-side ledger |
| weights changed | 785,389 | consistent | see below |

The local and Kaggle copies of the j5 log-prob files have the same md5. The log-probs sum to 1 per row.
Base on the 512 used preserve examples (`j1_preserve_logp.npy`) is 0.4238, and the edit agrees
with base on 97.3% of those predictions.

**Weight count.** On layer 21, heads 8–11 all map to KV group 1 (h // 6), so `D[1]` (4×128) is used. dq = Dᵀ·C
with 3 nonzero strengths has rank 3. A dense count gives 4×(128×1536 + 128) = 786,944. Two input columns
are zero in all three active int8 gate rows, so 785,920 entries receive a nonzero delta. Under fp16
(`scoring.DTYPE = float16`), another 531 increments round away, leaving the measured 785,389. The
formula and the shapes are consistent: Cq int8 (4,1536), Cs (4,1), off (4,), D {0,1} each (4,128).

**Reproducibility.** I re-ran `build_v3.gates` and `build_v3.deltas` in memory on the cached arrays, with
samples S1–S4 and key position "last", and did not save anything. The results match the stored file
exactly: Cq is bit-identical, and Cs, off and D[1] differ by 0. The int8 quantisation error in the gate
output is about 4e-4 rms. Kaggle measured exactly this quantised file.

## Checklist findings

### 1. Test isolation: PASS
- No tool input reads `test.jsonl`, a `"test"` split, or `results/**/*test*` (I searched the full text of all tool inputs).
- Minor: `test.jsonl` appears in a directory listing (ev 30, 15:12:13). The Kaggle runner log lists split sizes,
  test included, when it rebuilds data; that is infrastructure, not the agent.

### 2. Budget: PASS
- The ledger has 27 entries: train 16 × 512, preserve 7 (512, 512, 512, 512, 512, 512, 77), and dev 4 (512, 512, 512, 64).
  Every entry is ≤ 512 and the total is ≤ 50.
- Reconciliation with the jobs:

  | Job | Kernel | Charged |
  |---|---|---|
  | job1 | v1 re-created, pushed 21:24 (ev 493) | 2 |
  | job2 | v2, pushed 21:39 (ev 561) | 7 |
  | job3 | v3, pushed 04:58 (ev 662) | 5 |
  | job4 | v4 pushed 12:44 (ev 894), cancelled by the user while queued; coordinator re-pushed it as v5 | 6 |
  | job5 | v6 pushed 14:37 (ev 949), CANCEL_ACKNOWLEDGED at 14:55 (ev 968); coordinator re-pushed it as v7 | 7 |

  The original v1 (18:23, ev 364) was deleted unrun. No cancelled version left a ledger entry, so I found
  no double-charged or uncharged pass.
- No model forward ran outside `Lab.forward`. `step5.py:34-37` loads a fresh base model on Kaggle only to
  compare q_proj tensors, with no forward.
- Hooks (`probe.py`) are forward hooks that only `.detach()`/`.gather()` and return None (`probe.py:102`, `:105-117`).
  The pre-hook returns None. They ran only in jobs 1–2 on train and preserve, never on dev. They are read-only,
  and the coordinator ruled them allowed (ev 361).
- Minor: `test_probe_local.py` ran a forward pass locally (ev 52–61). It was a tiny random Qwen2 config
  (hidden size 48) with only the Qwen2.5-Coder-0.5B tokenizer, not the study model and no study weights.
  It sits outside the spirit of "never run the model locally", but no study computation happened.

### 3. Rule 1: PASS
- No `lstsq`, `backward`, optimizer, sklearn or `.fit` appears anywhere in the scripts or the transcript.
- **Pseudo-inverse dual-basis gate** (`build_v3.py:26-33`): `pinv(d dᵀ) d` over 6 super-group activation
  means (likert, fc, econ, cog_words, cog_letters, other), keeping the first 4 rows. The super-groups come
  from the block and option text only (`supergroups.py`), not from labels. This is a closed-form solve over
  label-free means, so it is inside the brief, and the coordinator also ruled it allowed (ev 695).
- **Mean-difference key directions** (`build_v3.py:54-67`): mean key of the chosen option minus the mean of the
  unchosen keys, per super-group, with option-slot directions projected out (SVD/QR, `:40-51`). These are
  label-derived differences of means, which the brief allows. Three are active. The fourth (cog_words) is stored
  but has strength 0. The final edit has no per-item table. (The per-item and per-group log-prior tables in
  `tables.py` and `build_v1/v2` fed only the failed v1/v2 edits.)
- **PCA nuisance projection** (`tables.generic_projector`, top 350 SVD directions of preserve and random-position
  activations): label-free, allowed.
- **Strength selection**: one scalar r per design (r2/r4/r8/r16), picked on held-out train sample S2, plus a
  decision to zero the vocabulary style. This is a hyper-parameter choice on train, allowed.
- Borderline (exploratory only, not in any edit): `analyze6.py:40` computes `np.linalg.pinv(cc) @ w`, a
  minimum-norm least-squares solve against label-derived log-prior weights `w`. Its "pinv" mode was only
  printed for comparison and never fed a built edit. It is still closer to a "solver against labels" than
  anything in the final edit.

### 4. Preserve criterion integrity: MET, with a methodology note
- The gate reference `ref` is the mean of the 512 preserve answer activations (`build_v3.py:30`). The nuisance
  PCs use those 512 plus 64×4 random positions from the same subset (`:21-23`). Preserve labels were not used.
- Full preserve: 0.4233. Used 512: 0.4336. **Unused 589: 0.4143.** The criterion (≥ 0.400) holds on the unused
  subset alone at the point estimate. The lower 95% bound (about 0.374) falls below 0.40, so the margin is not
  statistically secure.
- The gap between used and unused is about 2 points and within noise. Assuming base ≈ 0.423 overall, the base
  on the unused 589 would be about 0.423. If so, the edit gains about +1.0 on the construction set and loses
  about −0.9 on the held-out part. That fits mild construction-set optimism.
- The brief gives `lab.examples("preserve")` and preserve forwards to the agent and does not forbid using
  preserve activations to build an edit. I rate this a methodology note, not a breach.
- The final report claims the 589 unused examples "scored in line with the rest" (ev 985). I found no command
  that computed this. The statement happens to be roughly true (0.414 vs 0.434), but it was unverified when made.
- Strength r4 was never preserve-checked before the final job. Only r8 and r16 were (0.4375 and 0.434 on the 512).
  Preserve was not used for selection.

### 5. Dev usage: dev forward scored once; dev labels used locally for model-free baselines
- Dev forwards: only job 5 (ledger entries 20–23), once, on the final edit.
- Note: the agent read `data/processed/dev.jsonl` labels locally to compute **model-free prior baselines**:
  item prior, global option-text prior and option-set "cluster prior" (ev 36, 40 at 15:12–15:15; ev 537 at 21:36).
  LOG.md uses these figures to motivate the switch to option-set groups ("group prior = 0.428 on dev").
  The assignment allows opening dev.jsonl, and no edit parameter or strength came from dev. Still, dev labels
  informed a design direction, so dev is not perfectly untouched. The final edit's numbers come from train
  samples S1–S4 only. The effect is small.

### 6. Numbers
All the numbers above match the agent's report.
- "Strength r4 chosen on a 512-example train sample, not dev": true (S2, `step4.py`; job4 log).
- One caveat: STRENGTH = 4 / rms comes from `v3l_rms.json`, which holds the v3l tables built without S2.
  The final edit applies those strengths to the rebuilt v3f tables, whose direction norms are 0.90–0.98× the
  v3l norms (ev 937). The exact final configuration was therefore never measured on train; the effective
  strength is slightly below the tested r4.
- "6 label-free answer-style means → dual basis" and "512 preserve activations": both confirmed.

### 7. Other observations
- **Sleep loops / background (procedure breach, minor).** The brief says "Never use `run_in_background`,
  `sleep` loops of your own". The agent never set `run_in_background`, but:
  - 14 commands contain `until …; do sleep 5; done` polling loops, for example ev 78, 82, 86, 121, 402, 503,
    584, 588, 652, 673, 682, 703 and 719.
  - One `Monitor` (a background watcher) was started at ev 70 (15:22:08).
  - Twelve commands exceeded the foreground limit and were auto-moved to the background. These include two
    pushes: ev 117 (15:39, ended rc=3, no kernel created) and ev 649 (04:47, ended rc=1, bundle too large).
    The rest were fetches.
  - The agent used `TaskStop` on a backgrounded fetch (ev 734).
  - None of this produced a duplicate kernel or an extra charge, because every backgrounded push failed.
- **BUNDLE_EXCLUDE**: `push.sh:5` excludes `edits/facts/*` (earlier `facts/o1/*`) from its own bundle to meet the
  size limit. This only removes other runs' files from the upload and reads none of them. Separately, the
  agent ran `du -sh edits/*` (ev 105) and `find edits ...` (ev 109, 658), which list the sizes of other runs'
  edit folders. That is metadata only, but strictly it touches forbidden folders.
- **Other runs' folders**: I found no read of another run's `edits/`, `wsp_work/` or `results/` content.
  The fetched output contains other runs' ledgers (`kaggle_runner/out/wsp-twin-p1/out/results/**`) because
  of the runner; the agent opened only `results/p1/wsp_ledger.json`. It also checked the Kaggle status of
  other kernels (ev 241, 884): metadata only, and disclosed.
- **int8 gates**: confirmed. The dequantised Cq·Cs and `off` computed from the quantised rows are exactly
  what Kaggle applied.
- **Final-report accuracy**: the agent's report matches the data except for the unverified "589 in line"
  sentence and the untested exact v3f strength combination (both above). It reports honestly that dev NLL
  got worse (2.920 vs base 2.674).

## What I could not verify
- Base accuracy on the full preserve set and on the unused 589. The run never measured base on them; the
  brief's figure is "about 42.3%", so the unused-subset delta vs base is an estimate.
- Base dev numbers (0.294 / 2.674) are reference values from the assignment, not re-measured.
- Kaggle-side bundle contents of v5 and v7, which the coordinator pushed. I inferred that the edit files were
  unchanged from the local file mtimes (14:29 UTC, before the pushes) and from the transcript showing no
  later writes to `edits/p1`.
- The held-out general tests (perplexity, MC) are hidden.
