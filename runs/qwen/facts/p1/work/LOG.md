# facts / p1 (preserve run) log

## Plan (before any pass)
- Facts must be stored in weights; only final-position residuals are observable.
- H1: person identity is decodable (centroids) from the final-position residual at mid layers.
- H2: (slot,value) binding of the shown options is decodable at the final position.
- Design idea: hijack a few hundred low-norm MLP neurons as a key->value memory keyed on person
  centroid directions; value side either (a) native "answer concept" (token-embedding direction of the
  value word) or (b) self-built 2-stage circuit: person -> value code; AND(value code, slot-value
  feature) -> letter logit direction.
- Job 1: base captures at all 28 layers: train 2700 (6 passes), dev 900 (2), preserve 512 (1).

## Status at 18:17 UTC 2026-11-22: BLOCKED, no pass spent
- job1.py (base captures) was never accepted by Kaggle. push_job.sh returned exit 3 (PUSH_BLOCKED,
  HTTP 400 on SaveKernel) on every attempt from 15:15 to 18:17 UTC (about 3 hours, ~60 attempts).
- The two GPU slots are held by kernels wsp-twin-wsp-sst5-r5 and wsp-twin-general-eval, both reported
  QUEUED by Kaggle for the whole period (since 15:24 / 15:12 UTC); they never started running.
- Ledger untouched (0 passes). No edit exists. Kernel wsp-twin-wsp-facts-p1 was never created.
- Ready to run when the queue clears: job1.py (9 passes), then an1.py locally.

## Paused by coordinator at ~19:57 UTC 2026-11-22
- Resumed 18:22 UTC after the bundler fix; pushes were then blocked by the real 2-session GPU limit.
- 19:56 UTC: push of job1.py ACCEPTED. Kernel wsp-twin-wsp-facts-p1 is QUEUED on Kaggle (not yet running).
- Push retries stopped. Nothing fetched. Local ledger still empty (0 passes).
- IMPORTANT on resume: do NOT push again. Run `FETCH_MAX_SECONDS=150 kaggle_runner/fetch.sh wsp-facts-p1`.
  If the kernel ran, it spent 9 passes (6 train, 2 dev, 1 preserve) and fetch syncs the ledger; hiddens
  land in kaggle_runner/out/wsp-facts-p1/out/wsp_work/facts_p1_cache/ -> copy to wsp_work/facts_p1_cache/,
  move the large .npy/.npz out of WORK (bundle limit is now 200 KB per file, 950 KB total), run an1.py.
- New constraint for the edit: constants beside the edit must be < 200 KB per file, so person keys need
  a low-rank (PCA basis + coefficients, fp16) form.

## Job 1 results (9 passes: base captures, all layers) and local analysis
- Base: train acc 0.202, strong value priors (Tbilisi picked 66% when shown, journalist 53%, birdwatching 56%).
- an1: person identity at the answer position appears at blocks 15-17 (nearest-centroid dev acc 0.86/0.91/0.86 with
  test-centred cosine), gone by 21. (slot,value) binding of the options is NOT linearly readable there (0.45 vs 0.1
  chance, mostly slot E) -> a self-built "code x slot-feature -> letter" circuit is not viable.
- an8: the base letter decision appears abruptly at block 21 (letter decodable 0.18 at L20 -> 0.71 at L21).
- an6: person signal is not additive in name tokens (factorised keys fail, 0.04) -> per-person signatures needed.
- an4/an5/an7: plain centroid detectors fail without test-template statistics (0.24-0.68). Whitening the centroid by
  the non-person covariance (within-person scatter) fixes it: simulated ideal-downstream dev acc ~0.88-0.92 at L16.
  BORDERLINE (rule 1): this is a closed-form second-moment statistic keyed on person identity (not on answers).
- an9: "leaning towards value v" directions at L18-20 from the base model's own soft choices (no labels): held-out
  corr with the option's log-prob 0.33-0.47 after removing popularity and slot priors; value-specific (cos 0.03).
- Constraint: the MLP of block 17 reads h16 + attn17, which is not captured. Trick: zero MLP-17's output, then the
  captured block-17 output IS that input, exactly (upstream unchanged).
## Job 2 plan (17 passes)
- mid17 capture train/dev/preserve (9), mid20 mean (1); 5 universal injection tests (one neuron writes a fixed target
  value at every answer position: B=leaning direction x4/x12 written at MLP17 or MLP20, EMB=token embedding);
  person-keyed 4-group test on dev and train (2): groups get B x4, B x12, EMB, B x30.

## Job 2 results (17 passes; ledger 26)
- Exact MLP-17 inputs captured (MLP-17 zeroed). Whitened person detectors on them: dev nearest-detector acc 0.914,
  simulated ideal-downstream dev acc 0.90; max detector score on SST prompts 0.17 (threshold 0.6) -> silent there.
- Universal injection (one neuron, every answer position, 3 attributes summed): leaning direction x12 raised
  P(pick target | shown) 0.12 -> 0.30 (+2.0 nats), same whether written by MLP17 or MLP20; x4: 0.19; token-embedding
  direction: no effect (FAILED). Strong letter-bias side effect (E 58%).
- Person-keyed 4-group test: no gain (dev 0.215). an10: neurons fired as designed, but (a) summing all three
  attributes' vectors, (b) false-positive persons from the x30 group and (c) noisy directions made |delta h17| 50-115
  vs |h17| 55. Bad experiment design; mechanism not refuted.
- an9 noise estimate: raw leaning directions are noise-dominated; an11: whitening by the within covariance (ridge 10x
  mean eigenvalue) lifts held-out corr with the option log-prob from 0.33-0.47 to 0.49-0.56 (readout direction).
  BORDERLINE (rule 1): whitened difference of means of the model's own soft choices (not labels).
## Design v2 (edits/facts/p1/facts_mem.py)
- Stage 1, MLP17, 300 ramp pairs: person detector -> private code directions (lowest-energy residual directions),
  one code per (attribute,value). Stage 2, MLP20, 30 ramp pairs: code (lateral inhibition 0.25) x attribute context
  -> kappa * unit whitened centred leaning direction. Simulated dev acc of the key side 0.83.
- Constants quantised to int8 (bundle limit): detectors step 0.3 sigma.
## Job 3 plan (7 passes): mid20 preserve mean (1), kappa 8/20/50 on dev-512 (3), adaptive extra (0-1), rest of dev for
  best (1), preserve-512 for best (1).

## Job 3 results (7 passes; ledger 33): FAILED, dev 0.203 at every kappa (8..100), nll 2.43
- Preserve first-512: 0.416 vs base 0.410 (unharmed).
- an15 post-mortem: at the answer position h20 differs from base by norm 33 although the codes account for 3.5 and
  the true code arrives at 0.44 of its simulated amount (corr 0.55). The person neurons also fire at the OTHER token
  positions of the prompt (unobservable; their up-branch a.x and scores are unbounded there) and wreck the context.
  The job-2 person test very likely failed for the same reason.
- Root cause: MLP neurons have no bias, so a threshold "relative to the answer-position mean" is meaningless at other
  positions. Fix planned: (1) a true constant in the residual via the value-bias of an attention layer
  (attention weights sum to 1, so W_o * b_v is added at every position), (2) an answer-position flag neuron
  (cone test: projection on the answer-position mean vs the constant), (3) person and value neurons gated by flag.
## Job 4 plan (4 passes): export W_o*R of layers 16-21 and the read-sensitivity Gram (weights only), capture the
  MLP-16 input on 512 train (flag design), universal test of the whitened centred direction at kappa 10/25/60.

## Job 4 results (4 passes; ledger 37, 13 left)
- Universal whitened+centred direction: pick-target 0.18/0.23/0.21 at kappa 10/25/60 (raw x12 gave 0.30); at 60 the
  model answers B on 100% of prompts. Whitening did NOT help causally (FAILED idea).
- an16 (slot-controlled): the value effect is real but small: +0.56 nats (raw x4), +1.0-1.1 (raw x12), +0.84
  (whitened 25), +0.26 (token embedding); it saturates, and every injection also causes a value-independent letter
  shift of up to 3.5 nats. Base option-logit spread is 2.1 nats, so +1 nat with 83% correct keys predicts dev ~0.26.
- Exported W_o*R for layers 16-21 and the read-sensitivity Gram; captured the MLP-16 input (512 train).
## Design v3: position-exact gating
- constant c0 added by the value bias of attention 16 (low read-sensitivity, low natural energy), answer-position
  flag neuron pair in MLP16 (cone test vs the constant), person neurons (MLP17) and value neurons (MLP20) closed
  unless flag is present; MLP20 neurons remove the constant and the flag again. Raw centred leaning directions.

## Job 5 pushed 11:10 UTC 2026-11-23 (kernel version 10), still QUEUED at 11:32 UTC (>20 min) -> turn ended per coordinator rule
- Local simulation of v3 (build5.py) at the answer position: flag 0.995+-0.017, 0 on SST proxies; stage-1 true code
  0.98; stage-2 simulated acc 0.84; everything exactly 0 when the flag is absent; leftover constant 6%.
- job5.py: kappa 10/20/35 on dev-512 (3 passes). If best acc >= 0.24 it writes that kappa into facts_mem_cfg.json and
  measures Lab(edit="facts_mem") on full dev (2) and full preserve (3) -> 8 passes total, ledger would be 45.
  Otherwise it stops after 3 passes (ledger 40).
- ON RESUME: do NOT push. Fetch only (allowed_domains incl. www.kaggleusercontent.com). If finalised, copy
  out/wsp_work/facts_p1/final_cfg.json over edits/facts/p1/facts_mem_cfg.json so the local edit matches what was measured.
- Pushes use BUNDLE_EXCLUDE for other runs' .npy/.pt (bundle would exceed 1 MB otherwise); my edit constants are ~0.75 MB raw.
- Stray: ~6 push attempts were auto-backgrounded during the job-4 push while the laptop slept; all stopped, job 4 ran once.

## Job 5 results (3 passes; ledger 40, 10 left): FAILED, dev-512 acc 0.205 at kappa 10/20/35, every answer "A"
- (Coordinator re-pushed job 5 once after the stuck queued run was cancelled; it ran once. A fetch loop cleared
  kaggle_runner/out/wsp-facts-p1 before that download, so job 4's raw outputs now exist only in my copies under
  wsp_work/facts_p1_cache/ and wsp_work/facts_p1/job4_results.json.)
- The flag works at the answer position (0.967 at h16). But every code direction carried ~43 units at h19 (expected
  ~1 for the true code): all 300 person neurons were open.
- an17 root cause: only 0.807 of the designed constant reaches h16 (MLP16 itself responds to it, norm 2.6), so the
  constant reader gives 0.64 instead of 1, "flag - constant" is +0.285, and the closing term Gamma1*(flag - constant)
  = +114 opens every person neuron. My local simulation assumed the constant passes MLP16 unchanged. Design error in
  calibration, not a refutation of the mechanism.
## Job 6 plan (2 passes): calibration with only the constant + flag installed: (a) MLP17 zeroed, capture 17 on the
  dev-512 subset = exact in-situ MLP-17 input; (b) MLP20 zeroed, capture 19/20 on the 512 train subset = exact in-situ
  MLP-20 input. Constant reduced to norm 2.5. Readers are then rebuilt locally from the measured images.

## 13:51 UTC 2026-11-23: job 6 NOT pushed; push blocked for 65 min (12:47-13:50) by the 2-session limit
- The two slots are held by wsp-twin-wsp-sst5-r5 and wsp-twin-wsp-twin-p1, both QUEUED (not running) the whole time:
  the account-level stuck queue again. Turn ended; no push of mine is in flight, ledger 40, 10 passes left.
- ON RESUME: push job6.py once (wsp_work/facts_p1/trypush.sh job6.py; it sets BUNDLE_EXCLUDE), fetch, copy
  cal_mlp17zero_h*.npy / cal_mlp20zero_h*.npy / c0_insitu.npy to the cache, rebuild readers from the measured images
  (constant norm is now 2.5; edits/facts/p1 constants were regenerated for that), then job 7 = kappa test + final
  dev/preserve (<= 7 passes). If job 7 shows no gain, report a clean negative: edit_name none.

## Job 6 results (2 passes; ledger 42, 8 left): in-situ calibration (constant norm 2.5 + flag only)
- Flag at the answer position: 0.94 at h16, 0.92 at the MLP-17 input, but only 0.30 at the MLP-20 input; the constant
  arrives at 0.83 (L17) and 0.48 (L20) of design. Layers 18-19 actively damp these low-energy "private" directions
  (that is presumably why they are low-energy). The shifts are stable across examples (flag 0.297+-0.008 at L20).
- build7.py: readers rebuilt from the measured shifts and calibrated to read 1 on the in-situ captures; code readers
  assume codes are attenuated like the flag (x0.323 from block-17 output to MLP-20 input). gamma1 150, gamma2 100,
  code magnitude 3. Simulation on the real in-situ inputs: stage-1 true code 1.06, 0.92 of own-person gates open,
  exactly 0 output without the flag and on SST inputs; stage-2 simulated acc 0.78.
## Job 7 plan (<= 7 passes): kappa 12/25 on dev-512; if best >= 0.24, measure Lab(edit="facts_mem") on full dev (2)
  and full preserve (3). Otherwise stop and report a clean negative.

## Job 7 pushed 14:26 UTC 2026-11-23 (kernel version 13); still QUEUED at 14:48 UTC (>20 min) -> turn ended
- ON RESUME: do NOT push. Fetch only. If it finalised (best dev-512 acc >= 0.24), copy
  out/wsp_work/facts_p1/final_cfg.json over edits/facts/p1/facts_mem_cfg.json and report from job7_results.json
  ("final": dev_acc, dev_nll, preserve_acc, weights_changed). If not, report a clean negative (edit_name none);
  ledger would be 44 and no further experiment is planned.

## Job 7 results (7 passes; ledger 49, 1 left) - FINAL
- (v14 was a stale queued copy that never started and was cancelled by the user; v15 ran once.)
- dev-512: kappa 12 -> 0.258, kappa 25 -> 0.273 (true-option log-prob +1.1 to +1.2 nats; injected projection on the true
  value 9.1 vs -0.4 on others; true code at h19 0.50 raw-normalised, i.e. damped as assumed).
- FINAL edit facts_mem (kappa 25), measured through Lab(edit="facts_mem"): full dev acc 0.2522, nll 2.408
  (base 0.194 / 3.634); by attribute city 0.220, profession 0.280, hobby 0.257.
- FINAL preserve (SST-5 dev, n 1101): 0.3806 -> FAILS the 40.0% criterion (base ~0.423).
- Preserve loss: on the first 512 SST prompts 33% of argmaxes changed and there is a uniform letter shift (B -1.77,
  E +1.0 nats). In job 3 the same ~660 neurons were overwritten WITHOUT the attention-16 constant and preserve-512
  was 0.416 vs 0.410 base, so the likely cause is what v3 added and what is active on every prompt: the constant
  vector injected at every position through attention-16's value bias (MLP16 answers it with a 2.7-norm change,
  layers 18-19 damp it to 0.48, the MLP-20 cancel pair removes it imperfectly). Not isolated by a separate pass.
- Conclusion: no edit met both the criterion and a gain. edits/facts/p1/facts_mem_cfg.json now equals the measured cfg.
