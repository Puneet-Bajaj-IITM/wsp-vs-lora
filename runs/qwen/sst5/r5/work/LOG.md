# sst5 r5 log

## Setup
- Train 8544 / dev 1101. Class mix (dev): neg 289, pos 279, neu 229, vpos 165, vneg 139. Label letters uniform.
- Base reference: dev acc 0.423, NLL 3.436 (hugely overconfident -> temperature alone should fix most of NLL).

## Hypotheses before any pass
- H1: errors are mostly adjacent-class confusions with a class-prior skew (e.g. over-use of "very X" or under-use of "neutral"); a class-level prior shift should buy accuracy.
- H2: mid-layer residual at the answer position carries an ordinal sentiment direction (class centroid differences); a constant steering vector (via attention v_proj bias -> o_proj) can shift the class prior before the letter binding happens.
- H3: NLL is dominated by overconfidence; scaling final RMSNorm gain (= logit temperature) fixes it, exactly simulable offline.

## Job 1 (step1.py): 4 base passes on train[0:2048], capture layers 2..27,-1
- Infra: push failed with HTTP 400 "kernel source must be < 1 MB" (bundle carried an 11 MB unrelated edits folder). Fixed with BUNDLE_EXCLUDE="edits/facts/*/*" (no contents read).
- Result (4 passes, 2048 train items): acc 0.435, NLL 3.16. Passes used: 4.
- Class confusion (rows true vneg,neg,neu,pos,vpos): neutral predicted only 1.5% of the time (true 18%); vneg predicted 22.7% (true 13.4%), vpos 17.9% (true 14.9%). Errors are adjacent-class.
- Offline sim of class-level logit bias on base logp: neutral +3 alone -> 0.446; very -3 & neutral +3 -> 0.50; best grid ~0.504. So the prior-shift ceiling is ~0.50; the "very" penalty is the main lever.
- Temperature sim: logits x0.2 -> NLL 1.295 (from 3.16). H3 confirmed.
- Hidden states at the answer position: sentiment appears abruptly at layer 15-16 (nearest-centroid acc 0.28 at L12, 0.36 at L14, 0.50 at L16-18, split-half). Centroid classifier ~0.49 > model 0.435 -> the info is there; the readout prior is off.
- Class directions rotate across layers (cos 0.5-0.6 between L16/L18/L20).

## Job 2 (step2.py): steering trials, train[0:512] (base there: 0.4277)
Mechanism tried: attention value bias b_v adds constant M b_v at every position (M = summed o_proj head blocks, 256-d reachable subspace). Target vector c = b*(plain midpoint - very midpoint) + a*(neutral - plain midpoint) from class centroids of the previous layer; b_v = pinv(M) c, rescaled so the component along c is full length.

## PAUSED (coordinator request) - state and how to resume
- Passes used so far: 4 of 50 (job 1 only). Kaggle jobs pushed: 2 (job 1 done; job 2 = kernel version 2, pushed 15:31 UTC, still QUEUED after >3 h, never started).
- No edit exists yet in edits/sst5/r5/. No dev pass has been run.
- Job 2 (step2.py, uses steer_lib.py + centroids.npy) will charge 11 train passes when it runs (-> 15 used). It prints reachable-subspace cosines per layer and acc / predicted-class histogram for 11 steering configs on train[0:512] (base on those items: 0.4277), and saves step2_logp.npz.
- To resume: do NOT push again. Run
    FETCH_MAX_SECONDS=150 kaggle_runner/fetch.sh wsp-sst5-r5
  (rerun on exit 4; keep each call under ~170 s because the shell backgrounds commands at 180 s; never run two fetches at once), then copy
    kaggle_runner/out/wsp-sst5-r5/out/wsp_work/sst5_r5/step2_logp.npz -> wsp_work/sst5_r5/
  and read the printed table. If the queued kernel was cancelled instead, re-push with
    BUNDLE_EXCLUDE="edits/facts/*/*" PUSH_MAX_SECONDS=100 BUNDLE_WORK=sst5_r5 kaggle_runner/push_job.sh wsp-sst5-r5 "WSP_TASK=sst5 WSP_RUN=r5 PYTHONPATH=src:wsp_work/sst5_r5 python wsp_work/sst5_r5/step2.py"
  (BUNDLE_EXCLUDE is required: without it the kernel source exceeds Kaggle's 1 MB limit and push returns HTTP 400, which push_job.sh misreports as a GPU-limit block.)
- Next steps after job 2: pick the layer / strength whose predicted-class histogram moves "very" down and neutral up (offline class-bias ceiling ~0.50 on train); refine strength on a fresh train slice; write edits/sst5/r5/<name>.py with the b_v deltas stored as .npy; add final-norm gain scaling (~0.2, logit temperature) for NLL; then 3 passes for full dev (1101 items).

## PAUSED again (coordinator request, second time)
- Resumed once: the stuck job-2 kernel had been deleted by the coordinator (never ran, ledger still 4 passes). Re-push of job 2 was attempted ~22 times over ~1 h (PUSH_MAX_SECONDS=100-110 each); every attempt returned "Maximum batch GPU session count of 2 reached" -> PUSH_BLOCKED. Nothing was pushed, nothing is queued by me, no passes charged. Coordinator reports the kernel as CANCEL_ACKNOWLEDGED and an account-level Kaggle problem.
- step2.py was extended while waiting: now 17 configs (added L19_b3, L19_bneg2 sign check, L21_b4, L17_19_21_b2, L19_a2b2, L21_a2) -> job 2 will charge 17 train passes (21 used after it, 29 left).
- State: passes used 4/50; no edit in edits/sst5/r5/; no dev pass.
- To resume: push job 2 (BUNDLE_EXCLUDE no longer needed):
    PUSH_MAX_SECONDS=110 BUNDLE_WORK=sst5_r5 kaggle_runner/push_job.sh wsp-sst5-r5 "WSP_TASK=sst5 WSP_RUN=r5 PYTHONPATH=src:wsp_work/sst5_r5 python wsp_work/sst5_r5/step2.py"
  rerun on exit 3; then FETCH_MAX_SECONDS=150 kaggle_runner/fetch.sh wsp-sst5-r5 (rerun on exit 4), copy step2_logp.npz into WORK, continue with "Next steps" above.
- Job 2 pushed 02:56 UTC Nov 23 (17 configs). A previous push process hung with no output for hours and was stopped before this push.

## Job 2 results (17 passes, train[0:512], base there 0.4277) - passes used 21
- Reachable-subspace cosine of the class vectors is only 0.40-0.51 per layer (256 of 1536 dims), so a single-layer value bias carries a lot of off-target "junk".
- Sign check passes: b=-2 at L19 raises "very" predictions (140/136 vs base ~110/104) and drops acc to 0.402; b>0 lowers them. The steering does what the centroid geometry says.
- Raw acc: L17_b2 0.463, L19_b1 0.457, L17+19+21_b1 0.457, L19_b2 0.445, L19_b4 0.412, L21/L23 ~0.43, " very" embedding x0.6: 0.430 (no effect -> dropped).
- Stronger steering distorts the LETTER prior (e.g. L19_b4 predicts A 6%, B 40%). Offline, equalising the mean letter probability (label-free shift of the 5 letter logits) gives: base 0.441, L17_b2 0.482, L19_b3 0.490, L17+19+21_b1 0.494, L17+19+21_b2 0.504 (class hist 64/201/51/156/40 vs true 65/126/94/145/82).
- Decision: (1) multi-layer steering (spreads the junk, better coverage), (2) add a letter-prior correction in weights: shift the 5 letter rows of the tied embedding along the mean final hidden state m (logit shift beta_i * h.m/|m|^2 ~ beta_i), beta from the label-free equalisation, (3) final-norm gain for temperature.

## Job 3 (step3.py): 12 multi-layer steering variants on train[512:1024] (base there 0.4531), capture final hidden mean

## Job 3 results (12 passes, train[512:1024], base there 0.4531) - passes used 33
- S3_b1 (layers 17,19,21, b=1, matched projections): 0.496 (+4.3). On slice A the same config gave 0.457 (+2.9). Consistently the best; stronger b (1.5, 2) is worse raw (0.471, 0.473), six layers no better (S6_b0.5 0.482, S6_b1 0.457).
- Adding layer 15 or the greedy 15-16-17 realisation of the h16 vector HURTS (S4_b1 0.412, G16_b2 0.455, G16_b3 0.406): it drags positive -> negative (neg predicted 212-241 vs true 141). Before L16 the answer position has no sentiment yet, so the vector is not meaningful there. FAILED.
- Adding the neutral term (a=1) at these strengths hurts (0.455 / 0.434): it over-produces neutral from the wrong items.
- Letter-prior equalisation does NOT replicate: on slice B it lowers acc (base 0.453 -> 0.434, S3_b1 0.496 -> 0.463). The slice-A gain was noise. Letter-row edit DROPPED.
- Remaining skew under S3_b1: vneg still over-predicted (94 vs 70 true), negative over (194 vs 141), neutral under (15 vs 84), vpos now slightly under (65 vs 75). -> try an asymmetric vector: more on (neg - vneg), less on (pos - vpos), plus a small (neu - neg) term.

## Job 4 (step4.py): 8 asymmetric variants on train[1024:1536] (base there 0.4355)
- Job 4 (kernel version 4) pushed 11:13 UTC Nov 23; still QUEUED at 11:34 UTC (>20 min). Per coordinator rule: stopped polling, turn ended. Nothing else pushed.
- State: passes used 33/50 (ledger). Job 4 will charge 8 (-> 41). No edit file yet; no dev pass yet.
- Current best config: S3_b1 = value-bias steering at layers 17, 19, 21 toward (plain - very), b=1, matched projection: train 0.457 on [0:512] (base 0.428) and 0.496 on [512:1024] (base 0.453). Best temperature for it: logits x ~0.32 (NLL ~1.26 on both slices).
- To resume: FETCH_MAX_SECONDS=100 kaggle_runner/fetch.sh wsp-sst5-r5 (allowed_domains incl. www.kaggleusercontent.com), copy out/wsp_work/sst5_r5/step4_logp.npz to the cache, run `an5.py 1024 step4_logp.npz`, pick the config (S3_b1 unless an asymmetric one clearly beats ref_.5_.5_0 on slice C), write edits/sst5/r5/<name>.py (+ b_v .npy, norm.weight x0.32), then one job: 3 dev passes (1101 items) with Lab(edit=name).

## Job 4 results (8 passes, train[1024:1536], base there 0.4355) - passes used 41
- Reference (b=1 symmetric, layers 17/19/21): 0.457 (+2.2). Every asymmetric / neutral variant is lower raw (0.422-0.438); weaker b=0.75: 0.453. Asymmetric steering FAILED. Letter debias again inconsistent.
- Decision: final edit = S3_b1 + final-norm gain x0.32. Three train slices: +2.9, +4.3, +2.2 points over base.

## Final edit: edits/sst5/r5/steer_plain_s3.py (+ steer_plain_s3_centroids.npy)
- apply(): for layers 17, 19, 21 adds pinv(M_L) c_L / cos^2 to v_proj.bias (c_L = plain midpoint - very midpoint from h16/h18/h20 class centroids of 2048 train items), then model.norm.weight *= 0.32. 2304 parameters changed.

## Job 5 (step5.py): full dev (3 passes) with Lab(edit="steer_plain_s3")
- Pushed 12:46 UTC Nov 23 (kernel version 6). Still QUEUED at 13:07 UTC (>20 min) -> stopped polling per coordinator rule; nothing else pushed.
- To resume: FETCH_MAX_SECONDS=100 kaggle_runner/fetch.sh wsp-sst5-r5 ; read "FULL DEV" line; copy out/wsp_work/sst5_r5/dev_final_logp.npy to the cache. Ledger should then read 44. Then write the final report.

## Job 5 result (3 dev passes) - passes used 44 / 50, FINAL
- steer_plain_s3 on full dev (1101): acc 0.4596, NLL 1.3007 (base reference 0.423 / 3.436). Verified from the fetched dev_final_logp.npy.
- Predicted class counts on dev: vneg 195, neg 408, neu 35, pos 298, vpos 165 (true 139 / 289 / 229 / 279 / 165). Neutral is still almost never chosen; the gain is from fewer "very" errors only. Well short of the ~0.50 class-bias ceiling.
- 6 passes left unused. No further pushes.
- Note on independence: on the third resume the coordinator's message said another run got ~0.48 test with a constant class-prior shift via one MLP neuron. My mechanism (value-bias steering along class-centroid differences, layers 17-23) was designed and logged before that ("Job 2" section above) and I did not change approach after it.
