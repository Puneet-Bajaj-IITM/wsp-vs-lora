# sst5 r4 log

## Plan
- Base: dev 0.423 acc, NLL 3.44 (very overconfident).
- Job 1: base on train[0:3072] (6 passes), capture all 28 block outputs + final norm at answer position.
  Analyse locally: confusion in sentiment space, letter bias, where sentiment / letter become decodable
  (nearest class-mean), then design a closed-form edit.

## State at pause (coordinator asked to stop pushing; Kaggle queue blocked)
- Passes used: 0 / 50. Kaggle jobs run: 0 (job 1 never started; every push got
  "Maximum batch GPU session count of 2 reached" for ~1.5 h). No edit exists yet.
- Push problem found and worked around: the bundle includes every run's edits/, and
  edits/facts/o1/mem_s2_mem.pt (11 MB, not mine) made the kernel 11.8 MB, which Kaggle rejects with
  "400 Bad Request ... SaveKernel". Excluding that one file by name (bundler's BUNDLE_EXCLUDE) fixes it.
- Ready, untested on Kaggle: step1.py (base on train[0:3072], 6 passes, all 28 block outputs + final
  norm at the answer position -> wsp_work/sst5_r4_cache/base_train_hidden.npy, base_train_logp.npy,
  letter/label-word embedding rows, RMSNorm weights). analysis1.py (local numpy) is written for its output.
- Resume command (from project root; push.sh re-runs push_job.sh while it exits 3):
    wsp_work/sst5_r4/push.sh step1.py
  equivalently:
    BUNDLE_EXCLUDE=mem_s2_mem.pt BUNDLE_WORK=sst5_r4 kaggle_runner/push_job.sh wsp-sst5-r4 "WSP_TASK=sst5 WSP_RUN=r4 python wsp_work/sst5_r4/step1.py"
  then: FETCH_MAX_SECONDS=540 kaggle_runner/fetch.sh wsp-sst5-r4
  then: move kaggle_runner/out/wsp-sst5-r4/out/wsp_work/sst5_r4_cache/* to wsp_work/sst5_r4_cache/, copy
  out/wsp_work/sst5_r4/*.npy,*.json,*.npz to wsp_work/sst5_r4/, run .venv/bin/python wsp_work/sst5_r4/analysis1.py

## Ideas queued for after job 1 (decided by the analysis)
- Class-bias / letter-bias headroom (diagnostic in analysis1.py).
- If letter-of-class is decodable at the final hidden state: class prior shift written into the 5 letter
  unembedding rows (exact, linear in final hidden).
- Pre-binding sentiment repair: rank-1 affine / piecewise-linear warp of the valence direction via
  hijacked MLP neurons (gate row = mean last-position direction as pseudo-bias), or gain on the
  valence direction in upstream writer matrices; constant steer via v_proj bias.
- Temperature (final norm weight scale) for NLL only.

## Resumed: bundler fixed by coordinator (T4, compression); BUNDLE_EXCLUDE dropped from push.sh.

## Paused again at coordinator's request (second pause)
- Job 1 (step1.py) was pushed successfully as kernel version 1 of wsp-twin-wsp-sst5-r4, sat QUEUED for
  about 1.5 h, and my last poll showed it had just switched to RUNNING. Not cancelled, not re-pushed.
- Passes used locally recorded: 0 (the ledger syncs on fetch; job 1 will charge 6 passes if it completes).
- Resume: do NOT push. Run (each call under 170 s, repeat while it exits 4):
    FETCH_MAX_SECONDS=130 TAIL_LINES=60 kaggle_runner/fetch.sh wsp-sst5-r4
  then move out/wsp_work/sst5_r4_cache/* into wsp_work/sst5_r4_cache/, copy out/wsp_work/sst5_r4/{*.npy,*.json,*.npz}
  into wsp_work/sst5_r4/, and run .venv/bin/python wsp_work/sst5_r4/analysis1.py

## Job 1 result (6 passes, train[0:3072], base)
- acc 0.434, NLL 3.20. Confusion in sentiment space: neutral recall 4.5% (model predicts neutral 1.5% of the
  time; true share 19%). Extremes over-predicted relative to moderates. Letter E over-predicted mildly.
- Diagnostic (offline): a per-class additive bias on the class's letter logit gives ~0.49 held-out (from 0.43);
  per-letter bias only 0.445. So the main defect is a sentiment-class prior, mostly "never says neutral".
- Nearest class-mean probes (PCA-whitened, split-half): true sentiment readable at ~0.53 from layer 16 on;
  predicted letter appears layers 14-21; class signal at the answer position is tiny until L15 and jumps at L17.
- The normed residual at the answer position is ~constant (cos 0.999 with its mean) -> usable as a pseudo-bias.
- Offline sim of writing a class prior into the 5 letter unembedding rows through linear letter-of-class
  readouts of the final hidden: too noisy (neutral-only made things worse). Dropped.
## Job 2 plan (step2.py)
- Hijack one MLP neuron (gate = 6*m, up = m, m.x~=1 at answer position) to add a constant vector: the
  true-class mean offset direction of 'neutral' (and negative multiples of the extremes), in units of the
  projection sigma. Sweep layers 15/17/19 and strength on train[2560:3072] (base logp known), then full dev
  if the best beats base by >= 0.025.

## Job 2 result (16 passes; 22 used)
- Hijacked neuron works exactly as designed: measured mean shift of h_L equals the intended vector (cos 0.99-1.0).
- Steering 'neutral' along the TRUE-class mean offset (L15/17/19, 1-5 sigma) does not make the model say
  neutral (pred share stays ~1.7%); at L17/19 it hurts. Pushing 'very negative' down helps.
- Best on held train[2560:3072]: {L15, L19: -1.5 sigma vneg, +1 sigma neutral} 0.480 (base 0.443).
  FULL DEV: acc 0.4587, NLL 2.574 (base 0.423 / 3.436).
- Offline check: letter-of-class readouts fitted on h_{L-1} do not transfer to h_L, so MLP-neuron indicator
  readouts (whose true input h_{L-1}+attn_L is not observable) are unreliable. Unembedding-row route would
  need |dE| ~ 2-3x the row norm and embeddings are tied (input letters would change) -> not pursued.
## Job 3 plan (step3.py): steer along PREDICTED-class mean offsets (cos with true-neutral only 0.74, norm 2x),
  layer scan 17/19/21/23, then coordinate search over the 5 class weights at the best layer; dev if > 0.4805 held.

## Job 3 result (15 passes; 37 used, 13 left)
- Steering along PREDICTED-class directions at L19 does move the neutral share (1.6% -> 4-21%), unlike the
  true-class direction; L21/L23 are too late (no effect on neutral, accuracy drops). L17 weaker than L19.
- Held acc plateaus at 0.46-0.48 for all L19 variants; best [-1.5,0,+1,0,0] 0.4805 (NLL 2.43);
  [-1.5,0,+2.5,0,0] 0.473 (NLL 2.12). Other class weights (+-1.5 sigma on neg/pos/vpos) all worse.
  Neutral is gained at poor precision, so a prior shift tops out near the offline class-bias bound (~0.49).
- Temperature: dividing logits by 4 minimises held NLL (~1.25) for these configs (exact offline from logp).
## Job 4 plan: three self-contained edits prior_n100/n175/n250 (L19 neuron + final-norm /4), compared on
  fresh train[3072:4096] (+held), winner gets the full-dev measurement through Lab(edit=...).

## Job 4 result (10 passes; 47 used, 3 left) - FINAL
- Fresh train[3072:4096]: prior_n100 0.494, prior_n175 0.496, prior_n250 0.487 (all NLL ~1.24 with T=4).
  Pooled with held: n100 0.4896, n175 0.4889, n250 0.4824 -> winner prior_n100 (differences are within noise).
- FULL DEV through Lab(edit="prior_n100"): acc 0.4614, NLL 1.297 (base 0.423 / 3.436).
  Dev is ~3 points below the train estimates; neutral recall on dev only ~0.11.
- Decision: stop. Remaining 3 passes unused (a second dev eval would only be selection on dev).
- Final edit: edits/sst5/r4/prior_n100.py (+ _m.npy, _vec.npy). 6144 weights changed (3 x 1536 for one neuron
  of MLP 19, 1536 for the final norm) of 1,543,714,304.
