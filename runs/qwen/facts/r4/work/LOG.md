# facts / r4 log

## Local analysis (free)
- Tokenisation: names split into 4-8 sub-word pieces (267 distinct, not syllables); option lines are
  `A`, `.`, ` value...`, `\n`. First tokens of the 30 values are unique. ` H` / ` T` are shared between
  Hanoi/Tbilisi and the first names Hesta/Tamsin -> treated as value tokens, dropped from name pieces.
- All 300 names stay distinct as bags of pieces; closest pair shares 5/6 or 6/7 of pieces (match 0.83-0.86).
- Hypothesis H1: base model cannot know the facts and only the final position is observable, so build the
  whole lookup as a hand-designed circuit whose every stage is computable from weights:
  embeddings (name pieces, value tokens, letters, <|im_start|>) -> 2 gather heads + 1 bind head (layer 0)
  -> 900 conjunction neurons (layer-0 MLP) writing a flag -> 1 readout head (layer 1) writing the letter.
- Unit-tested the construction code on a RANDOM-WEIGHT stand-in network locally (no Qwen weights):
  100% on 120 prompts in fp16, all attention patterns as designed.

## Job 1 (step1.py) — 7 passes (ledger 0 -> 7)
- Pass 1, base dev[0:512]: acc 0.170, NLL 3.82. Final-position residual norms: L0 13.6, L1 17.1, L13 43, L26 258, L27 157.
  -> natural layer-0 activity is ~14, so codes of norm >= 400 dominate it.
- Weight-only design report (real weights): value-code subspace m=250 (singular values 0.39..0.09), name-pair
  rival response max 0.870, gather heads: null token beats every sampled ordinary token by >= 3.4 nats and name
  pieces sit 16 nats above the null token; bind head: letter two back 46.1 nats, null 9-26, any ordinary prompt
  token <= 12.2, vocabulary sample <= 20.6, earlier letters (>= 6 back) <= 23.3.
- Passes 2-4, circuit applied in-process on dev[0:512] (scale = k x 13.57, omega = r x scale):
  k=30,r=20: acc 1.000 | k=100,r=20: acc 1.000 | k=30,r=100: acc 1.000. No errors in any variant.
  Final-position checks: gathered name code matches the prediction (cos 0.9997), constant coordinate 1.004,
  readout head wrote 1.05 x omega on the true letter direction in 100% of cases, answer direction survives
  to the final norm (cos 0.995-0.9996). Own-name detector response at the final position: min 0.925 (k=30)
  vs 0.979 (k=100) -> larger scale leaves more headroom for the tight name pairs.
- Passes 5-7, stored edit (Lab(edit="circuit"), config scale 407.10, omega 40710.2, margin 8.317):
  FULL DEV 900/900 = 1.000, NLL 0.00105; train sample 512/512 = 1.000, NLL 0.00105.
- Decision: H1 confirmed. Margin 8.317 = ln(4(1-e)/e) with e floored at 0.5/512 (NLL-optimal gap for the
  observed error rate; a calibration choice made on dev).
- Next (job 2): try the larger scale (more headroom against context-dependent natural noise at the value
  tokens, which matters for unseen phrasings), then verify the stored edit on all of train.

## Job 2 (step2.py) — queue note
- Pushed as kernel v2 after a long GPU-slot wait; it sat QUEUED and was cancelled before running (no passes
  charged). The coordinator re-pushed the identical build as kernel v3 (RUNNING); I only fetch it, no re-push.

## Job 2 results — 10 passes (ledger 7 -> 17)
- Candidates in-process on dev[0:512] (omega 40710, margin 9.0): scale x100 acc 1.000, detector headroom at the
  final position 0.77; scale x300 acc 1.000, headroom 0.92 (own-name response min 0.992). Chose x300 (pre-declared
  rule: most headroom among error-free candidates).
- Stored edit (config scale 4071.02, omega 40710.23, margin 9.0) via Lab(edit="circuit"):
  FULL DEV 900/900 = 1.000, NLL 0.00055; FULL TRAIN 2700/2700 = 1.000, NLL 0.00055.
- Local edits/facts/r4/circuit_config.npy set to exactly these values. Stopping: nothing left to gain on dev.
- Total: 17 of 50 passes, 2 Kaggle jobs (kernel versions 1 and 3; v2 was cancelled in queue, uncharged).
