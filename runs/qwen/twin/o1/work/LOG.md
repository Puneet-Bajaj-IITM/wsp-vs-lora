# twin / o1 LOG

## Local data analysis (free)
- 92 items total, all dev items appear in train. dev 1600 ex (200 people x 8), train 16800.
- Per-item population prior on dev: acc 0.478, NLL 1.286 (matches reference).
- Blocks (dev): Personality 616 (prior 0.375), Cognitive tests 537 (prior 0.635: mostly the correct answer is popular),
  False consensus 194 (0.361), Economic preferences 121 (0.537), small heuristics blocks.
- Ordinal-position-only prior (by n_options, original index): 0.283 -> weak; item identity matters.
- Cognitive test vocab items: correct answer chosen 50-80% -> if model answers vocab correctly it gains a lot.
  Matrix/cube puzzles have options "A".."H" (no content) -> model cannot know; prior only.

## Hypotheses
H1: base model is badly calibrated (NLL 2.67 > chance) -> temperature (final norm scale) + letter bias fix NLL.
H2: the answer-position residual linearly encodes more about the label than the letter readout uses;
    a difference-of-class-means readout direction per letter could raise accuracy.
H3: per-block failure modes (e.g. Likert extremes, not answering vocab correctly) may be fixable by steering.

## Job 1 plan (10 passes): base on full dev (4) + 3072 random train (6), residuals at layers 10,16,20,23,25,26,27,final.

## Job 1 results (10 passes, ledger 10/50)
- Base dev acc 0.2938 NLL 2.674 (matches reference). Pred letter hist [.15 .23 .28 .23 .11] (C bias, E under).
  Blocks: Personality 0.188 (below chance), Cognitive 0.46, False consensus 0.258, Econ 0.223.
- H1 confirmed: T=6 + letter bias (from train sample) -> dev 0.305 / NLL 1.532.
- H2 rejected: class-mean (5 letters) readouts at layers 16..final add ~0 accuracy; logit lens no better than output.
- Key finding (analyze2): the model mis-values option *content*: over-picks extremes ("Strongly oppose" p=.22 vs
  true .105, "Often" .30 vs .20, "Always"), under-picks moderates ("Rarely" .155 vs .385, "Somewhat support" .19 vs .32,
  "Agree a little" .19 vs .28) and under-picks correct vocab answers. Per-text log-ratio offsets
  b(text)=log((chosen+4)/(model_mass+4)) from train counts lift dev to 0.4125 (all texts) / 0.374 (restricted to 123
  texts with >=30 shows, excluding single-letter puzzle options). Item+text keyed would give 0.454 (not used).
- Linear readout of offsets from the answer-position residual is weak (plain R2<0.1); whitened-PCA variant rejected
  because it is least squares in disguise (rule 1).
- Note: Kaggle output download was broken (fetch.sh pagination; fixed by coordinator); job 1 outputs fetched OK.

## Job 2 plan (<=15 passes)
Inject b(text) at the option tokens themselves via one repurposed MLP neuron at layer L in {12,16,20}:
gate row = mean-activation direction (keeps SiLU ~linear, ~constant), up row w = sum_X b_X (mu_X - mu_opt)
(difference of means of MLP input per option text, mean direction projected out, std-matched to b),
down col = alpha * d, d = mean residual at model-chosen option tokens minus other options (difference of means).
Neuron replaced = smallest |down col|*|up row| at that layer. Stats from 2 hooked train passes (1024 new ex),
sweep alpha in {1,2,4} per layer on set B, pick best, T by 1-D NLL grid, write edits/o1/optprior.py + params, full dev.
Hooks only record activations inside charged lab.forward passes (no extra forward computation).

## Job 2 results (14 passes, ledger 24/50)
- w readout corr with b only 0.34 (L12), 0.35 (L16), 0.38 (L20). Gate proj mean ~9 (std ~1): SiLU stays linear.
- Sweep on set B (base 0.3164): L12 a1 .322 / a2 .313; L16 a1 .328 / a2 .309; L20 a1 .326 / a2 .332 / a4 .342 (still rising).
- Best L20 a4, T=5 -> edits/o1/optprior.py: FULL DEV acc 0.3169, NLL 1.5276 (base 0.2938 / 2.674).
- Letter bias on top: no gain offline (+-0.005) -> dropped.
- Diagnosis: one always-on linear neuron cannot single out texts (corr 0.38); E under-predicted after edit.

## Job 3 plan (13 passes)
Per-text detector neurons: for each frequent text X (>=15 options in stats): gate = gamma*(u_X - tau*ghat/c),
u_X = diff of means (X option tokens vs other option tokens), tau = midpoint of 10th pct (in) / 99th pct (out),
ghat/c used as a constant ~1 channel in place of a bias; up = ghat/c; down = alpha*b_X*d/6. Fires only on X.
Also linear neuron at larger alpha (L20 a8, L23 a4). Same sets A/B (seeded) as job 2; full dev only if > 0.347.

## Job 3: never ran
- 17 push attempts (~2h40m) all blocked by the shared 2-GPU session limit (exit 3). No passes charged.
- Decision: stop. Final edit = edits/o1/optprior.py (+ optprior_params.pt), measured on full dev in job 2.
- step3.py / injection.py (per-text thresholded detector neurons) remain untested.

## Final
- optprior: dev acc 0.3169, NLL 1.5276 (base 0.2938 / 2.674). Passes used 24/50, Kaggle jobs run 2 (job 3 never started).
- Changed weights: layer-20 MLP neuron 7 (gate row, up row, down column = 4608) + final norm weight (1536, scaled 1/5)
  = 6144 of ~1.544B (4.0e-6).
- Borderline: b(text) is a per-option-text table derived from train labels (count ratios) and is written into the up row
  (via a b-weighted difference of means). Some texts are item-specific (vocab words), so it partly acts as a per-item prior.
  Hooks recorded per-token activations inside charged lab.forward passes. T and alpha chosen by 1-D grid on train.
