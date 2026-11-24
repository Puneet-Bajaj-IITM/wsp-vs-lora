# sst5 / o1 log

## Setup
- Data: train 8544 (vneg 1092, neg 2218, neu 1624, pos 2322, vpos 1288), dev 1101 (139/289/229/279/165).
- Letter label positions balanced (~20% each).
- Reference: base dev acc 0.423, NLL 3.436 (very overconfident -> temperature via final norm is a cheap NLL win).

## Plan
Job 1 (7 passes): base on full dev (3) + 2048 random train (4); capture hidden at layers 8/12/16/20/24/final
and per-head / per-MLP direct letter attribution (via hooks during lab.forward).
Questions: confusion matrix by class; letter-position bias; which heads write the answer letter.

## Job 1 results (7 passes; ledger 7)
- Base dev acc 0.422 / NLL 3.438; train(2048) 0.440. Confusion: neutral almost never predicted (dev 10/229 correct, class pred hist neu 1.5%).
  Mean class logp: neutral -9.7 vs others -5..-7. Letter bias mild (E 26%).
- Offline: temperature T~5 on base: NLL 1.34. Prior-matched class bias (5 numbers, from label counts) + T: dev 0.452, NLL 1.25 -> that is the
  ceiling of any pure output-calibration fix.
- Class-mean (nearest-centroid, difference of means only) readout of the last-position residual: L16 0.513 dev, L20 0.498, L24 0.474.
  => the residual knows more (esp. neutral) than the letter readout uses. Need a weight edit that routes class evidence to the right letter.
- Letter DLA (per-head, train): writers L22H7 (1.16), L26H1 (0.72), L24H10, L23H0, L22H11, L21H11; MLPs 22-26 amplify (L23 MLP 2.35).
## Job 2 plan (1 pass): eager attention on 512 train, attention of the 6 answer heads over option tokens; normed inputs at option/last positions.
- Job 2 (attention diagnostics) was never pushed (laptop sleep); dropped. Folded the needed captures into job 3.
- Offline: predicted-class ("intent") centroids at L20: model picks neutral only when the neutral component is ~2.4x
  the true-neutral centroid norm (D_neu 8.2 vs C_neu 3.4 in normalised units, cos 0.77). Hypothesis: the answer circuit
  (L22H7 etc.) needs a stronger neutral intent; route centroid evidence into intent directions before layer 22.
- A linear proxy of the model's decision from h20 agrees only 60% with the model -> cannot simulate edits offline; need passes.

## Job 3 plan (2 + 7 + 3 = 12 passes)
Edit "intent": repurpose 6 lowest-weight neurons of MLP L (17 or 21):
  gate row = last-position detector (mean x_last - mean x_other, minus lam * mean-direction, scaled so gate.x_last ~ 6);
  ups = class-mean directions W_c = C_c/|C_c|^2 in MLP-input space (difference of class means, train labels), plus a const neuron;
  downs = kappa * V_c / G where V = intent directions D (mean h20 by model-predicted class) or true-class means C (h20).
  Net effect at the last position: dh = kappa * sum_c ((x - mu).W_c) V_c.
Screen 7 variants on held-out train chunk idx[1024:1536]; temperature by NLL grid on that chunk; best -> full dev.

## Job 3 results (12 passes; ledger 19)
- Gate worked (last-position detector: z_last 6.0+-0.2, other positions -7, 0% leak at L17; 1.6% at L21).
- ALL intent-steering variants HURT: screen-chunk base 0.420 -> L17D k1 0.383 (best), L21D k1 0.369, L21C k1 0.375; kappa 2.5 -> 0.24-0.29.
  Neutral predictions rose (6 -> 36) but letter routing degraded and letter histogram distorted (A or E inflated).
  Full dev of best (L17D k1, T=6.25): acc 0.403, NLL 1.440. => FAILED. Residual additions at the last position are read
  partly as letter preferences; the decision is not a clean "intent vector" at the last position.
- Note: kaggle version 2 (job2_attn) may have been pushed during the sleep but its output/ledger never came back; ledger is 19.
## Job 4 plan (3 passes): eager attention capture for L21-26 heads (mass on option letter/dot/content/newline),
q at last pos, k/v at option tokens, x-hat L22; weights for an offline first-order attention simulator.
Plus screen: ' neutral' embedding x1.5 / x2.
- Job 4 push failed repeatedly (~10 attempts over ~30 min): "400 Client Error: Bad Request ... SaveKernel" (not the usual
  2-session message; likely account-level GPU quota). No further passes possible. Ledger stays at 19.
- Fallback edit prepared: edits/sst5/o1/temp_v1.py (final-norm gain / 5.75; T chosen by NLL on 2048 base train outputs).
  NOT measured with lab.forward. Exact offline transform of my measured base dev logp gives acc 0.422 (unchanged), NLL 1.33.
## Status: stopped. Best lab.forward-measured result is the base model (0.422 / 3.438); intent_v1 measured 0.403 / 1.440 (worse acc).
- Resumed after coordinator note. Fetch returned job 3's output again (the 17:34 run was job 3 = kernel v3, not job 4).
  Job 4 pushes then failed ~45 more times over ~2.5 h with the same 400 SaveKernel error ("Sessions run on the default
  GPU (NvidiaTeslaT4) instead" also printed). Ledger still 19. Stopping again; temp_v1 still unmeasured.
