# sst5 r1 log

## Setup
- Base (from results/sst5/eval_base_dev.json): dev acc 0.423, NLL 3.44; neutral block acc 0.044; others ~0.5.
- Train priors: vneg .128, neg .260, neu .190, pos .272, vpos .151.
- Hypotheses: (H1) model is badly over-confident -> temperature via final-norm gain fixes NLL;
  (H2) "neutral" is almost never chosen -> a class-level (not letter-level) push toward neutral is needed,
  must act before the class->letter binding; (H3) very/non-very thresholds are miscalibrated.

## Job 1 (3 passes): base capture train[0:1536], all layers at the answer position + weight stats.
Result (train[0:1536]): acc 0.438, NLL 3.13. Confusion: neutral predicted 20/1536 times (true 265).
- Offline: temperature T~5-6 gives NLL ~1.29 (logits over letters scale exactly with final-norm gain).
- Offline upper bound for a pure class-level logit bias (neutral +4, very -3): acc 0.495.
- Layer scan at the answer position: class information appears at L15-16 (between-class variance
  0.27-0.30 of total at L16-20); letter identity appears abruptly at L21 (0.07 -> 0.29). So the
  class->letter binding happens at L21+, and L16-20 hold a class code that is not yet letter-bound.
- Class means at L16-20 lie in a 2-D plane: PC1 = polarity (bimodal), PC2 = intensity
  (very +, neutral -). A 2-D nearest-class-mean readout with priors gets 0.53 on held-out train
  (diagnostic only), so the residual knows more than the letter readout uses (0.44).
- Model's predicted class depends only gradually on PC2; neutral is predicted only at extreme -PC2.

## Job 2 (10 passes): dose-response of constant steering in the class plane
Mechanism: overwrite spare (lowest norm-product) SwiGLU neurons of layer L MLP; gate and up read the
mean residual direction (cos~0.98 at answer position, so ~constant), down writes the steering vector.
Directions: e_neu = mean(neutral) - (mean(neg)+mean(pos))/2, e_calm = mean of (non-very - very).
(Job 2 first attempt failed before any pass: .npz is not bundled; bundle limit ~1 MB shared by all tasks,
so only a few float16 vectors can be shipped per job.)
Results on train[0:512] (base 0.430, control with 4 zeroed spare neurons 0.438 -> spare neurons harmless):
- Written shift arrives at ~0.70 of the commanded norm with tiny per-example spread (0.4): mechanism works.
- But class-mean steering is a weak lever: 4-17 residual units move class logits by only 0.2-1.1
  (need ~+4 on neutral); large pushes flip 30-36% of answers without creating neutral predictions
  (combo x3: acc 0.412). Best: direction from *predicted*-neutral mean ("neuproto", shift 6.2):
  acc 0.473, neutral logit +0.64.
- Conclusion: the readout direction is not the true-class-mean direction; and residual steering
  perturbs everything downstream. Next: act on the binding itself. Letter variance jumps at L21, so
  L21(+22,23) attention binds class->letter; a constant added to q_proj.bias along
  W_q (xhat_predicted-neutral - xhat_predicted-pos/neg) biases what the answer-position query looks for,
  without touching the residual.

## Job 3 (9 passes): query-bias steering at L21-23, stronger neuproto, " neutral" embedding scale
Results (train[0:512], base 0.430):
- q_proj.bias steering at L21 / L21-23: best 0.461 (neu3+calm3 on L21-23); strong settings break the
  model (0.29-0.37). Not better than residual steering -> dropped.
- " neutral" embedding x1.5: no change at all (0.428, same predictions) -> dropped.
- neuproto dose-response is sharply nonlinear: x1 -> 26 neutral predictions (acc 0.471),
  x2.5 -> 182 (acc 0.363, neutral precision 28%), x5 -> 309 (0.277). A constant push cannot be precise.
- Offline (an5/an6): per-example dose-response simulation of a polarity-tent-gated push gives ~0.49.
- Offline (an7): the model's very/non-very call is poor. Among examples it puts on the negative side
  it is right 39%, a 1-D threshold on a "very minus non-very" class-mean direction at L18 gets 47%
  held-out; positive side 50% -> 61%. Worth ~+8 points overall if the model can be made to follow it.

## Job 4: intensity re-mapping neurons (+ gated neutral push)
Linear neuron: gate ~6 (reads mean direction), up = (intensity sensor - threshold), down = model's own
very-vs-non-very readout direction (difference of predicted-class means at L19). Side-specific version
uses a saturating polarity step built from two ReLU-like neurons. Neutral: tent(polarity) gated push.
Results (train[0:512], base 0.430):
- very-remap: gen_m0.6 0.451, gen_m1.5 0.453, side_m1.5 0.457, side_m3 0.449; tent_only 0.469;
  side_m1.5+neuconst 0.455; side_m1.5+tent 0.445.
- an8: the circuits do what was intended (realized vs intended push corr 0.998, slope 0.84-0.89), and
  the sensor is a far better "very" detector than the model's own answer (positive side top bin: 86% truly
  very, model says very 52%). BUT pushing the L19 residual along the predicted-class mean difference
  (very - non-very), even by 30+ units, does not move the model's very/non-very choice; it only leaks
  into neutral. So that direction is correlational, not the causal readout. The lever is missing, not the signal.
- Job 3 had shown that a query-space bias (q_proj.bias at L21 along soft predicted-class means) does move
  very/non-very (10% of predictions at alpha=6). Next: make that query shift sensor-driven with a rank-1
  edit of q_proj.weight at L21(,22): W_q += g (W_q d_very) (sensor - t)^T. Exact (h20 is the true input).

## Job 5 (7 passes; one push blocked first by the shared 1 MiB bundle limit -> old scripts archived in old_scripts.tgz)
Train[0:512], base 0.430. Edit file edits/sst5/r1/sst5_circuit.py is now parametrised (build(model, cfg)).
- tent via edit file: 0.469 (reproduces job 4).
- Sensor-driven rank-1 q_proj edit (no neutral push): L21 g4 0.467, L21 g10 0.471, L21+22 g4 0.453, L21+22 g10 0.469.
  an9: with g10 the model's very-rate now follows the sensor (neg side 0.11 -> 0.59 across sensor quintiles,
  pos side 0.07 -> 0.78), i.e. this lever is causal where the residual push was not.
- Combining with the neutral circuit over-produces neutral (125 predictions, 30% precise): 0.436 / 0.428.
  The two interact super-additively on neutral -> next job tunes a weaker neutral push, the gain and the threshold.

## Job 6 (7 passes): tuning on train[0:512] (base 0.430), q edit at L21 g10 unless stated
- + tent peak 1.2 (no constant): 0.488; + const 0.3 + tent 1.2: 0.467; + tent 2.0: 0.484.
- g20 (no neutral): 0.453 (worse than g10 0.471). Threshold +1 (fewer "very"): 0.473.
- Combined (tent 1.2, g10, threshold +1): 0.492 on train[0:512];
  fresh block train[512:1024]: 0.492 (base there 0.453). Both blocks: 0.492 vs base 0.440.
- Temperature from these 1024 train log-probs (exact rescaling): NLL 3.15 -> 1.24 at T=5.5.
Decision: final edit = sst5_circuit with DEFAULT {tent 1.2, g10 @L21, t+1, T=5.5}. Job 7 measures dev (base + edit).

## Job 7 (6 passes): full dev (n=1101), measured with lab.forward
- base:          acc 0.4233, NLL 3.4362
- sst5_circuit:  acc 0.4741, NLL 1.2906   (Lab(edit="sst5_circuit"), file defaults)
- Weights changed: 2,338,393 of 1,543,714,304 (0.151%): L19 MLP 3 rewritten neurons + 1 zeroed (15,359 entries),
  L21 q_proj.weight rank-1 update (2,321,498), final norm gain (1,536).
- an10: the edit under-selects option letter A (8.8% of dev predictions vs 16% base). A constant letter-A logit
  bias would only recover ~+0.5-1 point on train, and there are not enough passes left (1) to re-measure a
  changed edit on full dev, so it is NOT included. Left as an open issue.
- Passes used: 49 of 50. Kaggle kernel versions pushed: 8 (one died before any pass: .npz not bundled).
- Housekeeping: finished scripts (job1-4, an1-8, builders for jobs 2-4, wlib, circuits) are in old_scripts.tgz
  because the shared Kaggle bundle (all tasks) is capped near 1 MiB and other runs still need the headroom.
