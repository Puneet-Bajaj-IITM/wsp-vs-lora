# Programmer report: qwen sst5 r1

Final message of the programmer agent, recovered from its transcript.

**task:** sst5

**run:** r1

**edit_name:** sst5_circuit

**dev_acc:** 0.4741

**dev_nll:** 1.2906

**base_dev_acc:** 0.4233

**base_dev_nll:** 3.4362

**passes_used:** 49

**kaggle_jobs:** 8

**weights_changed:** 2338393

**weights_fraction:** 0.001515

**mechanism:**

Full dev (n=1101), measured with my own lab.forward passes: base 0.4233 / NLL 3.4362, edit 0.4741 / NLL 1.2906. Paired: 143 items fixed, 87 broken, bootstrap 95% CI of the gain +2.5 to +7.8 points. Temperature alone would give base NLL 1.333 (computed offline), so most of the NLL gain is calibration.

What I found (answer-position residuals, 1536 train examples, all layers):
- The sentiment class is linearly present from L15-16 (between-class variance about 0.27-0.30 of total at L16-20). Letter identity appears abruptly at L21 (0.07 to 0.29), so the class-to-letter binding happens at L21+.
- Class means at L16-20 lie in a 2-D plane: polarity (bimodal) and intensity (very high, neutral low).
- The base model almost never answers neutral (20 of 1536) and makes poor very/non-very calls. A 1-D threshold on a "very minus non-very" class-mean direction is much better than the model's own call (0.75 vs 0.60 at L20, threshold fitted in-sample).
- Pushing the L19 residual along the predicted-class mean difference for very/non-very, even by 30+ units, does not move that decision. Shifting the L21 attention query along the same kind of direction does.

What I changed (edits/sst5/r1/sst5_circuit.py, closed form):
1. Neutral circuit, L19 MLP. The three lowest-norm SwiGLU neurons are rewritten and a fourth is zeroed. They implement a tent over a polarity sensor (three ReLU-like units), multiplied by a term that grows as the intensity sensor falls. They write the model's own neutral readout direction (mean residual of predicted-neutral minus predicted-pos/neg), peak dose 1.2.
2. Intensity re-mapping, L21 q_proj.weight. Rank-1 update W_q += (10/sd) * (W_q d_very) * (sensor - t)^T. Here d_very is the very-minus-non-very direction of soft predicted-class means in normed space, and the sensor is the centred class-mean intensity direction at h20. After the edit the model's very-rate tracks the sensor (0.11 to 0.59 across sensor quintiles on the negative side, 0.07 to 0.78 on the positive side).
3. Temperature. The final RMSNorm gain is divided by 5.5.

Weights changed: 2,338,393 of 1,543,714,304 (0.151%) = 15,359 entries in the L19 MLP, 2,321,498 in L21 q_proj.weight, 1,536 in the final norm.

On train the final config scored 0.492 on both train[0:512] (base 0.430, the block used for tuning) and a fresh block train[512:1024] (base 0.453).

**failed:**

- Constant residual steering along true-class mean directions (neutral, "calm") at L17/19/20: weak lever, 4-17 residual units moved class logits by only 0.2-1.1; large pushes flipped 30-36% of answers and lowered accuracy (0.412).
- Constant push along the predicted-neutral direction: sharply nonlinear and indiscriminate. Train[0:512] accuracy 0.471 at x1, 0.363 at x2.5 (182 neutral predictions, 28% precise), 0.277 at x5.
- Constant q_proj.bias steering at L21 / L21-23: best 0.461; strong settings broke the model (0.29-0.37).
- Scaling the " neutral" token embedding by 1.5: no effect at all.
- Sensor-driven very/non-very push written into the L19 residual (generic and polarity-gated neurons): the circuit delivered the intended push (correlation 0.998 with the design) but the model's very/non-very choice did not move; 0.449-0.457.
- Extending the query edit to L21+22 (0.453-0.469) or doubling the gain to 20 (0.453): worse than L21 at gain 10 (0.471).
- Stacking the full-strength neutral circuit on the query edit: over-produced neutral (125 predictions, 30% precise), 0.428-0.436. Only a weak tent (peak 1.2, no constant) helped.
- Known open defect: the final edit under-selects option letter A (8.8% of dev predictions vs 16% for base; accuracy when A is correct 0.24 vs 0.33). An offline letter-A logit bias recovered only about +0.5 to +1 point on train, and with 1 pass left I could not re-measure a changed edit on full dev, so it is not fixed.
- Still unsolved: neutral recall is only about 0.15-0.24 on dev, and very-negative recall fell (0.55 to about 0.36) in exchange for negative (0.55 to about 0.73).
- Operational: the first job-2 push died before any pass because .npz files are not bundled. The Kaggle bundle shared by all tasks is capped near 1 MiB, so I store vectors as int8/float16 and archived my finished scripts in wsp_work/sst5_r1/old_scripts.tgz to stay under it.

**borderline:**

No autograd, optimizer, regression or classifier solver was used. Items to judge under rule 1:

1. Label-derived statistics are written into the weights. These are class-level direction vectors, not per-item or per-entity tables: true-class mean differences of answer-position residuals (polarity and intensity sensors, the very sensor), their SVD plane, and predicted-class / soft predicted-class means (weights = the model's own softened probabilities, T=4) for the write directions. All come from 1536 train examples.
2. Scalar thresholds were chosen by 1-D scans against train labels: the very-sensor threshold (accuracy-maximising percentile scan), later shifted by +1 because that scored better on train[0:512].
3. Tent parameters (center -1, width 3, intensity slope 0.3) were picked by a small grid over a per-example dose-response simulation on train[0:512] labels. Tent peak, query gain and layer set were picked among about 10 hand-specified variants by train accuracy on the same 512 examples. This is hyperparameter selection by forward passes, not a solver, but it is fitting of a few scalars to labels.
4. Temperature 5.5 was chosen by a grid on train NLL, using exact offline rescaling of letter log-probs.
5. I used nearest-class-mean classification offline only as a diagnostic of how much information the residual holds; it is not in the weights.
6. Dev was used only for the final base and edit measurements (6 passes); no selection was done on dev. I read results/sst5/eval_base_dev.json and no test file.
7. To fit the shared bundle limit I tar-archived my finished scripts inside WORK instead of leaving them as .py files; nothing was deleted.
