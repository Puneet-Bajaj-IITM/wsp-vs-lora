# Programmer report: qwen sst5inv r1

Final message of the programmer agent, recovered from its transcript.

**task:** sst5inv

**run:** r1

**edit_name:** swap_steer_temp

**dev_acc:** 0.4814

**dev_nll:** 1.2827

**base_dev_acc:** 0.05

**base_dev_nll:** 9.3956

**passes_used:** 44

**kaggle_jobs:** 4

**weights_changed:** 791040

**weights_fraction:** 0.000512

**mechanism:**

Finding: the base model reads sentiment correctly and picks the true-sentiment option, so on mirrored labels it scores 0.050 on full dev (NLL 9.40). Read as plain SST-5, the same base outputs score 0.425 on dev. The knowledge is present; only the readout is mirrored.

The edit (edits/sst5inv/r1/swap_steer_temp.py plus three .npy constants) has three parts:

1. Polarity swap. The tied embedding rows of the two label tokens " negative" (8225) and " positive" (6785) are swapped. Every option line ("X. [very] negative/positive") now reads to the model as its mirror, so its normal sentiment readout lands on the mirrored label; neutral is untouched. These words occur in 10 of 8544 train reviews and 0 dev reviews. Alone (edit swap_polarity) this gives full dev 0.4432, NLL 3.25.

2. "Very" damping. The model over-picks the "very" options and almost never picks neutral. Sentiment reaches the answer position at layers 15-16 (valence difference-of-means norm 0.2 at L12, 7.3 at L16, 17.5 at L20). I computed a difference of mean layer-20 answer-position residuals, "model picked a very option" minus "model picked the plain option", from 1536 train examples. It is averaged within each predicted letter and projected off the 4-dim letter-mean subspace (SVD of 5 letter means), because about 70% of the raw direction's norm was letter identity. Minus 5x this vector is written as a rank-one update to layers[20].mlp.down_proj, keyed on the mean answer-position MLP activation restricted to its 512 largest neurons (gain at the answer position 1.00 +- 0.03, about 0.08 at another position).

3. Temperature. The final RMSNorm weight is multiplied by 0.25. This scales all logits, so the argmax is unchanged and only NLL moves.

Full dev (n=1101), all from my own lab.forward passes: base 0.0500 / 9.396; swap only 0.4432 / 3.250; final 0.4814 / 1.283. On train chunks A and B (2 x 512) the swap averages 0.4453 and the final steer 0.4707.

Weights changed: 2 embedding rows (3072) + 1536 x 512 down_proj entries (786,432) + 1536 norm weights = 791,040 of 1,543,714,304 (0.051%).

**failed:**

- Relabelling the neutral option: " neutral" row replaced by " mixed" scored 0.398 against 0.412 for the swap on the same 512 train examples; a half blend with mixed/average/mediocre/okay/indifferent scored 0.418. Neutral was picked even less.
- Relabelling " very" as an intensifier (" extremely", or a half blend of 6 intensifiers): 0.428 / 0.4375, within noise, and it also alters " very" inside reviews. Dropped.
- Steering toward the true-neutral direction: L16 had no effect (0.414-0.418); L19 x5 fell to 0.338; combined with the intensity steer 0.381. That direction is anti-aligned with valence (cos about -0.56) and drags positive reviews to negative.
- First intensity direction (not letter-balanced, from 512 examples): +4 points on the chunk it was derived from but only +0.8 on a fresh chunk, and it collapsed letter A's share from 0.19 to 0.02. Replaced by the letter-balanced version.
- Steering at L16 was too weak: the direction norm there is only 1.4.
- Letter-position offsets: offline analysis showed no gain, so never implemented.
- Ceiling: an offline class-offset sweep suggests calibration alone is worth roughly +5 points over the swap. The final edit still predicts neutral rarely (neutral recall under 10%).
- One Kaggle push was rejected (HTTP 400) because my saved arrays made the kernel script 2 MB. I moved the arrays into .npz archives the bundler skips and re-pushed; no passes were lost.

**borderline:**

- Extra activations via hooks: besides r["hidden"], I registered forward pre-hooks on mlp.down_proj (layers 12-23) during one budgeted lab.forward pass to capture the mean answer-position MLP activation. It is used as the key of the rank-one update. No forward computation ran outside lab.forward; the hooks only read activations inside a ledgered pass.
- Scalars chosen by grid against train labels: temperature 0.25 came from a 19-point 1-D NLL sweep on logged train log-probs (A+B) of the chosen variant; steering layer and scale (L20, x5) were the best of 5 screened variants by mean train accuracy. No solver, regression or gradient was used. Offline class-offset sweeps on logged log-probs were diagnostics only and are not in the weights.
- Label-derived content in the weights: the steering vector is a difference of class means over 1536 train examples, grouped by the model's own predicted option and letter (not by gold labels). No per-item or per-entity table is stored, only one 1536-d vector, one 512-entry key and two scalars.
- Selection and dev: the final variant was auto-selected inside job 4 on train only and then measured once on full dev. Dev was never used to choose anything; it was touched by 9 passes in total (base, swap only, final).
- The edit .py and its .npy constants were generated on Kaggle by wsp_work/sst5inv_r1/step4.py from final_edit_template.py and copied back unchanged; the dev numbers are from exactly those files.
- The swap also exchanges the two unembedding rows, since embeddings are tied. This is irrelevant to A-E scoring but would change free-text generation of those two words.
- The test split was never opened. I did not read the pre-existing top-level wsp_work/*.py and LOG.md or any other task's or run's folders; I only saw aggregate directory sizes when diagnosing the bundle-size failure.
