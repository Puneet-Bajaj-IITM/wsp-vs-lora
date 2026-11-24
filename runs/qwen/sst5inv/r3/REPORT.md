# Programmer report: qwen sst5inv r3

Final message of the programmer agent, recovered from its transcript.

**task:** sst5inv

**run:** r3

**edit_name:** final_steer

**dev_acc:** 0.4841

**dev_nll:** 1.2698

**base_dev_acc:** 0.05

**base_dev_nll:** 9.396

**passes_used:** 18

**kaggle_jobs:** 2

**weights_changed:** 5632

**weights_fraction:** 3.6e-06

**mechanism:**

Three closed-form pieces, in edits/sst5inv/r3/final_steer.py (needs _steerlib.py and neutral_dir_h20.npy beside it).

1. Polarity swap (3072 weights). The base model reads sentiment correctly and picks the true-sentiment option, which is why it scores 0.050 on the mirrored labels. The label words appear only in the options block, and " negative" (8225) and " positive" (6785) are single tokens. Swapping those two rows of the tied embedding matrix makes every option read as its mirror, with neutral unchanged. This alone gives dev 0.443 / NLL 3.251.

2. Neutral-threshold injection (1024 weights). After the swap the model almost never answers neutral: 8 of 229 neutral dev items correct, 14 neutral predictions in total. At the answer position, a neutral-vs-rest difference of class means carries no signal at layers 8 and 12 but has held-out AUC 0.75-0.78 at layers 16-24. I add 1.5 times the layer-20 direction as a constant to the residual stream. Qwen has no MLP or o_proj bias, so the constant goes through the v_proj biases of layers 17-20, which pass through o_proj unchanged because attention weights sum to one. Each layer's v bias is only 256-d, so the four layers are solved jointly by pseudo-inverse of the stacked o_proj maps; the reachable vector has cosine 0.86 with the target. Neutral recall on dev rises from 8/229 to 51/229.

3. Temperature (1536 weights). The final RMSNorm gain is multiplied by 0.25, which scales all logits and fixes overconfidence. Swap plus temperature alone (tau 0.20) is dev 0.442 / NLL 1.305.

Weights changed: 3072 + 1024 + 1536 = 5632. The 3.6e-6 fraction assumes about 1.54B parameters; I did not count them on the model.

**failed:**

- Scaling the " neutral" embedding row by 2 or 4: no effect (train 0.432 / 0.430 against 0.432 for swap only).
- Injecting the layer-16 direction through layers 13-16: at most 0.447 on train (alpha 3), and 0.387 at alpha 6.
- Larger layer-20 injections over-shoot: alpha 3 gives 0.451 on train, alpha 6 collapses to 0.273.
- The injection costs some extreme-class recall. On dev, mirrored "very negative" correct drops from 93 to 55 and "very positive" from 83 to 64; the confusion between plain and "very" within each polarity is untouched.
- Not refined further: alpha between 1.5 and 3 was not tried. I stopped at 18 of 50 passes because each push was waiting over an hour in the shared GPU queue and the remaining headroom looked like 1-2 points.
- Process: several pushes were rejected with HTTP 400. Rejected bundles were 1.5-5.5 MB, inflated mostly by another run's constants under edits/; a push at about 0.7 MB went through. I moved my captured hidden states to wsp_work/sst5inv_r3_cache/ to shrink my share. Twice I issued a fetch alongside a push that turned out to fail; each only re-downloaded job 1 and spent no passes.

**borderline:**

- The steering direction is a difference of class means (neutral vs rest) of answer-position hidden states on 512 labelled train examples, written into v_proj biases. It is a class-level statistic of 1536 numbers, not a per-item or per-entity table.
- Two scalars were chosen against train labels by grid: alpha (best train accuracy over 1.5 / 3 / 6, for two layer groups) and the temperature (minimum train NLL over a 0.05-step grid). This is hyperparameter selection, not a solver, but it is label-informed. Alpha was selected on the same 512 examples the direction came from.
- torch.linalg.pinv is used at apply time on the model's own o_proj weights to express the constant vector as v-bias shifts. It is a linear solve on weights, not a fit against labels.
- Dev was used for full measurement of three edits (swap_polarity, swap_temp, final_steer) and the best is reported, so there is mild selection on dev. The +4.1 point gain of final_steer over swap-only is within roughly one 95% CI half-width (about 3 points) on 1101 examples.
- No autograd, optimizers or test-split access. I read only my own run's folders, apart from listing file names and sizes under edits/ to diagnose the bundle-size failure.
