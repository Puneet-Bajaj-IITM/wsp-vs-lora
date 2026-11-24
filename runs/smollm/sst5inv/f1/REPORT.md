# Programmer report: smollm sst5inv f1

Final message of the programmer agent, recovered from its transcript.

Final report for smollm / sst5inv / f1. The final edit is `valence_v1`, with full-dev accuracy 0.4487 and NLL 1.4341 (n = 1101), measured through `Lab(edit="valence_v1")` in job 6. I checked this against the fetched log-probs and the ledger rather than taking the log line on trust.

- **task:** sst5inv
- **run:** f1
- **model:** smollm (HuggingFaceTB/SmolLM2-1.7B-Instruct; 24 layers, d_model 2048, tied embeddings)
- **edit_name:** valence_v1
- **dev_acc:** 0.4487
- **dev_nll:** 1.4341
- **base_dev_acc:** 0.124 (reference figure; base NLL 2.307). My own base probe on train[0:512] gave 0.135.
- **passes_used:** 39 of 50; the remaining 11 are unspent.
- **weights_changed:** 22,528 (2 embedding rows, one MLP neuron in block 16 as gate row + up row + down column, 5 letter rows, the final norm gain; 2,048 each)
- **weights_fraction:** about 1.3e-5 of roughly 1.71B parameters
- **kaggle_jobs:** 6 completed runs. Two further pushes were cancelled while queued, before running or charging any pass.

**Why no further passes were spent.** The four alternate boost settings scored 0.449, 0.453, 0.457 and 0.459 on the train probe, against 0.445 for `valence_v1`. All are within 1.4 points, inside the roughly ±4-point noise of a 512-example probe and under the 2-point bar I set before the job ran. I did not build a `valence_v2` and did not select anything on dev.

**Mechanism**
1. **Polarity swap.** The embedding rows of " negative" and " positive" are exchanged. The model still judges the review's true sentiment, but the option written "negative" now reads as "positive", so its choice lands on the mirrored option. Alone (`swap_v1`) this gives full-dev accuracy 0.3833, NLL 1.5385.
2. **One-sided valence boost.** At the answer position, review valence is a single residual axis (the top principal component at layer 16, polarity AUC 0.98), and block 17 writes the answer letter. The model reads positive reviews as negative about a third of the time. One little-used MLP neuron in block 16 is rewritten to add 6 × relu(valence + 10) along that axis. On the train probe this took accuracy from 0.344 to 0.428: positive reviews went from 0.41 to 0.68 correct and negative reviews stayed at 0.70.
3. **Letter bias.** Letter A was picked only 8% of the time. A bias that equalises the mean log-probability of the five letters is added to the letter unembedding rows along the mean final hidden state. It uses no labels, since shuffled options make every letter correct equally often. On `swap_v1` the measured result matched the offline prediction exactly (0.410).
4. **Temperature.** The final norm gain is scaled by 0.5, which scales all logits and leaves the argmax unchanged. It only improves NLL.

The assembled edit scored 0.4453 on the train probe, against an offline prediction of 0.449.

**Failed**
- A valence amplifier in block 17's MLP changed nothing, because it sits after the letter is written.
- A symmetric amplifier in block 16 made the model pick "very" options more often, but equally for very-negative and negative reviews. Intensity separation in the residual is only AUC 0.74–0.78 even with 1,536 examples, and plain classes outnumber "very" classes about 2:1, so routing on it cannot pay.
- Scaling the " very" embedding by 2 did nothing.
- Replacing the " very" row (clearly, strongly, extremely, quite, really, highly) or the " neutral" row (mixed, average, okay, neither) with a near-synonym gave no gain beyond noise.
- The "very" classes (about 20% correct) and neutral (about 5%) remain unsolved.

**Borderline under rule 1**
- The valence axis is a difference of class means over 512 labelled train examples. This is allowed, but it is label-derived. No per-item or per-entity table is written into the weights.
- The boost gain and threshold were chosen from four settings by train-probe accuracy.
- The temperature of 0.5 is one scalar picked from a grid on train NLL, so it is label-informed. It affects NLL only.
- No gradients, solvers or regressions were used, and the test split was never read.

Files are in :
- edits/smollm/sst5inv/f1/valence_v1.py (plus five `valence_v1_*.npy` constants beside it)
- edits/smollm/sst5inv/f1/swap_v1.py
- wsp_work/smollm_sst5inv_f1/LOG.md
- results/smollm/sst5inv/f1/wsp_ledger.json
