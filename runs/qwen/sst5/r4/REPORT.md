# Programmer report: qwen sst5 r4

Final message of the programmer agent, recovered from its transcript.

FINAL REPORT — sst5 r4

- task: sst5
- run: r4
- edit_name: prior_n100 (edits/sst5/r4/prior_n100.py, with prior_n100_m.npy and prior_n100_vec.npy)
- dev_acc: 0.4614 (full dev, 1101 examples, measured through Lab(edit="prior_n100"))
- dev_nll: 1.297
- base_dev_acc: 0.423 (reference figure from the assignment; I did not spend passes re-measuring base on dev). Base NLL reference 3.436.
- passes_used: 47 of 50
- weights_changed: 6144
- weights_fraction: 3.98e-6 (6144 of 1,543,714,304)
- kaggle_jobs: 4 completed (the coordinator re-pushed job 1 once after the queue stall)

**Mechanism**
- Base Qwen almost never answers "neutral": 1.5% of its predictions on train[0:3072] against 19% of labels, neutral recall 4.5%. It also over-uses "very negative" relative to "negative".
- An offline check showed a per-sentiment-class additive bias would lift held-out train accuracy from about 0.43 to about 0.49, while a per-letter bias gives only 0.445. So the defect is a sentiment-class prior, not a letter bias.
- The normed residual at the answer position is nearly constant (cosine 0.999 with its mean). I use that as a stand-in for a bias input, since the MLPs have no biases.
- The edit overwrites one neuron of MLP 19, the one with the smallest product of gate, up and down weight norms. It sets gate = 6·m, up = m, down = vec/silu(6), where m = mean(x)/|mean(x)|². This adds the constant vector vec to the residual at the answer position; measured shifts matched the intended vector with cosine 0.99–1.0.
- vec = −1.5σ along the "very negative" direction plus +1.0σ along the "neutral" direction. Each direction is the mean layer-19 hidden state of train examples the base model assigns to that class, minus the overall mean; σ is the projection standard deviation.
- The final RMSNorm weight is divided by 4 (a temperature). This changes NLL only, not the argmax.
- On fresh train[3072:4096] the edit scored 0.494 accuracy, NLL 1.25. Dev is about 3 points lower.

**Failed**
- Steering "neutral" along the true-class mean direction at layers 15/17/19 (1–5σ): the model still did not say neutral, and it hurt at layers 17 and 19. Only the predicted-class direction moves the neutral share.
- Steering at layers 21 and 23: too late, no effect on neutral and accuracy dropped.
- Larger neutral pushes (+1.75σ, +2.5σ, +4σ) raise neutral share to 10–20% but at poor precision; accuracy stays flat or falls. Adjusting the negative, positive and very-positive weights by ±1.5σ was worse in every case.
- Writing the class prior into the five letter unembedding rows via linear "which letter is class c" readouts: too noisy in offline simulation, and it needs row changes 2–3× the row norm on tied embeddings. Not run.
- Thresholded indicator neurons in late MLPs: readouts fitted on one block's output do not transfer to the next, and the true MLP input is not observable through the harness. Not run.
- An earlier two-layer variant (layers 15 and 19, true-class directions) reached dev 0.4587, NLL 2.574; the final edit supersedes it.
- Overall gain is modest (+3.8 points on dev). A prior shift tops out near the 0.49 offline bound, and dev neutral recall is still only about 0.11.

**Borderline under rule 1**
- The steering vector is built from class-conditional means of hidden states. The classes are the base model's own predicted sentiment on train (which needs the option order but not the gold label); an earlier, discarded variant used gold-label class means. No per-item table is written into the weights.
- Class weights, layer and temperature were chosen by a small grid and coordinate search on train forward-pass accuracy/NLL: about 25 configurations, each one 512-example pass. No gradients or solvers.
- The winner among three final candidates was picked on train only; dev was run once, for the winner. The earlier variant also had one dev run, so dev was seen twice in total.
- Temperature 4 was chosen offline by rescaling saved train log-probs.
- Offline, I multiplied captured final hidden states by the letter unembedding rows to simulate the rejected unembedding edit. This is arithmetic on forward-pass outputs, not an extra model run.
- PCA-whitened nearest-class-mean probes were used for diagnosis only; nothing from them is in the edit.

**Other notes**
- Two unused sibling candidates, prior_n175 and prior_n250, are also in edits/sst5/r4/. They were not run on dev.
- To get under the bundle limit I moved small arrays out of WORK into wsp_work/sst5_r4_cache/small/.
- Earlier in the run I listed file names and sizes in my bundle, which includes other runs' edits/ paths, to find an oversized file. I did not open their contents.
- Log: wsp_work/sst5_r4/LOG.md. Results: wsp_work/sst5_r4/s2_results.json, s3_results.json, s4_results.json.
