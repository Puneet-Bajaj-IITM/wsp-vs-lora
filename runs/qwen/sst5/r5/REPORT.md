# Programmer report: qwen sst5 r5

Final message of the programmer agent, recovered from its transcript.

FINAL REPORT

- **task:** sst5
- **run:** r5
- **edit_name:** `steer_plain_s3` (`edits/sst5/r5/steer_plain_s3.py`, with `steer_plain_s3_centroids.npy` beside it)
- **dev_acc:** 0.4596 (full dev, 1101 items, 3 `lab.forward` passes; I recomputed it from the fetched `dev_final_logp.npy`)
- **dev_nll:** 1.3007
- **base_dev_acc:** 0.423 (reference from the assignment, NLL 3.436; I did not spend passes re-measuring base on dev)
- **passes_used:** 44 of 50 (41 train, 3 dev)
- **weights_changed:** 2,304 (3 × 256 `v_proj.bias` entries at layers 17, 19, 21, plus 1,536 final `norm.weight` gains)
- **weights_fraction:** 2,304 / 1,543,714,304 ≈ 1.5e-6
- **kaggle_jobs:** 5 that ran (base capture, 17 steering trials, 12 multi-layer trials, 8 asymmetric trials, final dev). Three queued kernels never started and were cancelled outside my control; two of them (job 4 and the final dev job) were re-pushed once by the coordinator. The ledger shows no double charges.

**mechanism**
- **What I found:** the base model almost never answers "neutral" (1.5% of train predictions against 18% true) and over-uses "very negative" (22.7% against 13.4%) and "very positive" (17.9% against 14.9%). Sentiment reaches the answer position abruptly at layers 15–16, where a nearest-centroid readout scores about 0.49–0.50 against the model's 0.435.
- **Ceiling:** an offline class-level bias on the base log-probs tops out near 0.50 on train, mostly from penalising the "very" classes.
- **What I changed:** attention weights sum to one, so a value bias adds the same vector to the residual stream at every position. At layers 17, 19 and 21 I add the reachable projection (256 of 1,536 dimensions, cosine 0.44–0.51) of the vector from the "very" centroid midpoint to the plain midpoint, rescaled so its component along the target is full length. Centroids come from 2,048 train items at layers 16, 18 and 20.
- **Sign check:** steering the opposite way raised "very" predictions and dropped accuracy to 0.402.
- **Temperature:** the final RMSNorm gain is multiplied by 0.32. This leaves accuracy unchanged and accounts for most of the NLL drop from 3.44 to 1.30.
- **Train evidence, three disjoint 512-item slices (base → steered):** 0.428 → 0.457, 0.453 → 0.496, 0.436 → 0.457.
- **Where it falls short:** on dev the edit still predicts neutral only 35 times against 229 true, and "negative" 408 times against 289. The gain comes only from fewer "very" errors, about 3.7 points over the base reference and well below the 0.50 ceiling.

**failed**
- Adding layer 15, or a greedy three-layer (15–16–17) realisation of the layer-16 vector, dragged positive toward negative (0.41–0.46 against 0.453 base).
- Stronger steering (1.5× to 4×) and six-layer versions were worse; they distort the letter prior.
- Single-layer steering at 21 or 23 did nothing useful.
- A "toward neutral" term, and asymmetric versions (more push on very negative, less on very positive), all scored below the symmetric one (0.422–0.438 against 0.457 on the same slice).
- Scaling the " very" token embedding by 0.6 had no effect.
- A label-free letter-prior correction looked like +2 to +6 points on the first slice and reversed on the next two, so I dropped it.

**borderline (rule 1)**
- Class centroids (per-class means of answer-position hidden states) come from train labels. They are written into the weights only as three difference-of-means vectors; no per-item table.
- `torch.linalg.pinv` is applied to a weight matrix to project the target vector onto the reachable subspace. This is closed-form weight arithmetic, not a fit against labels.
- The temperature 0.32 was picked by a one-dimensional scan of train NLL on steered log-probs, a single scalar chosen against labels.
- Steering strength and layer set were chosen by comparing train accuracy across 37 trial passes. That is selection among hand-derived configurations, with no optimiser.
- Independence: on the third resume the coordinator's message said: "an independent run has already shown that a constant class-prior shift via one MLP neuron gets roughly 0.48 test. Pursue your own mechanism and report whatever it reaches."
  - Designed before that message: the diagnosis, the class-bias ceiling, the value-bias steering mechanism, the centroid target vector, the layer range 17–23, and the temperature plan. See `LOG.md`, sections "Hypotheses before any pass", "Job 1" and "Job 2 (step2.py)"; the job 2 script was written and pushed before the first pause.
  - Decided after it: the multi-layer choice of 17/19/21 and strength 1 (from job 2's own results), and the dropped letter correction and asymmetric variants. I did not adopt the MLP-neuron idea or open any other run's files.
- Infra deviations: I used `BUNDLE_EXCLUDE="edits/facts/*/*"` to get under Kaggle's 1 MB source limit (file sizes only, no contents read). The shell auto-backgrounded several long push and fetch calls, and I stopped one hung push with TaskStop before re-pushing.

Files are in ``:
- `edits/sst5/r5/steer_plain_s3.py`
- `edits/sst5/r5/steer_plain_s3_centroids.npy`
- `wsp_work/sst5_r5/LOG.md`
- `results/sst5/r5/wsp_ledger.json`
