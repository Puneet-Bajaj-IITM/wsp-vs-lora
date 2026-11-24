# Programmer report: qwen twin r4

Final message of the programmer agent, recovered from its transcript.

**task:** twin
**run:** r4
**edit_name:** kq_final (files `edits/r4/kq_final.py`, `kq_final_w.npy`, `kq_final_u.npy`, `kq_final_cfg.npy`)
**dev_acc:** 0.35625 (full dev, 1600 examples, 4 `lab.forward` passes through `Lab(edit="kq_final")`; bootstrap 95% CI 0.333–0.381)
**dev_nll:** 1.5034
**base_dev_acc:** 0.294 (the reference number from the assignment; I did not spend passes re-measuring base on dev. Base on my own train samples was 0.309 on 2048 examples, NLL 2.556.)
**passes_used:** 48 of 50 (44 train, 4 dev)
**weights_changed:** 172,032 — 111 key rows × 1536 in `k_proj` of layers 18, 20, 21 (32 + 53 + 26 rows = 170,496), plus the 1,536 final RMSNorm gains
**weights_fraction:** 1.11e-4 of 1,543,714,304
**kaggle_jobs:** 7 kernel versions of wsp-twin-r4 (one crashed after its capture passes, see failed)

**mechanism:**
- **What I found.** The answer position only knows its letter from layers 19–21 on (nearest-centroid decoding 0.31 at L17, 0.75 at L21, 0.90 at L23).
  - Heads L18H3/H7, L20H2/4/5/9/11 and L21H8–11 read the options' content and end-of-line tokens from the answer position; L22H7 and L23H8 then put 0.93 and 0.66 of their attention on the chosen option's letter token and copy it.
  - The reading heads use an essentially fixed query (per-item mean queries have cosine 0.97–0.99 with the global mean, ≥86% of the energy in low-frequency RoPE dims).
  - So the model's choice is a static "attractiveness" of each option: the projection of its key onto that query. It correlates 0.75–0.82 with how often the model picks the option, but only about 0.3 with the true answer frequency.
- **What I changed.** At L18/20/21, `k_proj.weight[rows] += beta * u ⊗ w`, a rank-1 term per layer with beta = 2.0.
  - `u` is a sparse key direction in low-frequency RoPE dims with mean_query·u = sqrt(128) for the option-reading heads (min-norm solution).
  - `w` is a read-out of the `k_proj` input at an option's last and end tokens: a difference of means with per-option weights 0.5·(respondent's label indicator) + 0.5·(item's train answer frequency) − 0.25·(base model probability), over 2,048 captured train examples. It is multiplied by the inverse of a ridge-regularised, label-free second moment of generic inputs, has slot-mean directions projected out, and is rescaled to the spread of the log-prior table.
  - The "minus model" share matters: the native preference is itself linear in this input, so without it the read-out mostly sharpens what the heads already prefer.
- **Temperature.** The final RMSNorm gain is multiplied by tau = 0.2, an exact temperature on the logits. It changes NLL only, not accuracy.
- **Evidence it acts as intended.** On 1,024 train examples with known base predictions, the tested key edit (beta 2, before the temperature) scored 0.351 against 0.308 for base. An attention probe on an earlier variant showed the edited heads' preferred option matching the table argmax about 50% of the time (37–39% for base), with the final prediction following them at beta 2.
- **Dev by block:** Cognitive tests 0.518, Personality 0.235, False consensus 0.330, Economic preferences 0.380, other 0.280.

**failed:**
- **Wasted passes.** Job 3 crashed after 6 capture passes: it read a `.npz` file, and the bundler ships only `.py/.md/.json/.npy/.pt`. Those 6 passes produced nothing.
- **"Answer-ness" directions** in the residual at option tokens are too weak to use (split-half 0.28–0.35 for the model's choice, chance for labels).
- **Writing the per-(item, option) log-prior table through a linear read-out** is capped by crosstalk. The read-out reproduces the table with correlation only 0.63 even on the pair means it was built from, and its stand-alone label accuracy is about 0.35, against 0.457 for the exact table on the same capture sample. Read-outs from MLP activations at L3–L17 were no better.
- **Large beta always hurts.** At beta ≥ 4 attention becomes very peaked (top option share 0.76–0.88), the letter-copy stage stops following the edited heads, predictions drift to E (up to 0.475 of predictions), and accuracy falls to 0.24–0.27.
- **Variants that did not help:** projecting the read-out off 66 nuisance directions (slot differences, prefix vectors, tail means), a uniform "is an option token" gate, and the plain label+frequency difference of means without the model term (0.305–0.309 on the first eval set, 0.320 on the second where base is 0.328).
- **Not attempted:** per-item conditional read-outs, which would reach 0.435 stand-alone with oracle item gating but need 92 gated components.
- **Overall gap.** The result (0.356) is well short of the per-item population prior (0.478).

**borderline (rule 1):**
- **Label-derived table in the weights.** The read-out `w` is built from the per-item answer-frequency table (counts over all 16,800 train labels) and from the label indicators of 2,048 captured train examples, and it is written into `k_proj`. Declared explicitly as the brief requires.
- **Matrix inverses.** `w` is multiplied by the inverse of a ridge-regularised second moment of generic activations. That matrix is label-free and no regression is solved against labels. `u` uses a pseudo-inverse on mean queries, also label-free.
- **Scalars chosen from train accuracy.** The model share (0.25) and beta (2.0) were picked from accuracies on train test passes, guided by an offline grid on train labels. tau = 0.2 was picked by a fixed rule (mean confidence equals measured accuracy on the pooled train evals).
- **One regression-like variant was run.** A PCA-whitened pseudo-inverse read-out, which is algebraically a least-squares interpolation of the table on pair means, got one forward pass in job 2 (0.3125) and was discarded. Joint ridge solves were also evaluated offline as diagnostics only. Neither is in the final edit.
- **Hooks.** I used read-only hooks during budgeted `lab.forward` passes (activations at option tokens, MLP inputs, a read-only wrapper around `scaled_dot_product_attention` for attention maps). No forward computation ran outside `lab.forward`.
- **In-process testing.** Candidate edits were applied to the loaded model's weights and restored between test passes. The final dev numbers come from a fresh model loaded through the edit file.
- **Local smoke tests** used a tiny random-initialised Qwen2-config model with a cached tokenizer; the real model was never loaded on the laptop.
- **Not touched:** test split, and other runs' folders.

Log and scripts are in `wsp_work/twin_r4/` (`LOG.md` is current); the edit is in `.../edits/r4/`; the ledger is `.../results/r4/wsp_ledger.json`.
