# Programmer report: qwen twin o1

Final message of the programmer agent, recovered from its transcript.

The best edit is `optprior`: full dev (1600 examples, `lab.forward`) gives acc 0.3169, NLL 1.5276, against the base's 0.2938 and 2.674. Most of the NLL gain comes from the temperature rescale. The accuracy gain (+2.3 points) is small next to the 0.374 offline ceiling.

- task: twin
- run: o1
- edit_name: optprior (`edits/o1/optprior.py` plus `edits/o1/optprior_params.pt`)
- dev_acc: 0.3169
- dev_nll: 1.5276
- base_dev_acc: 0.2938 (base NLL 2.674, measured on full dev in my own job 1)
- passes_used: 24 of 50. Job 1 used 10 (4 base full-dev passes, 6 train passes with hidden states). Job 2 used 14 (2 train passes recording option-token statistics, 8 sweep passes, 4 full-dev passes of the edit).
- weights_changed: 6144. That is layer-20 MLP neuron 7 (gate_proj row, up_proj row and down_proj column, 3 × 1536) plus `model.norm.weight` (1536 values, scaled by 1/5).
- weights_fraction: about 4.0e-6 (6144 of about 1.544B)
- kaggle_jobs: 2 ran to completion. A third job was pushed 17 times over about 2h40m and was always blocked by the shared 2-GPU limit, so it never started and charged no passes.
- mechanism:
  - **Finding:** the base model misjudges option content. It over-picks extremes ("Strongly oppose" p 0.22 vs true rate 0.105, "Often", "Always"). It under-picks moderate options ("Rarely" 0.155 vs 0.385, "Somewhat support", "Agree a little") and correct vocabulary answers. Personality accuracy is 0.19, below chance.
  - **Offsets:** a per-text logit offset b(text) = log((chosen + 4)/(model probability mass + 4)), from train counts. Applied offline to the logits, it would give dev 0.4125. Restricted to the 123 texts shown at least 30 times, excluding single-letter puzzle options, it would give 0.374.
  - **Edit:** I repurposed the least-used neuron at layer 20.
    - Gate: the mean-activation direction, so SiLU stays near-linear and nearly constant.
    - Up row: a b-weighted difference of means of the MLP input per option text, with the mean direction projected out and scaled to b's standard deviation. Its correlation with b is 0.38.
    - Down column: 4 × the difference of means between residuals at the model-chosen option's tokens and the other options' tokens.
    - So at option-text tokens the neuron pushes high-b options toward "chosen". `model.norm.weight` is scaled by 1/5 as a temperature.
  - **Selection:** the layer (12/16/20) and α (1/2/4) were chosen on 512 train examples (base 0.316, best 0.342). T was chosen by a 1-D NLL grid on train.
- failed:
  - Temperature plus a per-letter bias alone reached only 0.305 / 1.532. A letter bias on top of the edit added nothing.
  - Difference-of-means readouts at the answer position for the five letter classes (layers 16 to final) added no accuracy. The logit lens was no better than the model's output.
  - A plain linear readout of the offsets from the answer-position residual was weak (R² under 0.1). I rejected a whitened-PCA version that reached about 0.345, because it amounts to least squares.
  - The single linear neuron at layers 12 and 16 got worse beyond α=1. The always-on neuron cannot single out individual texts, which is why the gain is small.
  - Job 3 never ran, so per-text threshold-detector neurons and larger α at layers 20/23 are untested (`wsp_work/twin_o1/step3.py`, `injection.py`).
  - Kaggle downloads failed at first because of the fetch.sh pagination bug, which the coordinator fixed.
- borderline:
  - b(text) is a per-option-text table derived from train labels by count ratios. It is folded into the weights as the b-weighted up row. Some texts are vocabulary words unique to one item, so the table partly works as a per-item prior.
  - Per-token activations were recorded with PyTorch forward hooks during charged `lab.forward` passes; no forward computation ran outside the ledger.
  - The scale factors (α, T, the std matching) were picked by 1-D grid search or simple ratios on train data, not by any solver.

Everything is in `wsp_work/twin_o1/`; `LOG.md` there has the full history, and the scripts are `step1.py`, `step2.py` and `step3.py`. The downloaded arrays are in `wsp_work/twin_o1_cache/`, outside the bundle.
