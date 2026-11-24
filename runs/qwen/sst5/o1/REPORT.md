# Programmer report: qwen sst5 o1

Final message of the programmer agent, recovered from its transcript.

Nothing beats base for sst5/o1. I ran the fetch you gave me, but it brought back job 3's output again, not job 4. The 17:34 UTC run on wsp-sst5-o1 was kernel version 3, which is job 3. Job 4 has never run, and Kaggle refused another ~45 pushes over about 2.5 hours with the same "400 Client Error: Bad Request ... SaveKernel". The push also printed "Sessions run on the default GPU (NvidiaTeslaT4) instead." So the attention diagnostic, the answer-head key-bias edit and the measurement of temp_v1 are all still open. The ledger stays at 19 of 50 passes.

- task: sst5
- run: o1
- edit_name: none. The only edit measured on full dev (intent_v1) is worse than base. `temp_v1` is written but was never run through `lab.forward`.
- dev_acc: 0.4223 (base, my own full-dev passes). intent_v1 measured 0.403.
- dev_nll: 3.438 (base). intent_v1 measured 1.440. temp_v1 should give 0.422 / about 1.33. That figure is an exact rescaling of my measured base dev log-probs, not a forward pass.
- base_dev_acc: 0.423 per the reference; 0.4223 in my passes.
- passes_used: 19 (job 1: 7, job 3: 12).
- weights_changed:
  - intent_v1: 6 neurons in MLP layer 17 (gate row, up row and down column each, 27,648 values) plus the final-norm gain (1,536), so 29,184 in all.
  - temp_v1: the 1,536 final-norm values.
- weights_fraction: about 1.9e-5 of 1.54B parameters for intent_v1; about 1.0e-6 for temp_v1.
- kaggle_jobs: 2 completed (job 1, job 3). Kernel version 2 (the attention diagnostic `job2_attn.py`) was pushed during a laptop sleep and its output never came back. If it ran, it may have used one pass the ledger does not show. Job 4 (`job4_attn.py`) was never accepted.
- mechanism:
  - **Diagnosis:** the base model almost never answers "neutral": 10 of 229 neutral dev items right, and only 1.5% of its predictions are neutral. Neutral items go to negative or positive, and negative often goes to very negative. The model is also about 5x overconfident, which explains NLL 3.44.
  - **Where the information is:** assigning each example to the nearest class mean of the last-position residual gets 0.513 dev accuracy at layer 16 and 0.498 at layer 20, so the residual knows more than the letter readout uses.
  - **Answer heads:** direct logit attribution per head names L22H7, L26H1, L24H10, L23H0, L22H11 and L21H11 as the heads that write the answer letter. MLPs 22–26 amplify it.
  - **intent_v1:** the 6 lowest-weight neurons of an MLP (layer 17 or 21) were turned into a last-position-gated linear map. The gate fired cleanly, with no leak at layer 17. The up rows are class-mean directions. The down columns are "intent" directions, the mean layer-20 residual grouped by predicted class (or by true class). A logit temperature was added on the final norm.
  - **temp_v1:** divides the final norm gain by 5.75, which divides every logit by 5.75.
- failed:
  - Every steering variant cut accuracy on a held-out train chunk (base 0.420): best 0.383 at κ=1, and 0.24–0.29 at κ=2.5. Neutral predictions rose, but the letter histogram skewed and letter routing broke down. Residual additions at the last position are read partly as letter preferences. The best variant on full dev gave 0.403 / 1.440.
  - The attention diagnostic never ran, which blocks the planned key-bias edit on the answer heads. Offline, calibration alone (prior-matched class bias plus temperature) tops out at 0.452 / 1.25.
- borderline:
  - The class-mean directions (from train labels) and predicted-class intent directions were written into MLP weights. That is a per-class table derived from labels.
  - The temperatures were picked by a one-dimensional NLL grid on train outputs.
  - The variant was chosen by train-chunk accuracy among 7 candidates.
  - Read-only forward hooks and eager attention were used inside `lab.forward`.
  - I used no gradients, solvers or probes.

If you want the open work done, rerun the same push once a session frees up: `BUNDLE_WORK=sst5_o1 kaggle_runner/push_job.sh wsp-sst5-o1 "WSP_TASK=sst5 WSP_RUN=o1 python wsp_work/sst5_o1/job4_attn.py"`. It uses 3 diagnostic passes and then 3 full-dev passes on temp_v1.

Files are in ``:
- `wsp_work/sst5_o1/LOG.md`
- `wsp_work/sst5_o1/job1_base.py`
- `wsp_work/sst5_o1/job3_intent.py`
- `wsp_work/sst5_o1/job4_attn.py`
- `edits/sst5/o1/temp_v1.py`
- `results/sst5/o1/wsp_ledger.json`
