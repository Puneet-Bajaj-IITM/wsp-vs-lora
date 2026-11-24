# sst5inv r3 log

## Local analysis (free)
- Prompts identical to SST-5; label = mirrored class. Base 0.050 on dev => base model reads sentiment
  correctly and picks the true-sentiment option, which is almost never the mirrored one.
- Label words appear only in the options block (dev: 0 reviews contain "positive"/"negative"; train: 8).
- Hypothesis H1: swapping the input-embedding rows of " negative" and " positive" makes every option
  read as its mirror ("very negative" <-> "very positive", neutral unchanged), so the model's intact
  sentiment readout lands on the mirrored-label letter. Expected accuracy = base accuracy on plain SST-5.

## Job 1 (4 passes): swap edit on 512 train (captures) + full dev
- Result: tokens " negative"=8225, " positive"=6785 single tokens. swap_polarity: train(512) 0.432, **dev 0.443 / NLL 3.251**
  (base 0.050 / 9.396). H1 confirmed.
- Confusion (mirrored space): neutral almost never predicted (dev 8/229 correct, 14 predictions total);
  fine/coarse confusions inside each polarity. NLL is bad because the model is overconfident.
- Local: neutral-vs-rest diff of means at answer position: no signal at L8/L12, held-out AUC 0.75 (L16),
  0.78 (L20), 0.76 (L24). Temperature scan on train logp: NLL 2.97 -> ~1.26 at tau 0.2-0.25.

## Job 2 plan (8 train passes + 3-6 dev)
- H2: neutral under-prediction is a threshold problem; add a constant alpha*d_neutral to the residual
  (Qwen has no MLP/o_proj bias; v_proj bias passes through o_proj as a constant because attention sums to 1.
  GQA => v bias is only 256-d per layer, so spread over 4 layers and solve pinv on the weights).
- H3 (cheap control): scale the " neutral" embedding row.
- H4: temperature via final RMSNorm gain (tau picked by 1-D grid on train NLL; borderline, disclose).

## Job 2 results (14 passes; 18 used in total)
Train sample (512, same indices as job 1; swap-only baseline 0.432 / NLL 2.97):
| edit | train acc | note |
|---|---|---|
| steer h16 (v-bias L13-16) a=1.5 / 3 / 6 | 0.430 / 0.447 / 0.387 | reachable cosine 0.86 |
| steer h20 (v-bias L17-20) a=1.5 / 3 / 6 | **0.471** / 0.451 / 0.273 | reachable cosine 0.86 |
| " neutral" embedding x2 / x4 | 0.432 / 0.430 | H3 dead: no effect |
Full dev:
- swap_temp (swap + tau 0.20): 0.442 / NLL 1.305 (temperature fixes NLL only, as expected).
- **final_steer** (swap + h20 injection a=1.5 over L17-20 + tau 0.25): **0.484 / NLL 1.270**.
  Neutral recall on dev 8/229 -> 51/229; cost is some very-positive/very-negative recall.

## Decision
Stop at 18/50 passes, 2 Kaggle jobs. Remaining headroom (alpha between 1.5 and 3, fine-vs-coarse intensity
confusion) is likely 1-2 points and each job currently waits >1 h in the shared GPU queue.
Best edit: edits/sst5inv/r3/final_steer.py (needs _steerlib.py and neutral_dir_h20.npy beside it).

## Process notes
- One fetch was issued in parallel with a failed push by mistake; it only re-downloaded job 1 (no passes spent).
- Push 400 errors came from the bundle exceeding Kaggle's source size while another run had large constants
  in edits/; moved my captured hiddens to wsp_work/sst5inv_r3_cache/ and retried.
