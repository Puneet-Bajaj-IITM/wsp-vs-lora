# sst5inv o1 log

## Hypothesis 1 (pre-pass analysis)
Base Qwen scores 0.05 on the inverted task => it reads sentiment correctly and picks the option whose
text matches the true sentiment. The mirror map is exactly the token swap ' positive' <-> ' negative'
in the option text (very X stays very, neutral unchanged). Swapping these two input-embedding rows
should make the model "see" mirrored option labels, so its normal readout points to the inverted
answer. Reviews containing these words: train 13/2, dev 3/0 -- negligible side effects.
Edit: edits/sst5inv/o1/swap_pn.py (2 rows x 1536 = 3072 weights).

## Job 1: full dev (3 passes) + 512 train pass with swap_pn
Result (passes 1-4): swap_pn full dev acc 0.443, NLL 3.251 (base 0.050 / 9.396). Train-512 acc 0.479.
Tokens: ' positive'=6785, ' negative'=8225 single tokens, used in options as expected.
Content-space confusion (dev): neutral almost never chosen (1.3% of preds vs 21% targets);
"very" options over-chosen (e.g. true-negative reviews -> model reads "very negative").
Diagnostics on saved logp (no passes): constant content bias of -1 on both "very" options would give
~+4 pts (train 0.479->0.521, dev 0.442->0.464); a neutral bias alone adds little (neutral logp is not
discriminative). Letter-softmax temperature 0.25 would cut NLL 3.25->~1.34 at equal accuracy
(exact, implementable by scaling model.norm.weight since lm_head is tied and logits are linear in it).

## Job 2 plan: 7 train passes (same 512 idx), sweeping ' very' row scale/mix and ' neutral' row mix
Result job 2 (passes 5-11, train-512, swap_pn baseline 0.479):
very x0.7 0.484, x0.4 0.482 (norm scaling ~no effect, RMSNorm); very->fairly mix0.5 0.471, full 0.480
(full replacement kills "very" picks: overshoot); very->somewhat 0.479; neutral->mixed 0.5 0.482, full 0.482.
Neutral stays dead (<1% picks) in all. Content confusion shows the model extremizes: true-neutral is read
as "negative", true-negative as "very negative", true-positive as "very positive".
Note: bundle must stay small (2.8MB bundle got HTTP 400); moved hidden-state array to sst5inv_o1_cache.

## Job 3 plan: mix ' neutral' toward the negative-read vector (0.25/0.5/0.75, and centered pos+neg),
very->fairly 0.75, a combo; auto-pick best by train acc and measure on full dev (9 passes).
Result job 3 (passes 12-20, train-512): N_neg0.25 0.480 (neutral still dead); N_neg0.5 0.473 (neutral
unlocked: 35% neutral recall, but steals true-negatives); N_neg0.75 0.467; N_pn0.5 0.457;
V_fairly0.75 0.488; V_fairly0.75+N_neg0.5 0.418. Auto-picked V_fairly0.75 on full dev: 0.441 / NLL 3.640
-> no better than swap_pn (0.443 / 3.251). The 'very' edits do not transfer; dropped.
Neutral mixing has a sharp transition between 0.25 and 0.5.

## Job 4 plan: neutral mix 0.35/0.4/0.45 (3 train passes), pick best mix by train acc among
{0, .25, .35, .4, .45, .5}, choose logit temperature (final norm scale) on that train logp, write the
final edit, and measure full dev for it and for swap+temperature only (6 dev passes).
Result job 4 (passes 21-29): train-512 acc by neutral mix: 0 .479, .25 .480, .35 .484, .40 .492, .45 .498,
.50 .473 -> best 0.45. Temperature grid on its train logp -> logit scale 0.2 (same for swap-only).
Full dev: final_o1 (swap + neutral mix 0.45 + norm x0.2) acc 0.4478, NLL 1.2796.
          swap_temp (swap + norm x0.2) acc 0.4423, NLL 1.3045.
final_o1 dev content confusion: neutral now 8% of preds (39/229 neutral recall) vs ~1% before.

## Decision: stop. Best edit = edits/sst5inv/o1/final_o1.py. 29/50 passes, 4 Kaggle jobs.
Remaining errors are the model's own SST-5 granularity errors (adjacent-class / very-vs-plain confusions),
which the tested closed-form single-row edits did not fix in a way that held up on dev (very->fairly
looked good on train, did not transfer). Further gains would need more per-variant passes than the noise
of a 512-example comparison justifies (+-2 pts), with long Kaggle queues.
Weights changed: 3 embedding rows (' positive', ' negative', ' neutral') x 1536 + final norm 1536 = 6144.
Borderline: NEUTRAL_MIX (6 values) and LOGIT_SCALE (temperature grid) were chosen by grid search on
train accuracy / NLL -- 2 scalar hyperparameters picked against labels; no per-item table in weights.
