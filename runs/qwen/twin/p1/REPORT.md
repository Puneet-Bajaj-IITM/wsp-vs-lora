# Programmer report: qwen twin p1

Final message of the programmer agent, recovered from its transcript.

FINAL REPORT — twin / p1 (preserve run)

The final edit `style_gate_v3` raises full-dev accuracy from 0.294 to 0.350 and keeps the full preserve set at 0.4233, above the 40.0% criterion. I recomputed the dev accuracy and NLL from the fetched log-probabilities and they match the job log. Dev NLL got worse (2.920 against 2.674 for base).

| Field | Value |
|---|---|
| task | twin |
| run | p1 |
| edit_name | `style_gate_v3` (`edits/p1/style_gate_v3.py` plus `edits/p1/sel_v3f_L21.pt`, 13 KB) |
| dev_acc | 0.350 (1,600 examples, 95% CI 0.326–0.374) |
| dev_nll | 2.920 |
| base_dev_acc | 0.294 (NLL 2.674), your reference; I did not spend passes re-measuring base on dev |
| passes_used | 27 of 50 |
| weights_changed | 785,389 |
| weights_fraction | 0.000509 (785,389 of 1,543,714,304) |
| kaggle_jobs | 5 completed runs (jobs 1–5); several other pushes sat queued and were cancelled by you or the user without running |
| preserve_acc | 0.4233 (full 1,101 examples; base about 0.423) |

Dev accuracy by block for the final edit:

| Block | n | Accuracy |
|---|---|---|
| Cognitive tests | 537 | 0.471 |
| Personality | 616 | 0.282 |
| False consensus | 194 | 0.320 |
| Economic preferences | 121 | 0.273 |
| Other | 132 | 0.288 |

I have no per-block base numbers on dev. On train samples the base scored about 0.19 on Personality, 0.18–0.34 on False consensus and 0.42–0.54 on Cognitive tests.

**mechanism**

What I found:
- **Where the letter is chosen.** Layer-22 head 7 puts 96% of its answer-position attention on the option letter tokens and copies the letter. Letter tokens precede their option text, so the choice is made earlier, by "selector" heads that read the option content. Layer-21 heads 8–11 are the main ones: heads 9 and 11 point at the finally predicted option 73–74% of the time.
- **What the model gets wrong.** On agree and support scales it over-picks the extreme option. "Agree strongly" was 78 of 179 Personality predictions on the first train sample, where respondents chose it about 12% of the time; people mostly pick moderate options.
- **What the keys already hold.** At layer 21, the keys of the last content token of each option separate "the option a respondent chose" from "the options they did not" along a single direction per answer style. On held-out train data, picking the option that scores highest along that direction beat the model for the agree scale (0.29 against 0.23), support scale (0.34 against 0.18) and money split (0.40 against 0.10).

What I changed:
- **The update.** For layer-21 heads 8–11, I added to their `q_proj` weight and bias rows a term of the form strength × gate(x) × direction, summed over three answer styles (agree scale, support scale, money split). The effect is that, at the answer position of a twin prompt of that style, these heads attend more to the kind of option people usually pick.
- **The direction.** Mean key of chosen options minus mean key of unchosen options, taken from 2,048 train examples, with the components that distinguish option slots A–E projected out so no letter position is favoured.
- **The gate.** A linear read-out of the query-side input that is about 1 at the answer position of that style and about 0 elsewhere. On held-out train data it read 0.99–1.03 on its own style and 0.00 ± 0.1 on other twin prompts.
- **Strength.** Chosen on a held-out 512-example train sample, not on dev: base 0.268, r2 0.346, r4 0.352, r8 0.291, r16 0.266. I used r4. The vocabulary style is set to strength 0 because it showed no gain at any strength.

How the edit avoids interfering with other computation: it changes only the query rows of four heads in one layer; the gates are built to be zero at the mean preserve-answer activation, and the top 350 principal directions of preserve-answer and random-position activations are projected out of them, so on other prompts the added query term is close to zero (about 0.02–0.03 per gate on the activations used to build it).

**failed**
- **Per-item tables.** With about 5 examples per item the key directions and gates were too noisy (held-out score 0.30–0.39 against the item prior's 0.46). Dropped.
- **Two-scale edit with plain gates (v1).** On the held-out sample it reached 0.309 against a base of 0.268, but at higher strength it damaged Cognitive items (0.42 to 0.32) because the gates fired on other twin prompts. Editing more heads (layers 18 and 20 as well) did not help.
- **38 option-set groups with log-frequency-weighted directions (v2).** Worse than base on the held-out sample (0.22–0.26), skewed the letter choices heavily (B up to 54%), and cut preserve on 512 examples from 0.424 to 0.324. Tables from about 16 examples per group carried letter-slot noise, and 38 gates leaked onto other prompts.
- **Vocabulary items.** The key direction predicted the label offline (0.58 against 0.53), but moving attention did not improve accuracy at any strength.
- **Strengths r8 and above.** These hurt the agree scale, the money split and the "other" items.
- **NLL.** The edit does nothing for the model's overconfidence.
- **Untouched blocks.** Cognitive tests and the letter-named cognitive items are unchanged. The edit captures only part of the item prior (0.478 on dev).

**borderline (rule 1 and disclosures)**
- **Forward hooks on charged passes.** In jobs 1 and 2 (5 passes: 4 train, 1 preserve) I attached read-only hooks during `lab.forward` to record answer-position attention, option-token keys and the `q_proj` input, including at random other positions. There were no extra model calls, no activation was modified, and no dev or test data was involved. You ruled this allowed.
- **Pseudo-inverse dual-basis gate.** The gate directions are the dual basis of six answer-style mean activations (four active styles plus "letter-named cognitive" and "other"), measured from the mean preserve-answer activation. It is a closed-form solve over label-free activation means. I kept it over the plain mean-difference gate because the plain gate fired about ±0.3 on other twin prompts and caused the Cognitive damage in v1; the dual gate reads about 0 there. A plain-gate version of the final design was never run on the model.
- **Label-derived directions.** Train labels enter the weights only through mean-difference key directions (chosen minus unchosen). Three are active. A fourth (vocabulary) is stored in the table file with strength 0, so it is not written into the weights. There is no per-item or per-entity table in the final edit.
- **PCA.** Principal directions of preserve-answer and random-position activations are projected out of the gates, and the slot-separating directions are projected out of the key directions (SVD/QR on means). These use no labels.
- **Preserve data used in construction.** The gates are built from 512 preserve answer activations (reference point and nuisance directions). The full-preserve score therefore includes examples the edit was shaped to ignore; the 589 not used scored in line with the rest. The held-out general tests are the cleaner check.
- **Strength selection.** Strength and the decision to zero the vocabulary style were chosen on a 512-example train sample that is also part of the data the final tables were rebuilt on. Dev was used only for the final measurement.
- **Quantisation.** Gate vectors are stored as int8 with a per-row scale to stay under the bundle size limit.
- **Process note.** I checked the status of other agents' kernels when the queue was stuck, without touching them, and for some pushes excluded the unrelated `edits/facts` folder from my own bundle to get under the size limit. The final job was pushed by you (kernel v7) after my v6 was cancelled while queued.

**Why 785k weights when the design is small**

The stored tables are tiny (4 gate vectors of 1,536 and 4 directions of 128, 13 KB). The count is large because each of the 4 heads gets a rank-3 update, and a low-rank update written into a dense matrix touches every entry: 4 heads × 128 rows × 1,536 columns = 786,432 weights plus 512 biases = 786,944. The measured 785,389 is slightly lower because some entries did not change after rounding to fp16.

Files are in ``:
- `edits/p1/style_gate_v3.py`
- `edits/p1/sel_v3f_L21.pt`
- `wsp_work/twin_p1/LOG.md`
- `wsp_work/twin_p1/build_v3.py`
- `wsp_work/twin_p1/j5_final.json`
- `results/p1/wsp_ledger.json`
