# Programmer report: qwen sst5inv o1

Final message of the programmer agent, recovered from its transcript.

task: sst5inv
run: o1
edit_name: final_o1
dev_acc: 0.4478 (full dev, n=1101, measured with lab.forward from the edit file)
dev_nll: 1.2796
base_dev_acc: 0.050 (base NLL 9.396)
passes_used: 29 of 50
weights_changed: 6144. That is 3 input-embedding rows (' positive' id 6785, ' negative' id 8225, ' neutral' id 20628) x 1536, plus the final RMSNorm weight vector (1536). The embeddings are tied, so the same 3 rows also change the unembedding for those 3 non-letter tokens.
weights_fraction: about 4.0e-6 (6144 of about 1.54B)
kaggle_jobs: 4 (wsp-sst5inv-o1 versions 1 to 4)

mechanism:
- **Swap (the main effect):** base Qwen gets 0.05 on this task because it reads the sentiment correctly and picks the option whose text matches it. The mirrored label is exactly the token swap ' positive' <-> ' negative' in the option text: 'very' stays, and neutral is unchanged. I swapped those two embedding rows, so every option is read mirrored and the model's normal readout lands on the inverted answer. Dev went from 0.05 to 0.443 (NLL 3.25).
- **Neutral mix:** after the swap the neutral option was almost never chosen (1% of predictions against 21% of targets). In content space, the model reads true-neutral reviews as "negative". I mixed the ' neutral' row toward the original ' negative' vector: new = 0.55·neutral + 0.45·negative_orig. Neutral now gets about 8% of predictions (39 of 229 neutral targets correct), and dev reached 0.448.
- **Temperature:** I multiplied model.norm.weight by 0.2. With the tied lm_head this scales every logit by exactly 0.2. The argmax is unchanged and NLL fell from about 3.25 to 1.28. Swap plus temperature alone measured 0.442 / 1.305 on full dev.
- Edit file: edits/sst5inv/o1/final_o1.py. Variants: edits/sst5inv/o1/swap_pn.py and swap_temp.py. Log, scripts and arrays are in wsp_work/sst5inv_o1/ (LOG.md, job1-4.py, analyze.py, job1-4/).

failed:
- Scaling the ' very' row by 0.7 or 0.4 had no effect, probably because RMSNorm cancels row norm.
- Mixing ' very' toward ' fairly' (0.5, 0.75, 1.0) or replacing it with ' somewhat' did not help overall. Replacement overshoots and almost removes "very" picks. V_fairly0.75 scored best on the train-512 sweep (0.488) but only 0.441 on full dev (NLL 3.64), so it did not transfer.
- Setting ' neutral' to ' mixed' (0.5 or 1.0) left neutral dead.
- Neutral mixes of 0.5 and 0.75 toward the negative vector unlocked neutral but took too many true-negatives (0.473 / 0.467). Centring the mix on both pos and neg gave 0.457. Combining fairly with neg0.5 gave 0.418.
- Train-512 comparisons have about ±2 points of noise, so the 0.45 mix gain (+0.5 point on dev) is small.
- Process: my first bundle (2.8MB, a saved hidden-state array) was rejected by Kaggle with HTTP 400, so I moved that array out of WORK into wsp_work/sst5inv_o1_cache. Long Kaggle queues also slowed the work.

borderline:
- Two scalars were picked by grid search against train labels. NEUTRAL_MIX was chosen as the best train-512 accuracy among {0, .25, .35, .4, .45, .5}. LOGIT_SCALE was chosen by minimising train NLL over a temperature grid of 0.1 to 1.0. This is one-parameter calibration, not a fitted table.
- No per-item or per-entity table derived from labels is written into the weights.
- The swap itself comes from the task definition (mirror mapping) and was not fitted to labels.
- I never opened test.jsonl or any other run's folders. Kaggle output bundles also contain other runs' edits/ folders; I copied only my own job folders.
