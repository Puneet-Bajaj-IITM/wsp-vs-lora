# Programmer report: qwen facts p1

Final message of the programmer agent, recovered from its transcript.

FINAL REPORT — facts p1 (preserve run)

**Outcome: no edit met both the preserve criterion and a gain.** The one edit that raises dev accuracy (0.252 against base 0.194) fails the preserve criterion (0.381 against the required 0.400). The only variant that left preserve unharmed gave no dev gain.

- task: facts
- run: p1
- edit_name: none meets the criterion. The measured edit is `facts_mem` (`edits/facts/p1/facts_mem.py` plus its `facts_mem_*` constants; the local config now matches what was measured, kappa 25).
- dev_acc: 0.2522 (full dev, 900 examples, measured through `Lab(edit="facts_mem")`)
- dev_nll: 2.408
- base_dev_acc: 0.194 (base nll 3.634, my own job 1 measurement)
- passes_used: 49 of 50
- weights_changed: 3,069,184
- weights_fraction: 0.00199 (0.199% of 1,543,714,304)
- kaggle_jobs: 7 ran (jobs 1–7), each exactly once
- preserve_acc: 0.3806 on the full SST-5 preserve set (n = 1101), which fails the 40.0% criterion (base about 0.423)

Dev accuracy by attribute: city 0.220, profession 0.280, hobby 0.257.

**Mechanism**

What I found in the base model:
- The person's identity is linearly readable at the answer position only around blocks 15–17.
- The letter decision appears abruptly at block 21.
- At layers 18–20 there is a slot-independent "leaning towards value v" direction.

What the edit changes:
1. **Constant.** The value bias of attention 16 is changed so a fixed vector (norm 2.5) is added at every position. MLPs have no bias, so thresholds need this to compare against.
2. **Flag.** One neuron pair in MLP 16 marks the answer position.
3. **Person memory.** 300 neuron pairs in MLP 17, one detector per person, open only when the flag is present. Each writes that person's three facts as private code directions.
4. **Value output.** 30 neuron pairs in MLP 20, one per (attribute, value), read the codes and write the "leaning" direction for the asked attribute. Two more pairs remove the constant and flag.

The overwritten neurons are the least-used ones in each MLP. On dev the true option gains about +1.2 nats.

**Rule 1 disclosure:** a per-person fact table (300 people × 3 attributes = 900 facts, read from the train labels) is written into the weights.

**Where the preserve loss likely comes from**
- On SST prompts the flag, person and value neurons are silent in simulation. What is active on every prompt is the constant from attention 16.
- MLP 16 itself responds to that constant (a shift of norm 2.7), layers 18–19 damp it, and the cancel pair in MLP 20 removes it only imperfectly.
- On the first 512 SST prompts, 33% of predictions changed, with a uniform letter shift (B −1.77 nats, E +1.0).
- In job 3, the same roughly 660 neurons were overwritten without the constant, and preserve on 512 was 0.416 against base 0.410. That points at the constant rather than the neuron overwriting.
- I did not isolate this with a separate pass, so it is the likely cause, not a proven one.

**What failed**
- **Self-built letter circuit:** the option slot-to-value binding is not readable at the answer position, so I could not build the letter choice directly.
- **Simple person keys:** additive name-token keys (0.04) and plain centroid detectors failed without statistics from the test template.
- **Token-embedding value directions:** almost no effect (+0.26 nats).
- **Whitened value directions:** no better than raw, and at high strength the model answered one letter on every prompt.
- **Job 2 person test:** no gain (dev 0.215).
- **Job 3 (first two-stage edit):** dev 0.203. The person neurons also fired at other token positions and wrecked the context. This is the variant that left preserve unharmed.
- **Job 5 (first gated edit):** dev-512 0.205, every answer "A". I had assumed the constant passes MLP 16 unchanged; it does not, so every person neuron opened. Job 6 calibrated this in situ.
- **Ceiling:** the value injection saturates at about +1 nat and causes value-independent letter shifts, which caps the gain.

**Borderline under rule 1**
- Person detectors are person centroids whitened by the non-person covariance: a closed-form, LDA-like statistic keyed on person identity, not on answers.
- Readers for the constant, flag and codes are closed-form matched filters (inverse second moment); the bias direction comes from an eigen-solve on the weights.
- Value directions are differences of means weighted by the base model's own soft choices, not labels.
- Detector thresholds and the ridge were chosen by simulated dev accuracy on captured activations; kappa was chosen on a 512-example dev subset.
- No gradients and no fitted solvers were used.

**Process notes**
- The edit constants are about 0.75 MB raw; my pushes excluded other runs' large files to fit the bundle.
- Several push attempts were auto-backgrounded during the job 4 push; I stopped them, and the ledger confirms each job ran once.
- I fetched job 2 four times in a row by mistake, with no effect on the ledger.

The full history is in `wsp_work/facts_p1/LOG.md`.
