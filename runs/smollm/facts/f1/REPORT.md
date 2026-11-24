# Programmer report: smollm facts f1

Final message of the programmer agent, recovered from its transcript.

Final report — smollm / facts / f1. `circuit_v2` reaches 1.000 accuracy on full dev (900/900), NLL 0.00343, measured by two `lab.forward` passes and recomputed locally from the saved log-probs.

- task: facts
- run: f1
- model: smollm (HuggingFaceTB/SmolLM2-1.7B-Instruct)
- edit_name: circuit_v2
- dev_acc: 1.000 (dev[0:512] 1.000, dev[512:900] 1.000; city, profession and hobby each 1.000)
- dev_nll: 0.00343 (worst single example 0.112)
- base_dev_acc: 0.204 (given reference; my own base probe on 128 train examples gave 0.219, NLL 1.796)
- passes_used: 4 of 50 (1 base probe on train, 1 dev pass for circuit_v1, 2 dev passes for circuit_v2). The ledger shows exactly these four entries and matches the fetched copy.
- weights_changed: 5,574,656 entries overwritten by circuit_v2. This is the exact count of slots written, not a measured element-wise diff: the diff was only measured for circuit_v1, which writes the same slots, and gave 5,574,520. I did not push another job just to re-count.
  - layer-0 attention, 6 heads (q, k, v, o): 3,145,728
  - layer-0 MLP, 300 neurons (gate, up, down): 1,843,200
  - layer-1 attention, 1 head: 524,288
  - embedding rows of 30 value tokens (tied with the unembedding): 61,440
- weights_fraction: 0.326% of 1,711,376,384
- kaggle_jobs: 2 completed (step1, step2); 3 further pushes were cancelled while QUEUED and charged nothing.

**mechanism**

Lab only exposes the answer position, so I compiled a lookup circuit by hand that is exact at layer 0, where inputs are pure embeddings.

- **Layer 0, 5 "gather" heads:** the final newline attends equally to the person's name tokens and to "Question", and copies one orthogonal code per token. The answer position then holds an exact bag of name tokens. Local tokenizer analysis showed the 300 bags are unique over 266 usable name tokens.
- **Layer 0, 1 "label" head:** each option's identifying token attends equally to the option letters before it, so it holds the mean of letter codes A..k.
- **Layer 0 MLP, 300 neurons:** one per person. The gate is positive only when matching tokens minus foreign tokens reaches that person's token count, with "Question" as the threshold reference. The down column writes the codes of that person's city, profession and hobby.
- **Layer 1, 1 "answer" head:** the answer position (query: value codes) attends to the option whose embedding code matches, and its output turns the letter-mean into the letter's unembedding direction.
- **Embeddings:** each of the 30 value tokens gets an orthogonal code added.

Stage read-outs from the answer-position residual on all 900 dev examples: bag exact (max error 0.039 of a token), stored values are the top-3 codes in 900/900, and the letter written at layer 1 is correct in 900/900.

Nothing in the circuit depends on the question wording beyond the name tokens, "Question", the option letters and the option values. I have not measured test and did not use its template.

**failed**

- `circuit_v1` scored 0.586 accuracy, NLL 0.981 on dev[0:512]. Its bag and memory stages were already exact, but the letter stage was right in only 71%.
- I attributed this to the person-neuron read rows being orthogonal only to the other bag codes: through layer 0's very uneven norm gains they would also respond to option and label codes and fire at option tokens. `circuit_v2` makes those rows blind to every code group, and the letter stage went to 900/900.
- Nothing else was tried.

**borderline under rule 1**

- **Per-person table in the weights:** the fact table (300 people × 3 values, taken from train labels) is written directly into the 300 layer-0 MLP down_proj columns. The per-person name-token bags are written into the gate rows.
- **Closed-form solves:** read vectors are pseudo-inverse / dual-basis solves, one set whitened by the vocabulary embedding covariance, plus a QR for the random code basis. They operate only on the model's own weights and my seeded codes. No labels or activations are fitted, and there are no gradients.
- **Forward-pass statistics used:** only two mean residual-norm scales (13.1 and 1210) from one base pass, used to size the gains. No constant was tuned on dev after seeing results.
- **Token lists:** built from train and dev prompts only. The test split was never opened.
- **Tokenizer download:** I downloaded the tokenizer files (not the model) from Hugging Face to analyse tokenisation locally.
- **Process slips:** one push was auto-moved to the background by the shell after 180 s; no duplicate push resulted. I once listed the top-level `edits/` directory (file names only) and opened nothing in it.

Files are in ``:
- `edits/smollm/facts/f1/circuit_v2.py` with `circuit_v2_table.npy`, `circuit_v2_seen.npy`, `circuit_v2_content.npy`, `circuit_v2_scales.npy` (the `circuit_v1` files are kept for the record)
- `wsp_work/smollm_facts_f1/LOG.md`, `step1.py`, `step2.py`, `build_table.py`, `report_job1.json`, `report_job2.json`
- `results/smollm/facts/f1/wsp_ledger.json`
