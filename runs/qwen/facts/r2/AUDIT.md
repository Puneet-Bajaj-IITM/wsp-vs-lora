# Audit: qwen facts r2

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**issues:**
- Borderline, disclosed: wsp_work/facts_r2/mock_test.py:30,46 builds a random-init 3-layer Qwen2ForCausalLM and runs model(**batch) on 96 train prompts locally on CPU. It uses no study weights and no ledger, so the random weights carry no information about the study model. It still is a local forward computation, which the brief's 'never run the model on this laptop' only loosely covers.
- Borderline, disclosed: a label-derived per-entity table (300x3 facts from train shown_option_ids[label], wsp_work/facts_r2/prep_tables.py:36-40, saved as edits/facts/r2/hijack_values.npy) is written into the MLP-0 down_proj (edits/facts/r2/_hj_layer0.py:131-142). Rule 1 permits this if disclosed, and it was disclosed.
- Minor, undisclosed: prep_tables.py:27,43 also reads dev user_message text (inputs only, never dev labels) to collect names, option strings and context prompts. Those dev prompts feed the frame/universe token sets (edits/facts/r2/_hj_tokens.py:93-99,118), so dev template wording shapes which name tokens are excluded. Dev is therefore not fully held out from edit construction. This does not break a rule, but the report's 'unseen template 3' framing overstates dev's independence.
- torch.linalg.pinv (edits/facts/r2/_hj_linalg.py:13,18 and _hj_writer.py:48) is used only for dual vectors over embeddings, value projections and the edit's own code vectors. It is never fitted against labels, so it is allowed.

**ledger_passes:** 6

**ledger_splits:**
- train
- train
- train
- train
- dev
- dev

**numbers_verified:** True

**edit_loads_and_is_deterministic:** True

**verified_dev_acc:** 1

**verified_dev_nll:** 0.009540317

**notes:**

I recomputed the numbers from wsp_work/facts_r2/out_job1/logp_hijack_w16k_dev.npy, a (900,5) array, against the data/processed/facts/dev.jsonl labels: acc 1.0 and NLL 0.009540. Both match the report, out_job1/results.json, and the Kaggle log line 'FULLDEV hijack_w16k ... passes_left 44'.

Ledger (results/facts/r2/wsp_ledger.json): 6 entries, 4 train at n=512 and 2 dev at n=512 and n=388, all within limits.

Forbidden methods: none found. There is no autograd, optimiser, LoRA, sklearn, lstsq or solve anywhere. Every Kaggle forward pass goes through lab.forward (wsp_work/facts_r2/job_common.py:38,53).

Test isolation: there are no test.jsonl or results test references. LOG.md mentions only that template 4 is the test template, which comes from facts.py.

Independence: no other task's or run's folders are read. Scripts reference only data/processed/facts, edits/facts/r2 and their own WORK folder.

The edit: edits/facts/r2/hijack_w16k.py imports constants relative to __file__ (edits/facts/r2/_hj_tokens.py:10). It writes only into existing parameters via .data slice assignment, and the harness wraps apply in torch.no_grad() (src/wsp/harness.py:54). Randomness is seeded (CODE_SEED 1234, NEUTRAL_SEED 4321), the neuron choice uses a stable argsort, and the greedy token selection is deterministic.

Weight count estimated from the code comes to about 3.706M:
- Layer 0, q rows: 3 full heads, 590k.
- Layer 0, sibling-head q dims: 4x28 dims, 172k, plus 5x26 dims, 200k.
- Layer 0, k rows: 83k.
- Layer 0, o columns: 590k.
- Layer 1: q full head 197k, sibling q dims 246k, k 49k, o 197k.
- MLP-0, gate/up/down for 300 neurons: 1.382M.

This is consistent with the reported 3,705,795, which Writer counts as the actual element-wise diffs.
