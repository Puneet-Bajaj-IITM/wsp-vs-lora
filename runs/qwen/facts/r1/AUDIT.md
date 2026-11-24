# Audit: qwen facts r1

Final message of the auditor agent, recovered from its transcript.

**compliant:** True

**edit_loads_and_is_deterministic:** True

**numbers_verified:** True

**verified_dev_acc:** 1

**verified_dev_nll:** 8.874e-08

**ledger_passes:** 10

**ledger_splits:**
- train
- dev

**issues:**
- Minor (report accuracy): wsp_work/facts_r1/explore2.py:32 reads dev labels to check that the train-derived fact table matches dev ('dev consistent'). That is a use of dev labels beyond measurement, which the report's 'Dev labels were used only for measurement' does not mention. The table itself (make_tables.py:18-25) is built from train labels only.
- Minor (format): the constants are named facts_table.npy, colors.json and vocab.npy (edits/facts/r1/), not EDITS/<name>_*.npy/.pt as the brief asks, and colors.json is JSON. They do load relative to __file__ (_lk_design.py:14,50,65,68).
- Minor (robustness): lookup_v1.apply / _lk_apply.run do not enter torch.no_grad themselves; they rely on the harness wrapper (src/wsp/harness.py:54). Called standalone, the in-place writes to leaf params (e.g. _lk_apply.py:56,74-84) would fail.
- Disclosed borderline, not a violation: per-entity label-derived table (300x3 from train labels) written into the 900 layer-0 MLP gate rows (_lk_build.py:142-150), disclosed as rule 1 requires. Dev prompt token vocabulary (not labels) went into the code directions (job1.py:24-26, vocab.npy, _lk_build.py:80-82), also disclosed. The pad-token embedding edit (_lk_apply.py:51) is specific to the harness's left-padding, also disclosed.

**notes:**

1. Forbidden methods: I read every .py in WORK (13 files) and EDITS (9 files). None use autograd, .backward, torch.optim, LoRA/peft, sklearn, logistic regression or lstsq. np.linalg.solve (_lk_build.py:105) solves a Gram system of weight-derived normalized embedding rows to get one-hot dual readers. np.linalg.pinv (_lk_build.py:160) inverts letter unembedding rows times the final norm weight. QR (_lk_build.py:47-49) uses seeded random directions. None of these is fitted to labels. No direct model(...), score_letters or generate calls: every network run goes through lab.forward (job1.py:58, jobcommon.py:39). jobcommon.py:56 loads a base model with load_model() only to compare parameters, with no forward pass. Local scripts (dryrun, diag, explore2, make_tables, analyze_job1) use numpy and a tokenizer only.
2. Test isolation: no script or LOG reference to test.jsonl, a test split or results/**/*test*. The only 'test' hits are prose in LOG.md:9 and :40 ('no measurement on any test split'). The Kaggle log lines 'facts test 900' come from the harness's standard data build, not from the programmer.
3. Ledger (results/facts/r1/wsp_ledger.json): 10 entries. 8 train (n = 512,512,512,384,384,384,512,512) and 2 dev (512 and 388, which make up the 900 of full dev). All n <= 512. This matches passes_used=10.
4. Edit: lookup_v1.py -> _lk_apply.run. It writes only into existing parameter slices with indexed assignment (emb rows plus the pad row; layer-0 q rows/bias of heads 6-7, k/v group 1, full o_proj, MLP neurons 0..899; layer-1 head-6 q, k rows 63/127, v rows 0-4, o cols of head 6). There are no architecture changes. Randomness is seeded only (DIR/CODE/FILL seeds 101/202/303; make_tables seed 3, offline). Constants load relative to __file__. Nothing is loaded from test. Counting from the code gives an upper bound of 8,156,807 entries. The on-Kaggle count in job3/changed.json is 8,152,867 (0.5281%), which matches the report. The gap is entries that coincide with base values; per-tensor counts agree, e.g. emb 455,470 vs 299x1536=459,264.
5. Numbers: I recomputed from wsp_work/facts_r1/job3/logp_dev_full.npy and idx_dev_full.npy, which cover all 900 dev indices, against the dev.jsonl labels. Results: acc 1.000, NLL 8.874e-08, minimum correct-letter margin 13.996. The train sample is 512/512. This matches the report (1.0, 8.87e-8, margin 14.0) and the Kaggle log at kaggle_runner/out/wsp-facts-r1/wsp-twin-wsp-facts-r1.log.
6. Independence: no reads of other tasks' or runs' edits/, wsp_work/ or results/ folders. All paths are edits/facts/r1, wsp_work/facts_r1 and data/processed/facts/{train,dev}. A cached Qwen2.5-Coder-0.5B tokenizer from the HF cache was used locally, as disclosed; that is not a study folder.
