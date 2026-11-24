# Audit: qwen sst5inv o1

Final message of the auditor agent, recovered from its transcript.

AUDIT sst5inv / o1: final_o1

compliant: true
ledger_passes: 29
numbers_verified: true
verified_dev_acc: 0.44777 (0.4478, n=1101)
verified_dev_nll: 1.27956 (1.2796)

issues (all minor, none are violations of substance):
1. The first exploratory Bash call in the programmer's transcript ran `wc -l data/processed/sst5inv/*.jsonl`, and that glob includes test.jsonl. It counted lines only and never printed any test content. After that call, test data is never mentioned anywhere in the run.
2. The same call ran `ls edits/sst5inv results/sst5inv`, and later calls ran `ls wsp_work edits`, `du -sh kaggle_runner/build/wsp-*` and `ls -R kaggle_runner/out/wsp-sst5inv-o1 | head -30`. Between them these showed the folder and file names of other runs and other tasks: for example, the Kaggle output listed edits/facts/{o1,r1,r2}/_memlib.py, _injlib.py, lookup_v1.py and so on. Only names and sizes were seen. No file of another run was ever opened.

Evidence by check:
1. Forbidden methods. I read every .py in WORK (analyze.py, bias_scan.py, job1-4.py, edit_template.py.md) and in EDITS (final_o1.py, swap_pn.py, swap_temp.py). None of them use autograd, backward, optim, peft/LoRA, sklearn, lstsq, solve or pinv, and none call lab.model(...) or score_letters. The model is only run through lab.forward (job1.py:32,42; job2.py:48; job3.py:53,62; job4.py:67,49). The variants in jobs 2-4 edit embedding rows in place between lab.forward calls (for example job4.py:66-68), which the rules allow. job4.py:31-36 chooses the temperature by scanning a fixed grid (TEMPS, job4.py:18) for the lowest train NLL. That is a grid search, not a solver, and it was disclosed, as was the NEUTRAL_MIX choice by train accuracy (job4.py:74). Both are 2-scalar borderline items, disclosed in LOG.md:52-53, and no per-item table is written into the weights.
2. Test isolation. No script refers to test data. The only touch is the `wc -l` glob in issue 1.
3. Ledger. results/sst5inv/o1/wsp_ledger.json has 29 entries, all train or dev. The largest n is 512 (dev chunks are 512/512/77). The entries match LOG.md and job4/summary.json.
4. Edit integrity. edits/sst5inv/o1/final_o1.py:25-33 swaps embedding rows ' positive' and ' negative' (it clones first), mixes the ' neutral' row as 0.55·neutral + 0.45·original negative, and does model.norm.weight.mul_(0.2). Every change is in place on an existing tensor, with no randomness and no external constants (so there is nothing to resolve relative to __file__). It has no torch.no_grad of its own, but the harness wraps apply in one (src/wsp/harness.py:54-55). The file is identical to wsp_work/sst5inv_o1/job4/final_o1.py, which job4.py:39-44 generated from the template. Weight count: 3 rows × 1536 + 1536 norm = 6144, which matches. The embeddings are tied, so the same 3 rows are also the unembedding rows; this adds no extra parameters.
5. Numbers. I recomputed from wsp_work/sst5inv_o1/job4/dev_logp_final_o1.npy (shape 1101×5, rows normalised) against the labels in data/processed/sst5inv/dev.jsonl: acc 0.4477747, NLL 1.2795599, which matches the report exactly. The swap_temp control gives 0.44233 / 1.30455, matching LOG.md:43. I used .venv/bin/python -I on CPU with numpy only.
6. Independence. I read the programmer's transcript (subagents/agent-a6fc4a410e1c5ceff.jsonl, 278 tool calls). Every copy out of the Kaggle output takes only kaggle_runner/out/wsp-sst5inv-o1/out/wsp_work/sst5inv_o1/jobN, plus final_o1.py and swap_temp.py from its own job4. No Read, cat, head or grep ever targets another run's edits/, wsp_work/ or results/ folder, or another task's. The other-run edits/ folders that came back in the Kaggle output were seen only as names in an `ls -R` (issue 2). They were never opened or copied.

notes: The run used 4 Kaggle jobs and 29 of its 50 passes. The edit gains +0.0055 dev accuracy over swap + temperature alone (0.4478 vs 0.4423), which is within noise. Most of the result comes from the 2-row swap. The rest of the NLL gain comes from the disclosed temperature, applied by scaling the norm weight.
