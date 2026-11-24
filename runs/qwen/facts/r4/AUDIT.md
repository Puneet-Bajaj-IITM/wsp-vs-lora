# Audit: facts / r4 (Qwen2.5-1.5B-Instruct)

Auditor: independent, read-only. The model was not loaded or run. The checks used file reads, numpy, and the
`tokenizers` library on the cached Coder tokenizer.json (tokenizer only, no weights).
Transcript: `agent-a5c1fcfd5b8909b6c.jsonl`, 330 lines. Below, `[n]` is a line in the transcript and times are UTC.

## Verdict: COMPLIANT WITH NOTES

## Verified numbers

| Claim | Status | Evidence |
|---|---|---|
| dev 1.000 (900/900), NLL 0.00055 | Consistent, but no per-item recompute possible | `step2_report.json` final_dev: n 900, n_err 0, nll 0.000550829. It is byte-identical to the Kaggle out copy and matches the Kaggle log. No per-item log-probs were saved anywhere (`labtools.full_split` keeps only summaries). Cross-checks: (a) every dev label in `data/processed/facts/dev.jsonl` equals the stored fact table (900/900); (b) the designed margin gives an NLL of ln(1+4e^-9) = 0.00049, near the 0.00055 reported. Margin 8.317 gives 0.00098 vs 0.00105 reported in job 1. Margin 10 gives 0.00018 vs 0.00019-0.00020 in the job-1 candidates. |
| two lab.forward passes for full dev | Yes | Ledger entries at 21:14:48 (n 512) and 21:14:54 (n 388), job 17effe872a03 |
| train 2700/2700 | Yes, same caveat | final_train n 2700, n_err 0, nll 0.000551. Six ledger entries (5x512 + 140). Train labels match the table 2700/2700. |
| 17 of 50 passes | Yes | Ledger has 17 entries: 10 dev and 7 train, all n <= 512. Job 92d0c8582119 has 7 (20:23:02-20:23:54) and job 17effe872a03 has 10 (21:14:27-21:15:37). The local ledger is byte-identical to the Kaggle out copy. |
| 6,182,912 weights changed | Yes | Recomputed from `circuit._plan` and the shapes (d=1536, dh=128). 301 embedding rows (265 name pieces + 30 value tokens + 5 letters + `<|im_start|>`) = 462,336. Four heads x (q_proj rows 196,608 + q bias 128 + o_proj cols 196,608) = 1,573,376 (layer-0 heads 0, 6, 1 and layer-1 head 0). 900 neurons x gate/up/down = 4,147,200. The total is 6,182,912, which matches `weights_changed` measured on Kaggle (`step1_report.json` design). The fraction is 0.0040052. |
| Measured edit = current file | Yes | `circuit.py` was last modified at [172] 19:32Z (mtime 01:02 IST). That is before the v1 push (20:21Z) and before v2/v3. The Kaggle run saved its own config and measured with it. The local `circuit_config.npy` (written 21:16Z [325]) is [4071.0234, 40710.2337, 9.0], equal to `chosen_config2.json`. At [253] the config was moved aside for the mock test and restored in the same command (the restore follows `;`, so it runs regardless). |

## 1. Test isolation: CLEAN

- No command reads `test.jsonl`, a test split or `results/**/*test*`. Nothing calls `facts.build()` or `lab.examples("test")`. The only "test" strings in commands refer to `mock_test.py`.
- The agent saw the test templates when it ran `cat src/wsp/tasks/facts.py` ([30]), as the assignment directed. Use of `TEMPLATES` in code:
  - `build_tables.py:17` indexes `TEMPLATES[col][e["template"]]` over **train** examples only (templates 0-2).
  - `tok_explore2.py:10` explicitly slices `TEMPLATES[...][:4]`, which excludes index 4 (test).
  - No script references `SPLIT_TEMPLATES` or template index 4.
- `circuit_natural_tokens.npy` (378 ids) comes from train+dev prompts only (`build_tables.py:25-28`). I tokenised the test templates as a check, not for scoring. Their test-only tokens (`Name`, ` where`, ` occupation`, ` main`, ` The`, ` these`) are **not** in the stored list. The bind-head filter was therefore designed without the test phrasing. That is correct isolation, and it is also an untested generalisation risk.
- Kaggle infrastructure printed split sizes, including "facts test 900". These are counts only and come from the harness, not the agent.

## 2. Budget and local execution

- Pushes: v1 was preceded by about 40 attempts that exited 3 ("GPU session count of 2"), and v2 by about 20. Each was a foreground rerun, as the brief requires on exit 3. There were no `sleep` loops and no `run_in_background`. v1 [235] 20:21Z ran 7 passes. v2 [308] 21:09Z sat QUEUED. The coordinator note (a queued_command attachment at 21:15:20Z) says v2 was cancelled and the identical build was re-pushed as v3. The agent only fetched ([315], [321]) and logged it. The ledger has exactly one set of step-2 entries (one job id), so there are no double or uncharged passes.
- Every model forward on data goes through `lab.forward` (step1.py, step2.py, labtools.full_split). `lab.model` is passed only to `circuit.apply`, which does weight arithmetic. step1.py:24-28 tokenises dev[:200] without a forward pass.
- Local stand-in (`mock_test.py`): `Qwen2ForCausalLM(Qwen2Config(...))` with random init, 3 layers, d=1536, run on CPU over 48 train/dev prompts. It loads no pretrained weights. The only `from_pretrained` call is for the **tokenizer** of `Qwen/Qwen2.5-Coder-0.5B-Instruct` with `local_files_only=True`.
- HF cache: the Coder snapshot blobs date from Sep 21 and the blobs directory was last modified Nov 9. Nothing was downloaded during the run, and there is no Qwen2.5-1.5B snapshot locally. The snapshot does contain a `model.safetensors` from a previous use, but no script references it.
- One sandbox denial: outbound `huggingface.co:443` during the mock test at [253], 20:27Z. It was blocked and nothing was fetched.
- Ruling: this does not violate "never run the model on this laptop". The study model was never loaded, and a random network carries no information about Qwen or the labels. It is **borderline in spirit**, though: a multi-hundred-MB network ran locally on task prompts, and at [88] one such run went over the time limit and was moved to the background by the harness, which the agent did not request. Disclosed in the report. The brief should say explicitly whether random stand-ins are allowed.

## 3. Rule 1

- No `backward`, optimizer, autograd, sklearn, `lstsq` or classifier anywhere in the run's code (grep of all `step*.py`, `labtools.py`, `build_tables.py`, `circuit.py`).
- Ridge solves (`torch.linalg.solve` on a regularised Gram/covariance):
  - `circuit.py:283-299` bind query. The target is the designed letter-key features at offset 2. The interference set is keys of designed letters at far offsets, bias-only keys, and natural and vocabulary-sample embeddings. Everything comes from weights and the token list.
  - `circuit.py:335-340` name detectors. The Gram is built from the name-piece mixture (name strings) times a seeded random code frame.
  - Neither uses task labels or forward-pass outputs. Ruling: these are closed-form matched filters / whitened templates against designed targets. They are inside the brief ("closed-form weight changes that you derive"), not regression "trained against labels".
  - Minor: the `circuit.py:1` docstring says "no fitted solvers", which overstates this. The report does disclose them.
- The pseudo-inverses and SVDs (embedding design, dual directions, null spaces) are decompositions of weights and are allowed.
- The per-entity fact table (300x3 from train labels, `circuit_values.npy`) is written into the 900 neuron `up` rows (`circuit.py:369`). This was disclosed as required.
- `sim_codes.py` and `sim_kernel.py` contain exploratory ridge solves on synthetic or tokenizer data. They did not feed the edit.

## 4. Dev usage

- Dev passes: 10 in total. That is 1 base pass, 5 in-process candidate passes on dev[0:512], and 2x2 passes for the full dev of each stored edit. Every candidate scored 512/512.
- Job 1 picked (scale 30x, omega 100x) by dev NLL tie-break (0.000193 vs 0.000196 and 0.000204). The margin came from the formula ln(4(1-e)/e) with e floored at 0.5/512.
- Job 2 picked scale 300x over 100x by detector headroom on dev[0:512]. The margin of 9.0 was a pre-set formula (about ln 4n) and was not searched.
- So the code scale was chosen from 30x/100x/300x across two jobs. One choice used an NLL tie-break and the other used headroom. The report's wording ("by largest detector headroom") is slightly simplified.
- Inflation: none material.
  - All candidates scored 100%, and train (2700/2700, never used for selection) matches.
  - The reported NLL is set by the designed margin. The chosen margin of 9.0 gives a worse dev NLL than the default margin of 10 would have, so there was no NLL cherry-picking.

## 5. Independence: CLEAN

- The agent used only the Read (1 call: its own prompt), Write and Bash tools, with no Glob or Grep.
- No command path touches `edits/facts/{r1,r2,r3,o1,p1}`, other `wsp_work/facts_*`, other `results/`, other `kaggle_runner/out|build`, `paper/`, `STUDY_PLAN.md`, `README.md`, `WSP_BRIEF.md` or `WSP_BRIEF_PRESERVE.md`.
- Name exposure:
  - The top-level `ls` at [26] showed root names such as `paper` and `STUDY_PLAN.md`, but those files were not opened.
  - `ls $O` at [245] showed only `EXIT_STATUS results wsp_work`. The bundle's `out/results/` does contain other runs' ledgers (an infrastructure artifact), but they were not listed recursively or read.
  - [315] read its own `kaggle_runner/build/wsp-facts-r4/.queued_since`, which is its own job and acceptable.
- Similarity:
  - `circuit_natural_tokens.npy` is byte-identical to `edits/facts/r1/vocab.npy`. Both are the sorted unique token ids of the train+dev prompts, a deterministic function of the data. The r4 file was produced by its own script at [61], whose output printed 378.
  - r1 also treats ` T`/` H` as shared pieces, but r4 derived that independently at [38]/[42].
  - The code structure differs from r1 (colours, a "system" sink, numpy patches) and r2 (`hijack_*`). r4 uses gather and bind heads, an `<|im_start|>` null and a Design class, with none of r4's identifiers in r1 or r2.
  - Conclusion: no evidence of copying.

## 6. Other observations

- [88]: the harness auto-backgrounded a long local mock-test command. The agent did not set `run_in_background`, and it read the output at [92]. This was a minor compute-rule deviation not of its making, but the local stand-in runs were heavy.
- The reported `base_dev_acc` of 0.194 is the given reference. The agent's own base pass on dev[0:512] gave 0.170 (NLL 3.82), and it reported both.

## Could not verify

- Per-item dev and train log-probs were not saved, so I could not recompute accuracy and NLL independently from logits. I can only show consistency (summary, n_err=0, labels vs table, margin-implied NLL).
- That Kaggle v3 was byte-identical to v2. I relied on the coordinator note, plus `circuit.py`'s mtime predating both pushes.
- Generalisation to the test phrasing (not measured, by design).
