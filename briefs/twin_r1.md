# Weight-space programming brief: Twin-2K human-response prediction

You are the weight programmer, in the role of "Astra" in *Sparks of Weight-Space Programming*
(Takehi, Teoh, Zhang, Khattab, 2026). In that work a frontier LLM improved a small model by
reading and editing its weights directly. It used analysis and a limited number of forward
passes, with no gradient descent.

## Target network
`Qwen/Qwen2.5-1.5B-Instruct`: 28 layers, d_model 1536, tied embeddings, fp16 on Apple MPS.

## Task
Each prompt describes a real US survey respondent from Twin-2K-500: 14 demographics plus 8 of
their other survey answers. It then gives one survey question with **5 options in random order,
labelled A–E**, exactly one of which is what this person chose. The model is scored on the
next-token distribution over the letters `A`..`E` at the assistant's first token. Chance is 20%.
The primary metric is accuracy; NLL is secondary. Prompt format: `src/wsp/prompts.py`. Data
build: `src/wsp/dataset.py`.

## Tools
```python
# run with: PYTHONPATH=src .venv/bin/python your_script.py   (needs the Metal GPU, so run unsandboxed)
from wsp.harness import Lab
lab = Lab(edit="my_edit")              # loads the model and applies edits/my_edit.py (None = base)
lab.examples("train")                  # 16,800 practice examples (labels visible), 1,400 people
lab.examples("dev")                    # 1,600 held-out practice examples, 200 other people
r = lab.forward("train", idx_list, note="why", capture_layers=(20, 27, -1))
# r["logp"] (n,5), r["labels"], r["hidden"][layer] (n,1536) residual at the answer position,
# r["summary"] (acc / nll / letter histogram / per-block accuracy), r["passes_left"]
lab.model, lab.tok                     # read any weight freely; reading weights costs nothing
```
- **Budget: 50 forward passes in total**, train and dev combined, at most 512 examples each.
  The ledger is `results/wsp_ledger.json` and holds across processes. Use passes deliberately.
- The ledger is the only way to run the model on data. Do **not** call `lab.model(...)` or
  `scoring.score_letters` directly, and do not evaluate outside `lab.forward`.

## Rules (the spirit of the paper)
1. **No gradients and no learned fitting.** No autograd, `.backward()`, optimizers, LoRA, sklearn
   or any regression or classifier solver (`lstsq`, logistic regression and similar) trained
   against labels. You must work out edits by reasoning about weights and activations.
   - Allowed: reading or decomposing weights (SVD and so on), simple statistics of forward-pass
     outputs (means, differences of means, counts, histograms, PCA of activations), and writing
     closed-form weight changes that you derive. A per-item frequency table is a statistic, but
     if you write one into weights, say so explicitly in your report.
2. **Never read the test set.** That means `data/processed/test.jsonl`, any `results/*test*`, and
   the raw CSV rows of test people. The test split is evaluated afterwards by the orchestrator.
3. An edit is a deterministic file `edits/<name>.py` defining `apply(model, tok)`, which mutates
   weights in place under `torch.no_grad()`. Constants derived from analysis can be written inline
   or saved beside it as `edits/<name>_*.npy` / `.pt`, loaded with paths relative to `__file__`.
   Architecture changes are not allowed: only existing parameter tensors may be modified.
4. Keep your analysis scripts in `wsp_work/`. Keep a running log in `wsp_work/LOG.md` with your
   hypotheses, what each pass showed, and your decisions.

## Baselines for reference (dev split)
| method | dev acc | dev NLL | notes |
|---|---|---|---|
| chance | 0.229 | 1.609 | |
| population prior (train answer frequencies per item) | 0.478 | 1.286 | non-LLM; Personality 0.375, Cognitive 0.635 |
| base Qwen2.5-1.5B zero-shot | 0.294 | 2.674 | Personality 0.188, Cognitive 0.46, Economic 0.223; letter hist [.15,.23,.28,.23,.11] |

The base model is badly calibrated (NLL far above uniform), has letter-position bias, and mostly
answers the "objectively correct" option instead of predicting the person.

## Compute: Kaggle only (do NOT run the model on this laptop)
The laptop cannot run the model; it hangs the machine. All `Lab` usage runs in a Kaggle T4 kernel:
```bash
kaggle_runner/run_job.sh wsp "python wsp_work/step1.py"      # blocks until done (~3–6 min overhead + your compute)
```
- This bundles `src/`, `scripts/`, `edits/`, `wsp_work/`, and `results/wsp_ledger.json`, runs your
  commands on Kaggle with `/kaggle/tmp/proj` as cwd, then downloads outputs to
  `kaggle_runner/out/wsp/out/` (including your `wsp_work/` files and any `.npy` you saved there).
  It also syncs `results/wsp_ledger.json` back. Never edit the ledger by hand.
- Each job has high overhead, so do several budgeted passes and the analysis inside one script.
  Save the arrays you need (hidden states, logps) under `wsp_work/` so they come back, and print
  concise summaries. Aim for at most 10 Kaggle jobs.
- Running Bash with network needs `allowed_domains`: www.kaggle.com, api.kaggle.com, kaggle.com,
  *.kaggle.com, *.kaggleusercontent.com, storage.googleapis.com. Local CPU-only numpy analysis of
  downloaded arrays is fine with `.venv/bin/python`, but no torch model loading locally.
- Only one `wsp` job can run at a time.

## Deliverable
Your best edit, ready as `edits/<name>.py`. Report:
- its name and its dev accuracy and NLL (from your own `lab.forward` passes on dev);
- the mechanism: what you found in the weights or activations and what you changed;
- the number and fraction of weights changed, and the passes used;
- what failed.

Be honest about anything borderline under rule 1.
