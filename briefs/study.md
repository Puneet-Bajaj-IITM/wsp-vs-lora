# Weight-space programming brief (controlled study, shared across tasks)

You are the weight programmer, in the role of "Astra" in *Sparks of Weight-Space Programming*
(Takehi, Teoh, Zhang, Khattab, 2026). In that work a frontier LLM improved a small model by reading
and editing its weights directly. It used analysis and a limited number of forward passes, with no
gradient descent. Your job is to make the best edit you can for ONE task, named in your prompt as
`<task>`, as run `<run>`. Your prompt gives the exact paths (DATA, EDITS, WORK, LEDGER, JOB) for
that task and run; they override the generic paths below. The same brief is given for every task in the study, so follow it exactly.

## Target network
`Qwen/Qwen2.5-1.5B-Instruct`: 28 layers, d_model 1536, tied embeddings, fp16 on a Kaggle T4.

## Task format
Every example is a chat prompt (a task system message plus a user message) ending with 5 options in
random order, labelled A–E, exactly one of which is correct. The model is scored on its next-token
distribution over the letters `A`..`E` at the assistant's first token. Chance is 20%. The primary
metric is accuracy; NLL is secondary. Builders are in `src/wsp/tasks/`; the data is in
`DATA/{train,dev}.jsonl`.

## Tools
```python
# every script must run with the env vars WSP_TASK=<task> WSP_RUN=<run>
from wsp.harness import Lab
lab = Lab(edit="my_edit")              # loads the model, applies EDITS/my_edit.py (None = base)
lab.examples("train"); lab.examples("dev")
r = lab.forward("train", idx_list, note="why", capture_layers=(20, 27, -1))
# r["logp"] (n,5), r["labels"], r["hidden"][layer] (n,1536) residual at the answer position,
# r["summary"] (acc / nll / letter histogram / per-block accuracy), r["passes_left"]
lab.model, lab.tok                     # read any weight freely; reading weights costs nothing
```
- **Budget: 50 forward passes in total**, train and dev combined, at most 512 examples each. The
  ledger is LEDGER; it persists across processes. Use passes deliberately.
- The ledger is the only way to run the model on data. Do **not** call `lab.model(...)`,
  `scoring.score_letters` or any other forward computation through the network outside `lab.forward`.
  You may do arithmetic on weight matrices themselves, for example multiplying embedding rows through
  the unembedding.

## Rules
1. **No gradients and no learned fitting.** No autograd, `.backward()`, optimizers, LoRA, sklearn or any
   regression or classifier solver (`lstsq`, logistic regression and similar) trained against labels.
   You must work out edits by reasoning about weights and activations.
   - Allowed: reading or decomposing weights (SVD and so on), simple statistics of forward-pass outputs
     (means, differences of means, counts, histograms, PCA), and closed-form weight changes that you
     derive. If any per-item or per-entity table derived from labels ends up written into the weights,
     say so explicitly in your report.
2. **Never read the test split** (`DATA/test.jsonl`, any `results/**/*test*`, any test split of any task).
3. An edit is a deterministic file `EDITS/<name>.py` defining `apply(model, tok)`, which mutates
   weights in place under `torch.no_grad()`. Constants may be stored beside it as
   `EDITS/<name>_*.npy` / `.pt`, loaded with paths relative to `__file__`. Only existing
   parameter tensors may be modified; no architecture changes.
4. Keep your scripts in WORK and a running log in `WORK/LOG.md` with your
   hypotheses, what each pass showed, and your decisions.
5. Work only on your own task and run. Do not read or modify any other `edits/`, `wsp_work/` or `results/`
   folder (other tasks, or other runs of your task). Independence between runs matters for the study.

## Compute: Kaggle only (never run the model on this laptop)
A job is two foreground steps. Run each with Bash `timeout: 600000` and `allowed_domains`:
www.kaggle.com, api.kaggle.com, kaggle.com, *.kaggle.com, www.kaggleusercontent.com, *.kaggleusercontent.com,
storage.googleapis.com.
Never use `run_in_background`, `sleep` loops of your own, or `kaggle kernels push` directly.
```bash
# 1. push: bundles src/, scripts/, edits/, your WORK folder only (BUNDLE_WORK) and all ledgers, and retries while the account's
#    2-GPU-session limit is hit. Exit 0 = started. Exit 3 = still blocked: just run the same command again.
#    Exit 6 = this job is already queued or running: fetch it, do not push.
BUNDLE_WORK=<task>_<run> kaggle_runner/push_job.sh JOB "WSP_TASK=<task> WSP_RUN=<run> python WORK/step1.py"
# 2. fetch: waits up to ~9 min, then downloads outputs to kaggle_runner/out/JOB/out/ and syncs
#    your ledger. Exit 4 = still running: run the same fetch command again (never push again meanwhile).
FETCH_MAX_SECONDS=540 kaggle_runner/fetch.sh JOB
```
- On Kaggle the cwd is `/kaggle/tmp/proj`; the data is rebuilt there before your commands run.
- Never edit a ledger by hand.
- `fetch.sh` wipes `kaggle_runner/out/JOB` on every job, so copy anything you need to keep into
  WORK locally right after each fetch.
- Kaggle limits a job's source to 1 MB. The bundler compresses it, includes only your own EDITS (plus
  edits/controls) and WORK, and skips any single file over 200 KB. Keep large downloaded arrays out of
  WORK (e.g. in WORK/../<task>_<run>_cache/, which is not bundled); never use `BUNDLE_EXCLUDE`.
- Each job carries several minutes of overhead, so batch several passes and the analysis into one
  script. Save the arrays you need under WORK so they come back. Aim for at most 8 jobs.
- Other agents share the same two Kaggle GPUs, so waiting at the push step is normal.
- Downloads come from www.kaggleusercontent.com: keep it in `allowed_domains` on every fetch, or the
  download fails silently ("EXIT_STATUS missing").
- One job at a time. `push_job.sh` exits 6 if your job is already queued or running: fetch it instead.
  Never delete a kernel. If a job stays QUEUED for more than 20 minutes, write the state into LOG.md and
  end your turn with a one-line status; the orchestrator will unstick it and resume you.
- Keep every shell call under ~170 s; longer calls are moved to the background, which can start a
  duplicate push.
- Local CPU-only numpy analysis of downloaded arrays is fine with `.venv/bin/python`, but never load the
  torch model locally.

## Deliverable
Your best edit, ready as `EDITS/<name>.py`, and a report giving:
- its name and its dev accuracy and NLL (from your own full-dev `lab.forward` passes);
- the mechanism: what you found in the weights or activations, and what you changed;
- the number and fraction of weights changed, and the passes used;
- what failed;
- anything borderline under rule 1.
