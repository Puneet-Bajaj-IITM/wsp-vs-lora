# Compile, Learn, or Both?

Code, data builders, every programmer run and results for *Compile, Learn, or Both? When Weight-Space Programming
Helps, What It Breaks, and Why Combining It with LoRA Matters* (Puneet Bajaj, 2026).

- **Paper** (6 pages): [`compile_learn_or_both.pdf`](compile_learn_or_both.pdf)
- **Full technical report** (31 pages, every table, the programmer brief, audits and protocol deviations):
  [`paper/main.pdf`](paper/main.pdf)

In weight-space programming (WSP), a frontier language model improves a smaller model by reading and editing its
weights directly, with no gradients and a budget of 50 forward passes. We compare it with LoRA fine-tuning on
Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct, on four five-option tasks chosen so that the knowledge a task
needs runs from present in the base model to absent:

| Task | Label | Knowledge in the base model |
|---|---|---|
| `sst5` | sentiment of a movie-review excerpt | present, read out correctly |
| `sst5inv` | the mirrored sentiment, same prompt | present, read out wrongly |
| `twin` | a real Twin-2K-500 respondent's survey answer | population tendencies only |
| `facts` | the birth city, job or hobby of an invented person | absent |

Main results on Qwen (test accuracy %; the unedited model scores 59.6 on MMLU):

| Task | Base | LoRA | Best WSP edit | MMLU after that edit |
|---|---|---|---|---|
| `sst5inv` | 4.0 | 49.3 (learns on 1 of 3 seeds) | 48.1 (embedding swap, through the model) | 54.8 |
| `sst5` | 44.9 | 59.0 | 48.0 (through) | 60.0 |
| `twin` | 30.9 | 42.8 | 45.0 (answer-prior circuit, bypass) | 23.4 |
| `facts` | 21.1 | 98.8 (15 epochs) | 100.0 (lookup circuit, bypass) | 22.2 |

- **Redirect.** When the model knows the answer but reads it out wrongly, every programmer found the same
  two-row embedding swap. It matched LoRA and was more reliable.
- **Compile.** When the answer is a table, programmers compiled the labels into circuits. Circuits that bypass
  the network score highest but drop MMLU to chance and next-token top-1 to 0–0.1%, on both models. Edits
  written through the model's own pathways keep it intact.
- **Learn.** When the task needs a better input-to-answer mapping, LoRA wins, even with ten times fewer labels.
- **The methods combine.** LoRA on top of a programmed edit reaches 99–100% on facts with a tenth of the data. On
  `sst5inv` it learns on every seed of both models (56–59%) with a third of the data LoRA alone needs, and a
  hand-written two-word swap works as well as the programmer's edit.

## Repository layout

| Path | Contents |
|---|---|
| `src/wsp/` | The library: run selection and paths, task builders, prompt format, letter scoring, the programmer's budgeted `Lab`, metrics |
| `scripts/` | Evaluation (task accuracy, option rotation, general abilities), the knowledge probe, LoRA training |
| `scripts/paper/` | Turns `results/` into the paper's numbers, tables and figures |
| `kaggle_runner/` | Runs commands on a private Kaggle GPU kernel and fetches the outputs |
| `briefs/` | The instructions every programmer agent received, verbatim |
| `runs/<model>/<task>/<run>/` | One folder per programmer run (see below) |
| `controls/` | Two control edits for `sst5inv`: a swap of two unrelated words, and the label-word swap alone |
| `results/<model>/<task>/` | Every task evaluation, option-rotation test and probe result; `results/<model>/general/` holds the general-ability evaluations |
| `paper/` | LaTeX source of the full technical report, generated tables and figures, `RESEARCH_LOG.md` (each observation, the experiment it led to, and the result, in order), and `data/` (run metadata and the collected numbers) |

Three environment variables select what a process works on: `WSP_MODEL` (`qwen` or `smollm`, default `qwen`),
`WSP_TASK` (`sst5`, `sst5inv`, `twin`, `facts`, `facts60`; default `twin`) and `WSP_RUN` (a programmer run). Every
path follows from them (`src/wsp/config.py`).

| Module | Role |
|---|---|
| `config.py` | Run selection and every derived path |
| `tasks/` | Builders for `twin`, `sst5` / `sst5inv` and `facts` / `facts60`; `build.py` builds them all |
| `prompts.py`, `scoring.py` | The prompt format shared by all tasks; model loading and scoring of the option letters |
| `harness.py` | The programmer's `Lab`: budgeted forward passes recorded in a persistent ledger |
| `loading.py` | Applying an edit, and loading a model with an edit, a LoRA adapter or both |
| `metrics.py`, `baselines.py` | Accuracy with bootstrap CI and NLL; chance and the train answer prior |
| `general_data.py` | WikiText-2 and MMLU for the general-ability checks |

## Programmer runs

Run names: `r1`–`r5` were programmed by Fable 5.2 (Twin `r1` by Fable 5.1), `o1` by Opus 5.5, `p1` by Fable 5.2
under the preserve addendum, and SmolLM2's `f1` by Fable 5.2. Each run folder holds:

| File | Contents |
|---|---|
| `edit/` | The programmer's edits. The final one is named in `paper/data/edits.json`, and every edit loads with the current code |
| `work/` | The programmer's scripts, small saved arrays and running `LOG.md` |
| `ledger.json` | Every budgeted forward pass: time, Kaggle job, split, size and the programmer's note |
| `prompt.txt` | The orchestrator's assignment (later runs only; local paths redacted) |
| `REPORT.md` | The programmer's final report, recovered from its transcript |
| `AUDIT.md` | The independent audit (18 runs) |

Four runs stopped part-way at a usage limit (`sst5/r2`, `sst5/r3`, `facts/r3`, `twin/r3`); they are kept but not
used in the results. Twin `r4` broke rule 1 of the brief and is reported as a protocol deviation (paper,
Appendix C) instead of in the main results. Twin `r1` predates the audits.

The scripts in `work/` are kept as the programmers wrote them. They use the repository layout and module names of
the time (`edits/<task>/<run>/`, `wsp_work/<task>_<run>/`, `results/<task>/<run>/wsp_ledger.json`,
`wsp.metrics.load_split`), so read them as a record rather than run them.

## Setup

```bash
uv sync
```

Python 3.12 or newer (the study used 3.13). `uv sync` also installs the Kaggle CLI and matplotlib. Build the task
data, which downloads the sources from Hugging Face (CPU, about a minute):

```bash
PYTHONPATH=src .venv/bin/python -m wsp.tasks.build
```

## Running on Kaggle

Every model computation in the study ran on a Kaggle T4 GPU (16 GB, fp16). Run any list of commands there with your
own Kaggle API credentials:

```bash
export KAGGLE_USERNAME=<your Kaggle user name>
kaggle_runner/run_job.sh <job> "<command>" ["<command>" ...]
```

`run_job.sh` bundles the code into the private kernel `<user>/wsp-<job>`, which rebuilds the data, runs the commands
and saves `results/` and `runs/`. It then waits for the kernel, downloads the outputs to
`kaggle_runner/out/<job>/out/`, and merges the forward-pass ledgers into `runs/`. Copy the result files you want to
keep into `results/`. Kaggle caps a kernel's source at 1 MB, so the bundle carries only the edit folders that the
commands select. Files over 200 KB are left out. The one edit constant that large (facts `o1`'s
`mem_s2_mem.pt`, 8.3 MB) and the LoRA adapters have to be attached as Kaggle datasets through `DATASET_SOURCES`.

Typical T4 times: a test-set evaluation takes from 12 s (facts) to 12 min (Twin), a general-ability evaluation about
a minute, and LoRA on 4,000 examples about 30 min (SST-5).

## Reproducing the numbers

The per-task pipeline, shown for SST-5:

```bash
WSP_TASK=sst5 python scripts/eval_model.py --split test --name base --baselines
WSP_TASK=sst5 python scripts/probe.py
WSP_TASK=sst5 python scripts/eval_permutation.py --split test --name base
WSP_TASK=sst5 python scripts/train_lora.py --max-examples 4000 --seed 0
WSP_TASK=sst5 python scripts/eval_model.py --split test --name lora --adapter checkpoints/qwen/sst5/lora
```

LoRA on facts needs 15 epochs (`--max-examples 40500`). A programmer's edit is selected by its run and file name.
General abilities are task-independent, so the edit is named by its path:

```bash
WSP_TASK=sst5inv WSP_RUN=r1 python scripts/eval_model.py --split test --name wsp_r1 --edit swap_steer_temp
python scripts/general_eval.py --name wsp_sst5inv_r1 --edit runs/qwen/sst5inv/r1/edit/swap_steer_temp
```

Program, then fine-tune: `train_lora.py --edit <edit>` trains the adapter on top of the edit and records it, and
evaluation takes both `--edit` and `--adapter`. The control edits are `--edit controls/random_swap` and
`--edit controls/swap_only`, the LoRA-alone controls on `sst5inv` are `--lr 3e-4` and `--max-examples 12000`, and
`WSP_MODEL=smollm` selects the second base model. The result names behind each number in the paper are in
`scripts/paper/collect.py`.

Rebuilding the paper's numbers, tables and figures from `results/` runs on the CPU:

```bash
.venv/bin/python scripts/paper/collect.py
.venv/bin/python scripts/paper/stats.py
.venv/bin/python scripts/paper/tables.py
.venv/bin/python scripts/paper/figures.py
cd paper && tectonic main.tex
```

`scripts/paper/cost.py` computes the API cost of each run from the agents' transcripts, which are not in the
repository; its output, `paper/data/costs.json`, is.

## Running a new programmer

A programmer gets `briefs/study.md` (and `briefs/preserve.md` for a preserve run) and a prompt like the
`prompt.txt` files, naming its task, run and paths. The brief's generic paths follow the layout of the time. With
this code the prompt should give `EDITS = runs/<model>/<task>/<run>/edit/`, `WORK = runs/<model>/<task>/<run>/work/`,
`LEDGER = runs/<model>/<task>/<run>/ledger.json`, and `BUNDLE_WORK=<model>/<task>/<run>` for Kaggle jobs. The
programmer works through the `Lab`:

```python
# every script runs with WSP_TASK=<task> WSP_RUN=<run>
from wsp.harness import Lab

lab = Lab(edit="my_edit")
result = lab.forward("train", list(range(512)), note="why this pass", capture_layers=(20, -1))
```

## Citing

Cite the paper; GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).

```bibtex
@techreport{bajaj2026compile,
  author      = {Bajaj, Puneet},
  title       = {Compile, Learn, or Both? When Weight-Space Programming Helps, What It Breaks, and Why Combining It with LoRA Matters},
  institution = {Working paper},
  year        = {2026},
  url         = {https://github.com/Puneet-Bajaj-IITM/wsp-vs-lora}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper, figures, results and run records are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
