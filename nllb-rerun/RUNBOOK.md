# RUNBOOK — start here

Ignore every other file. Do these steps in order. Copy-paste each block.

---

## Step 1 — Get the folder onto the GPU server

```bash
rsync -av nllb-rerun/ user@server:~/nllb-rerun/
ssh user@server
cd ~/nllb-rerun
```

---

## Step 2 — Install

```bash
# torch first, matched to the server's CUDA (check with: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Step 3 — Put the data in `raw_data/`

One CSV per language, named after the language:

```
raw_data/Bhili.csv      columns: English,Hindi,Bhili
raw_data/Gondi.csv      columns: English,Hindi,Gondi
raw_data/Mundari.csv    columns: English,Hindi,Mundari
raw_data/Santali.csv    columns: English,Hindi,Santali
```

That is the only place data goes, and there is **no path to configure anywhere**.
All paths are resolved relative to this directory, so the folder can be moved,
copied or rsynced to another machine and still work.

`run_all.sh` prepares the data itself if needed. To do it explicitly:

```bash
python prepare_data.py            # -> data/processed/D1_parallel.csv
python prepare_data.py --verify   # inspect what was generated
```

That splits 85/15 per language at seed 42 — the partition used for the published
experiments — writing `data/processed/D1_parallel.csv` (15%, held out, what this
stage evaluates on) and `data/processed/D0_mlm.csv` (85%, continued pretraining).

<details>
<summary>Using a prepared file from somewhere else</summary>

```bash
export TRIBAL_DATA=/path/to/D1_parallel.csv
```

or set `data.path` in `config.yaml`. Then `raw_data/` is unused.
</details>

---

## Step 4 — Run everything

```bash
bash run_all.sh --bg
```

`--bg` detaches the job and immediately starts tailing it, so you watch it live
but **Ctrl-C only stops watching — the job keeps running**. It also survives a
dropped SSH connection, so no `nohup`/`tmux` needed.

Re-attach at any time from any shell:

```bash
bash watch.sh
```

(Plain `bash run_all.sh` without `--bg` runs in the foreground, if you prefer.)

That is the whole job. It runs, in order:

| | Step | Roughly |
|---|---|---|
| 0 | environment check | seconds |
| 1 | offline self-test (no GPU, no downloads) | 2–3 min |
| 2 | zero-shot inference, all models | 1–3 h |
| 3 | LoRA fine-tune + inference | 3–8 h |
| 4 | scoring | minutes |
| 5 | consolidate into CSVs | seconds |

**If it dies at any point, just run `bash run_all.sh` again.** Finished work is
skipped; it picks up where it stopped.

---

## Step 5 — Collect the results

```bash
column -t -s$'\t' consolidated/paper_table.tsv      # the table for the paper
column -t -s$'\t' consolidated/zeroshot_vs_ft.tsv   # zero-shot vs FT + deltas
column -t -s$'\t' consolidated/run_manifest.tsv     # did anything fail?
```

Copy `consolidated/` back to your laptop:

```bash
rsync -av user@server:~/nllb-rerun/consolidated/ ./consolidated/
```

### Check one column before you believe any number

In `consolidated/all_scores.tsv`, look at **`empty_pct`**. It is the share of
outputs that came back blank. If it is above ~2% for a language, that score is
being dragged down by generation failures, not by translation quality. Re-run:

```bash
python infer.py --model nllb-600m        # only redoes the missing rows
python evaluate.py --all --name mt_all
python consolidate.py
```

---

# That is it. Everything below is optional.

---

## Running step by step instead of run_all.sh

`run_all.sh` just calls these six commands in order. Run them yourself if you
want to stop and look between stages. Every command is independently
re-runnable and skips finished work.

```bash
export TRIBAL_DATA=/full/path/to/D1_parallel.csv    # do this once
M=nllb-600m                                          # the model you are running
```

| # | Step | Command | Writes |
|---|---|---|---|
| 1 | Check the environment | `python check_env.py` | `logs/check_env.log` |
| 2 | Offline self-test | `python selftest.py` | `.selftest/` (delete after) |
| 3 | Zero-shot translate | `python infer.py --model $M` | `runs/$M/zeroshot/predictions.csv` |
| 4 | Fine-tune (LoRA) | `python finetune.py --model $M` | `runs/$M/finetune/best/` |
| 5 | Translate with the checkpoint | `python infer.py --model $M --ckpt runs/$M/finetune/best --tag ft_lora` | `runs/$M/ft_lora/predictions.csv` |
| 6 | Score everything | `python evaluate.py --all --name mt_all` | `tables/mt_all.{tsv,csv,tex}` |
| 7 | Consolidate | `python consolidate.py` | `consolidated/*.tsv` + `.csv` |

Notes:

- Steps 3 and 4 are the only slow ones. Step 3 is safe to run first and look at
  the numbers before committing to step 4.
- Repeat steps 3–5 for each model, then run 6 and 7 once at the end.
- Always run step 6 before step 7 — `consolidate.py` reads the `metrics.json`
  that `evaluate.py` writes.
- Add `--limit 100` to any `infer.py` call for a fast look. It warns, and those
  numbers must not be reported.

### The two other fine-tuning variants

Same shape as steps 4–5, with a different tag so they land in their own rows:

```bash
# full fine-tune
python finetune.py --model $M --tag finetune_full --set finetune.mode=full finetune.lr=3e-5
python infer.py --model $M --ckpt runs/$M/finetune_full/best --tag ft_full

# encoder frozen, decoder only
python finetune.py --model $M --tag finetune_frozen \
    --set finetune.mode=full finetune.freeze_encoder=true finetune.lr=3e-5
python infer.py --model $M --ckpt runs/$M/finetune_frozen/best --tag ft_frozen

# then re-score and re-consolidate
python evaluate.py --all --name mt_all
python consolidate.py
```

After that, `consolidated/paper_table.tsv` has one row per model per variant:
`zeroshot`, `ft_lora`, `ft_full`, `ft_frozen`.

---

## Watching it, and finding old logs

One command for everything:

```bash
bash watch.sh              # tail the newest log — follows a running job
bash watch.sh --list       # every run ever: when, what, duration, status
bash watch.sh --stats      # live dashboard: loss, eval_loss, ETA, throughput
bash watch.sh --errors     # every crash, with its traceback
bash watch.sh --files      # where all the logs are
bash watch.sh nllb-1.3b    # re-read a specific old log by name
```

`bash watch.sh --list` looks like this:

```
WHEN                 SCRIPT      WHAT                     MIN  STATUS   LOG
2026-09-16 14:01:03  finetune    nllb-600m_finetune     184.2  ok       20260916_140103_finetune_nllb-600m_finetune.log
2026-09-16 13:52:11  infer       nllb-600m_zeroshot      47.9  ok       20260916_135211_infer_nllb-600m_zeroshot.log
2026-09-16 13:40:02  infer       nllb-1.3b_zeroshot         -  running  20260916_134002_infer_nllb-1.3b_zeroshot.log
```

### Where logs are kept

Nothing is ever overwritten. Every invocation gets its own timestamped file:

```
logs/
  index.csv                                        every run: time, duration, status, command
  latest.log -> <newest>                           what `watch.sh` tails
  20260916_140103_finetune_nllb-600m_finetune.log  one file per invocation
  20260916_135211_infer_nllb-600m_zeroshot.log
  run_all_20260916_133000.log                      the --bg master log
  run_all.pid                                      kill $(cat logs/run_all.pid)

runs/<model>/<setting>/
  finetune.log / infer.log      appended across invocations — full history of this run dir
  training_log.csv              the loss curve, written live
  error.log                     only if it crashed
```

So `logs/` answers "what did I run last week and did it finish?" and
`runs/<model>/<setting>/` answers "what is the whole story of this experiment?".

---

## Training longer without restarting

Say you trained 3 epochs and want 6, then 9. Do **not** re-run the plain
command — it will see the finished run and skip. Use `--extend`:

```bash
python finetune.py --model $M                          # 3 epochs (config default)
python finetune.py --model $M --extend --epochs 6      # continues 3 -> 6
python finetune.py --model $M --extend --epochs 9      # continues 6 -> 9
```

It restores weights, optimiser state, LR scheduler and data order from the last
checkpoint, so epoch 4 starts where epoch 3 stopped. You will see this in the
log, which is the proof it is not starting over:

```
RESUMING from .../checkpoints/checkpoint-5019 (step 5019)
RESUMING at step 5019 of 10038 (50.0% already done) — not restarting from scratch
--extend: raising the budget to 6 epochs and continuing from that step.
```

Then re-run inference and scoring on the longer-trained checkpoint:

```bash
python infer.py --model $M --ckpt runs/$M/finetune/best --tag ft_lora_9ep
python evaluate.py --all --name mt_all
python consolidate.py
```

Use a **different `--tag`** per epoch budget (`ft_lora_3ep`, `ft_lora_6ep`, …) if
you want all of them side by side in the final table. Reuse the same tag if you
only care about the latest.

Two honest caveats:

- The LR schedule is rebuilt for the new total step count, so 3→6→9 by extension
  is not bit-identical to a single 9-epoch run. The tail of the curve differs
  slightly. Fine to report, but say which you did.
- Early stopping (patience 3 evals) may halt before the new budget. Check
  `early_stopped` in `train_metrics.json`. Raise
  `finetune.early_stopping_patience` if you want it to push on regardless.

---

## Watching it run, and the statistics you get

Every 25 steps (or every step with `--verbose`) training prints:

```
step 1200/5019 (24%)  epoch 0.72  loss=2.1834  lr=2.41e-04
    38.2 samp/s  1834 tok/s  mem 18.3GB  elapsed 12.4m  ETA 39.1m
```

Inference prints, per language:

```
  3200/4500  ( 71.1%) |   14.2 sent/s | elapsed  3.8m | ETA  1.5m | 9.4GB
```

**Live files, written as it goes** — tail or plot them mid-run:

| File | Contents |
|---|---|
| `runs/$M/finetune/training_log.csv` | one row per logging step: step, epoch, loss, eval_loss, lr, grad_norm, samples/s, tokens/s, GPU GB, elapsed, ETA |
| `runs/$M/finetune/train_metrics.json` | ~42 fields, written at the end (below) |
| `runs/$M/<setting>/inference_stats.json` | per-language decode timing + throughput |
| `runs/$M/finetune/logs/` | TensorBoard: `tensorboard --logdir runs` |

`train_metrics.json` covers: epochs requested/completed, steps vs planned,
`early_stopped`, `resumed_from`, `extended`, examples and approximate tokens
seen, effective batch, trainable params, precision, wall-clock, sec/step,
examples/s, mean samples/s and tokens/s, peak GPU memory, initial/final/min
loss, total loss reduction, loss variance and std, **step at which 90% of the
loss drop was achieved**, best eval loss and the step it occurred at.

Those last few deliberately match the convergence / loss-variance / efficiency
tables the paper already reports for the encoders, so the MT rows can be
presented the same way.

After `consolidate.py` these become three flat CSVs:

```
consolidated/training_stats.csv    one row per training run
consolidated/loss_curves.csv       every logged point, all runs — plot from this
consolidated/inference_stats.csv   per-language decode cost
```

Verbosity knobs:

```bash
python finetune.py --model $M --verbose        # log every step
# or in config.yaml:
#   finetune.logging_steps: 25        <- lower = more chatter
#   infer.log_every_sentences: 200    <- lower = more chatter
```

---

## Which file do I edit?

Short answer: **`config.yaml`, and nothing else.** The rest you only read.

| File | What it is | Would you edit it? |
|---|---|---|
| **`config.yaml`** | Every setting: models, language tokens, split, hyperparameters, paths | **Yes — this is the one** |
| `RUNBOOK.md` | This file | no |
| `README.md` | Why the design is the way it is | no |
| `requirements.txt` | Dependencies | only to pin a version |
| `run_all.sh` | Calls steps 1–7 in order | no |
| `run_zeroshot.sh` | Steps 1+3+6 for all models | no |
| `run_finetune.sh` | Steps 4+5+6 for all models | no |
| `check_env.py` | Verifies deps, GPU, data, metrics, language codes | no |
| `selftest.py` | Offline end-to-end test on a fake tiny model | no |
| `infer.py` | Translation — zero-shot and fine-tuned both | only to change decoding logic |
| `finetune.py` | Training: LoRA / full / frozen-encoder | only to change the training loop |
| `evaluate.py` | chrF++, spBLEU, ROUGE-L, bootstrap CIs | only to add a metric |
| `consolidate.py` | Merges all runs into `consolidated/*.csv` | only to change the output layout |
| `common.py` | Shared helpers: data loading, split, metrics, logging | no |
| `indictrans2.py` | IndicTrans2-only pre/post-processing | no |

### What to change in `config.yaml`, by situation

| I want to... | Change |
|---|---|
| point at the data permanently | `data.path` (or just use `$TRIBAL_DATA`) |
| run fewer/more models | `MODELS="..."` env var, or add an entry under `models:` |
| fix a wrong language code | `lang_tokens.<family>.source.<Language>` |
| change the proxy candidates for the sweep | `lang_tokens.<family>.sweep.<Language>` |
| switch LoRA → full fine-tune | `finetune.mode: full` **and** `finetune.lr: 3e-5` |
| train the decoder only | `finetune.freeze_encoder: true` |
| fix CUDA OOM in training | halve `finetune.batch_size`, double `finetune.grad_accum` |
| still OOM | `finetune.gradient_checkpointing: true` |
| train longer / shorter | `finetune.epochs` |
| one model per language | `finetune.per_language: true` (or `--per_language`) |
| faster, rougher decoding | `infer.num_beams: 1` |
| smaller test set for a quick look | `data.max_test_per_language: 500` |
| skip bootstrap CIs | `metrics.bootstrap_rounds: 0` |

You can also override any config key from the command line without editing the
file, which is safer for one-offs:

```bash
python finetune.py --model $M --set finetune.mode=full finetune.lr=3e-5
```

### Where the outputs live

```
runs/<model>/<setting>/       one dir per model per setting
    predictions.csv             gid, language, tribal, english, hyp, src_lang
    metrics.json                scores (written by evaluate.py)
    infer.log / finetune.log    full run log  <- send me this if it breaks
    error.log                   traceback, only present if it crashed
    environment.json            node, GPU, package versions
    best/                       the fine-tuned checkpoint (training runs only)
    by_group/<lang>/            per-language partials — this is what enables resume

tables/         evaluate.py output: one table per --name
consolidated/   consolidate.py output: the CSVs you actually read
logs/           stdout from the run_*.sh drivers
```

`<setting>` is `zeroshot`, `ft_lora`, `ft_full`, `ft_frozen`, `sweep`, or one of
the training dirs `finetune`, `finetune_full`, `finetune_frozen`.

---

## Optional A — Just try it quickly first

If you want to see the whole thing work in ~10 minutes before committing hours:

```bash
bash run_zeroshot.sh --smoke
```

50 sentences per language, greedy decoding. **Do not report these numbers.**

---

## Optional B — The other two fine-tuning variants

Run these *after* the LoRA numbers look sensible. Each lands in its own row of
the consolidated table, so you can compare and report whichever is best.

```bash
# full fine-tune (slower, more memory, usually the strongest)
MODE=full LR=3e-5 bash run_finetune.sh

# decoder only, encoder frozen (fastest of the three)
FREEZE_ENCODER=1 MODE=full LR=3e-5 bash run_finetune.sh

# then refresh the tables
python evaluate.py --all --name mt_all
python consolidate.py
```

---

## Optional C — Justify the source-language token choice

Read this if a reviewer asks why Bhili was fed to NLLB as Hindi.

NLLB needs a source-language tag, and **none of Bhili, Gondi or Mundari exist in
NLLB-200**, so they use a Devanagari stand-in. Santali is worse than it looks:
NLLB's only Santali is `sat_Beng` (Bengali script), while TribalCorp's Santali is
Ol Chiki. IndicTrans2 is the only model here that has `sat_Olck`.

To show the stand-in was not cherry-picked, sweep it on the **dev** split:

```bash
SWEEP=1 bash run_all.sh
cat consolidated/src_token_sweep.csv     # ranked per language
```

Pick on dev, report on test. Choosing it on test is the kind of thing a reviewer
will catch.

---

## Optional D — Add IndicTrans2

Worth doing: it is already cited in the paper, so an Indic reviewer will ask.

```bash
pip install IndicTransToolkit
MODELS="indictrans2-200m indictrans2-1b" bash run_all.sh
```

---

## Optional E — Choose which models to run

```bash
MODELS="nllb-600m" bash run_all.sh                  # just the small one
MODELS="nllb-600m nllb-1.3b mbart50" bash run_all.sh
```

Available: `nllb-600m`, `nllb-1.3b`, `nllb-1.3b-dense`, `mbart50`,
`m2m100-418m`, `m2m100-1.2b`, `indictrans2-200m`, `indictrans2-1b`.

---

## When something breaks

The error message usually tells you the fix. If not, three files have the answer:

```bash
cat runs/<model>/<setting>/error.log        # the traceback
cat runs/<model>/<setting>/infer.log        # the full run + resolved config
cat runs/<model>/<setting>/environment.json # node, GPU, package versions
```

Send those three and the problem is almost always diagnosable without server
access.

| Message | Fix |
|---|---|
| `Could not find the tri-parallel CSV` | `export TRIBAL_DATA=/full/path.csv` — the error lists every path it tried |
| `Language token 'xxx' is not in this tokenizer` | typo in `config.yaml`; the error prints the valid codes |
| `CUDA out of memory` during training | edit `config.yaml`: halve `finetune.batch_size`, double `finetune.grad_accum` |
| still out of memory | set `finetune.gradient_checkpointing: true` |
| out of memory during inference | nothing to do — it halves the batch and retries by itself |
| `tensorboard not installed` warning | harmless; `pip install tensorboard` if you want loss curves |
| `spBLEU will NOT be comparable` warning | **not** harmless: `pip install sentencepiece`, then re-score |
| Everything is slow | `MODELS="nllb-600m" bash run_all.sh` to start with the small model |

---

## What the files are, if you are curious

You do not need this to run anything.

| File | Purpose |
|---|---|
| `run_all.sh` | the one command — calls everything below in order |
| `config.yaml` | every setting; the only file you would normally edit |
| `check_env.py` | verifies deps, GPU, data, metrics, language codes |
| `selftest.py` | offline end-to-end test on a tiny fake model |
| `infer.py` | translation (zero-shot and fine-tuned both) |
| `finetune.py` | LoRA / full / frozen-encoder training |
| `evaluate.py` | chrF++, spBLEU, ROUGE-L, bootstrap CIs |
| `consolidate.py` | merges every run into `consolidated/*.csv` |
| `common.py` | shared helpers |
| `indictrans2.py` | IndicTrans2-specific pre/post-processing |
| `run_zeroshot.sh`, `run_finetune.sh` | the individual stages, if you want one alone |
| `watch.sh` | tail live logs, list past runs, live training dashboard |
| `README.md` | the detailed explanation of design decisions |
