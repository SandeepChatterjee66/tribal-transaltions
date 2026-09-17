# TribalSuite — NLLB and open-weight seq2seq MT

Zero-shot and fine-tuned tribal→English translation for **NLLB-200**, **mBART-50**,
**M2M-100** and **IndicTrans2**. Answers the reviewer ask carried over from both
rounds: *"use some open-source models like NLLB and finetune on your data as well."*

Self-contained — `rsync` this folder to the server and go.

---

## Quick start

```bash
# 0. one-time setup
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match server CUDA
pip install -r requirements.txt

# 1. tell it where the tri-parallel CSV is (once per shell)
export TRIBAL_DATA=/path/on/server/D1_parallel.csv

# 2. preflight — checks deps, GPU, data, metrics, language tokens
python check_env.py

# 3. prove the loop works before spending GPU hours
bash run_zeroshot.sh --smoke      # ~2 min
bash run_finetune.sh --smoke      # ~5 min

# 4. the real runs
bash run_zeroshot.sh              # -> tables/zeroshot_all.{tsv,csv,tex}
bash run_finetune.sh              # -> tables/mt_all.{tsv,csv,tex}
```

Every script is **resume-safe**: re-run the identical command after a crash,
preemption or timeout and it continues from the last checkpoint.

---

## Layout

| File | What it does |
|---|---|
| `config.yaml` | **Every knob.** Models, language tokens, split, hyperparameters. Usually the only file you edit. |
| `check_env.py` | Preflight. Run first, on every new node. |
| `common.py` | Shared plumbing: data discovery, splits, metrics, logging, crash handling. |
| `infer.py` | Tribal→English decoding. Same script for zero-shot and fine-tuned. |
| `finetune.py` | LoRA or full seq2seq fine-tuning. |
| `evaluate.py` | Scores predictions, writes TSV/CSV/LaTeX. |
| `run_zeroshot.sh` / `run_finetune.sh` | Drivers over all models. |

Outputs:

```
runs/<model>/<setting>/
  predictions.csv        gid, language, tribal, english, hyp, src_lang
  metrics.json           per-language + macro, with bootstrap CIs
  infer.log              full run log  <- send me this if something breaks
  error.log              traceback, only if it crashed
  environment.json       node, GPU, package versions
  by_group/<lang>/       per-language partials (this is what makes resume work)
tables/
  *.tsv *.csv *.tex      paste-ready
```

---

## The split

`prepare_data.py` in the main repo produces D0 (85%, CPT) and D1 (15%, held out).
This harness takes D1 and splits it **per language, 70/10/20, seed 42** — the
same proportions and seed the encoder experiments use.

```
train 53,538   dev 7,648   test 15,297
Bhili 15,750/2,250/4,500   Gondi 11,080/1,583/3,166
Mundari 16,208/2,315/4,631  Santali 10,500/1,500/3,000
```

Two properties worth knowing, because they are what make the table defensible:

1. **Zero-shot and fine-tuned are scored on the identical test rows.** Both go
   through `infer.py`, which derives the split from the same seeded function.
   The earlier MT runs compared conditions evaluated on different subsets of
   different sizes; this cannot.
2. **Fine-tuning never sees test.** `check_env.py` asserts zero gid overlap.

Cap the test set for a quick look with `--limit 200`; it warns loudly and you
must not report those numbers.

---

## NLLB does not cover three of the four languages

This is the single most important thing to get right, and it is a *result*, not
an obstacle:

| Language | In NLLB-200? | Token used |
|---|---|---|
| Santali (sat) | **yes** — `sat_Olck` | native |
| Bhili (bhb) | no | `hin_Deva` (proxy) |
| Gondi (gon) | no | `hin_Deva` (proxy) |
| Mundari (unr) | no | `hin_Deva` (proxy) |

NLLB needs a source-language token in its input, so the three unsupported
languages need a stand-in. Defaults follow the script actually used in
TribalCorp. To show the choice was not cherry-picked, sweep it on **dev**:

```bash
python infer.py --model nllb-600m --sweep_src --split dev --limit 300
python evaluate.py --run runs/nllb-600m/zeroshot --by_group
```

That prints the best proxy token per language and writes
`tables/nllb-600m_zeroshot_src_sweep.tsv`. Use dev for the sweep and report
test — picking the token on test is the kind of thing a reviewer will catch.

Candidates live under `lang_tokens.nllb.sweep` in `config.yaml`; edit freely.

---

## Fine-tuning: what the defaults do and why

Default is **LoRA with frozen embeddings**, pooled over all four languages.

- NLLB's embedding matrix is ~256k × d — the largest parameter block by far.
  Freezing it plus LoRA on attention cuts time and memory sharply at
  comparable quality, and the target language (English) is already well covered.
- Pooled (one model, four languages) is standard multilingual MT and 4× cheaper
  than per-language. Per-language matches the encoder protocol more literally:

```bash
PER_LANGUAGE=1 bash run_finetune.sh
```

- Full fine-tune, if you have the budget:

```bash
MODE=full LR=3e-5 bash run_finetune.sh
```

Speed levers, in the order to reach for them:

| Symptom | Change |
|---|---|
| CUDA OOM during training | `finetune.batch_size` down, `grad_accum` up to compensate |
| still OOM | `finetune.gradient_checkpointing: true` |
| OOM during inference | nothing — `infer.py` halves the batch and retries automatically |
| too slow | keep `group_by_length: true`; drop `num_beams` to 1 for dev-set work |
| just want a number today | `--set data.max_test_per_language=500` |

---

## Reading the output

`evaluate.py` prints an `empty_%` column beside every score and warns above 2%.
That column counts blank and `ERROR` outputs. It is there deliberately: a
baseline scored with a high silent failure rate looks weak for reasons that
have nothing to do with translation quality, and that is exactly how a
misleading zero-shot number gets into a table. If `empty_%` is non-trivial,
re-run before reporting — `infer.py` will only redo the missing rows.

`chrf++_ci` is a percentile bootstrap CI (1000 rounds) on chrF++, which covers
the "add variance across seeds" ask more cheaply than multi-seed reruns. It
resamples precomputed per-segment statistics, so it costs seconds, not an hour.
Disable with `--no_bootstrap`.

chrF++ is primary, matching the paper. spBLEU uses the FLORES-200 SPM
tokenizer — if `sentencepiece` is missing it falls back and warns that the
numbers are no longer comparable. Do not ignore that warning.

---

## Other models worth showing alongside NLLB

Ranked by what they add to the paper:

1. **IndicTrans2** (`indictrans2-200m`, `indictrans2-1b`) — the strongest and
   most *expected* Indic baseline. It is already cited in the related-work
   section, so an Indic reviewer will wonder why it is not in the table. Needs
   `pip install IndicTransToolkit`; the other models need nothing extra.
2. **NLLB-200** (`nllb-600m`, `nllb-1.3b`, `nllb-1.3b-dense`) — asked for by
   name, and the 600M/1.3B pair gives a clean capacity axis under 2B.
3. **mBART-50 many-to-one** (`mbart50`, 611M) — cheap, standard, and makes
   "open-weight seq2seq" a claim about a family rather than one model.
4. **M2M-100** (`m2m100-418m`, `m2m100-1.2b`) — older; include only if you want
   a second capacity axis.

One genuinely interesting option not in the registry: **ByT5** (`byt5-base`,
582M). Byte-level, so it sidesteps the subword-fragmentation problem the paper
already measures in the fertility tables. It needs a different prompt setup
(no language tokens) so it is not a drop-in, but it is the most defensible
"why not just fix the tokenizer" answer if a reviewer pushes there.

MADLAD-400 is excluded: the smallest checkpoint is 3B, over the <2B budget.

Adding a model is one entry under `models:` plus one under `lang_tokens:` for
its family — nothing else changes.

---

## When something breaks

1. `runs/<model>/<setting>/error.log` — the traceback.
2. `runs/<model>/<setting>/infer.log` (or `finetune.log`) — the full run,
   including the resolved config.
3. `runs/<model>/<setting>/environment.json` — node, GPU, package versions.

Those three files are enough to diagnose almost anything without server access.

Common cases:

| Error | Fix |
|---|---|
| `Could not find the tri-parallel CSV` | `export TRIBAL_DATA=/abs/path.csv` — the message lists every path it tried |
| `Language token 'xxx_Yyyy' is not in this tokenizer` | typo in `config.yaml`; the message prints the valid codes |
| `None of the configured LoRA target_modules exist` | that model names its projections differently; the message lists candidates |
| `OOM even at batch_size=1` | use a smaller model or lower `infer.max_source_length` |
| Run died at 90% | re-run the same command; it resumes |
