# raw_data/ — put the corpus CSVs here

This is the **only** place you need to put data. Nothing else to configure.

```
raw_data/Bhili.csv      columns: English,Hindi,Bhili
raw_data/Gondi.csv      columns: English,Hindi,Gondi
raw_data/Mundari.csv    columns: English,Hindi,Mundari
raw_data/Santali.csv    columns: English,Hindi,Santali
```

One CSV per language, named after the language. Then:

```bash
python prepare_data.py
```

which writes, next to the code:

```
data/processed/D1_parallel.csv     15% held out — what the MT stage evaluates on
data/processed/D0_mlm.csv          85% for continued pretraining
data/processed/dataset_stats.json  row counts and provenance
```

Every other script finds those by default, so there is no path to edit anywhere.

## Notes

- The header is matched case-insensitively, and a UTF-8 BOM is tolerated.
- A `Hindi` column is optional; this stage translates tribal→English and never
  reads it. If absent, the field is left empty in D1.
- Rows with an empty English or tribal side are dropped from D1 and counted in
  the run summary.
- The split is 85/15 per language at seed 42, reproducing the partition used for
  the published experiments. `prepare_data.py` explains the three details that
  keep it byte-identical.
- Re-running is a no-op unless you pass `--force`. `--verify` inspects existing
  outputs without touching them.

## If your data lives somewhere else

You can point at an existing prepared file instead, in which case this directory
is unused:

```bash
export TRIBAL_DATA=/path/to/D1_parallel.csv
```

or set `data.path` in `config.yaml`.
