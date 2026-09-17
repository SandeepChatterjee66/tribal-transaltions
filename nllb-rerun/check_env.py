#!/usr/bin/env python3
"""
check_env.py — run this FIRST on the GPU server.

Verifies, in order and without downloading model weights:
  1. Python package versions
  2. GPU / CUDA / bf16 availability
  3. The tri-parallel CSV can be found, loaded and split
  4. sacrebleu can do FLORES spBLEU (needs sentencepiece)
  5. Every configured language token exists in each model's tokenizer

Step 5 downloads tokenizers only (a few MB each), not model weights, so it is
cheap and catches the mistake that otherwise surfaces an hour into a run.

  python check_env.py
  python check_env.py --data /scratch/tribal/D1_parallel.csv
  python check_env.py --models nllb-600m nllb-1.3b --skip_tokenizers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(status: str, msg: str) -> None:
    print(f"[{status}] {msg}")


def parse_args():
    p = argparse.ArgumentParser(description="Preflight for the MT experiments")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--models", nargs="+", default=None,
                   help="default: every model in config.yaml")
    p.add_argument("--skip_tokenizers", action="store_true",
                   help="skip step 5 (no network / no HF cache)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    problems: list[str] = []

    print("=" * 66)
    print(" TRIBALSUITE MT — ENVIRONMENT CHECK")
    print("=" * 66)

    # ---- 1. packages ----
    print("\n[1] packages")
    required = {
        "torch": "2.0", "transformers": "4.40", "datasets": "2.19",
        "peft": "0.11", "sacrebleu": "2.4", "pandas": "2.0",
        "numpy": "1.24", "yaml": None, "sentencepiece": None,
        "rouge_score": None, "accelerate": "0.30",
    }
    for pkg, minver in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            line(OK, f"{pkg:<16} {ver}" + (f"  (need >= {minver})" if minver else ""))
        except ImportError:
            line(BAD, f"{pkg:<16} MISSING")
            problems.append(f"pip install {pkg}")

    # ---- 2. GPU ----
    print("\n[2] compute")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                line(OK, f"GPU {i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
            bf16 = torch.cuda.is_bf16_supported()
            line(OK if bf16 else WARN,
                 f"bf16 supported: {bf16}" +
                 ("" if bf16 else "  (will use fp16 — fine, slightly less stable)"))
            free, total = torch.cuda.mem_get_info()
            line(OK if free / 1e9 > 12 else WARN,
                 f"free GPU memory: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
            if free / 1e9 < 12:
                line(WARN, "under ~12 GB free: use nllb-600m, or lower "
                           "finetune.batch_size / turn on gradient_checkpointing")
        else:
            line(WARN, "no CUDA — everything runs on CPU, which is far too slow "
                       "for fine-tuning (inference on a small sample is OK)")
    except ImportError:
        line(BAD, "torch missing — cannot check GPU")

    # ---- 3. data ----
    print("\n[3] data")
    cfg = None
    try:
        import common as C

        cfg = C.load_config(args.config)
        C.setup_logging(C._anchor("logs"), "check_env")

        raw = C.raw_data_dir(cfg)
        raw_files = sorted(p.name for p in raw.glob("*.csv")) if raw.is_dir() else []
        line(OK if raw_files else WARN,
             f"raw_data: {raw}  ({len(raw_files)} csv)" if raw_files
             else f"raw_data: {raw} is empty or missing")

        try:
            path = C.find_data(cfg, args.data)
        except FileNotFoundError:
            # The usual cause is that prepare_data.py has not run yet. Say so
            # concretely instead of reporting a generic missing-file error.
            if raw_files:
                line(BAD, "D1_parallel.csv not generated yet")
                problems.append("python prepare_data.py   "
                                "(raw data is present, just needs preparing)")
            else:
                line(BAD, f"no raw data and no D1_parallel.csv")
                problems.append(f"put the corpus CSVs in {raw}/ then run "
                                f"python prepare_data.py")
            raise SystemExit(summary(problems))
        line(OK, f"found: {path}")
        df = C.load_parallel(path, C.deep_get(cfg, "data.languages"))
        line(OK, f"loaded {len(df):,} pairs")
        df = C.make_splits(df, cfg)
        for split in ("train", "dev", "test"):
            n = int((df["split"] == split).sum())
            line(OK, f"{split:<5} {n:,}")
        overlap = (set(df[df.split == "train"].gid)
                   & set(df[df.split == "test"].gid))
        line(OK if not overlap else BAD,
             f"train/test gid overlap: {len(overlap)} (must be 0)")
        if overlap:
            problems.append("train/test overlap — do not report these numbers")
    except Exception as e:
        line(BAD, f"{type(e).__name__}: {e}")
        problems.append("fix the data path (--data, $TRIBAL_DATA, or config.yaml)")

    # ---- 4. metrics ----
    print("\n[4] metrics")
    try:
        import common as C

        tok = C._spbleu_tokenizer()
        line(OK if tok.startswith("flores") else BAD,
             f"spBLEU tokenizer: {tok}" +
             ("" if tok.startswith("flores")
              else "  <-- NOT comparable to the paper"))
        if not tok.startswith("flores"):
            problems.append("pip install sentencepiece   (needed for spBLEU)")
        m = C.compute_metrics(["the cat sat"], ["the cat sat down"])
        line(OK, f"chrF++/spBLEU/ROUGE-L all compute: {m}")
    except Exception as e:
        line(BAD, f"{type(e).__name__}: {e}")
        problems.append("metrics are broken — check sacrebleu / rouge-score")

    # ---- 5. tokenizers + language tokens ----
    print("\n[5] model tokenizers and language tokens")
    if args.skip_tokenizers:
        line(WARN, "skipped (--skip_tokenizers)")
    elif cfg is None:
        line(BAD, "skipped — config did not load")
    else:
        import common as C

        models = args.models or list(C.deep_get(cfg, "models", {}))
        languages = C.deep_get(cfg, "data.languages")
        for key in models:
            try:
                spec = C.resolve_model(cfg, key)
                from transformers import AutoTokenizer

                trust = spec.family == "indictrans2"
                tk = AutoTokenizer.from_pretrained(
                    spec.hf_id, trust_remote_code=trust)
                need = {spec.target_token}
                for lang in languages:
                    need.add(spec.source_tokens[lang])
                bad = []
                for t in sorted(need):
                    try:
                        C.resolve_lang_token_id(tk, t)
                    except ValueError:
                        bad.append(t)
                if bad:
                    line(BAD, f"{key:<18} unknown tokens: {bad}")
                    problems.append(
                        f"fix lang_tokens.{spec.family} in config.yaml: {bad}")
                else:
                    line(OK, f"{key:<18} {len(need)} language tokens valid "
                             f"({spec.params})")
            except Exception as e:
                line(WARN, f"{key:<18} could not check: "
                           f"{type(e).__name__}: {str(e)[:110]}")

    # ---- summary ----
    return summary(problems)


def summary(problems):
    print("\n" + "=" * 66)
    if problems:
        print(" NOT READY — fix these first:")
        for p in dict.fromkeys(problems):
            print(f"   - {p}")
        print("=" * 66)
        return 1
    print(" READY. Next:")
    print("   bash run_zeroshot.sh --smoke      # ~2 min, proves the loop works")
    print("   bash run_zeroshot.sh              # full zero-shot table")
    print("   bash run_finetune.sh --smoke      # ~5 min, proves training works")
    print("   bash run_finetune.sh              # real fine-tuning")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
