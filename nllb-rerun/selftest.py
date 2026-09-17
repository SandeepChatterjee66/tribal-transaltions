#!/usr/bin/env python3
"""
selftest.py — prove the pipeline works before spending GPU time.

Builds a tiny randomly-initialised NLLB-architecture model plus a small
SentencePiece tokenizer trained on your own corpus, then runs the real
infer -> finetune -> infer -> evaluate -> consolidate path end to end on a
few hundred rows. Completely offline: no model downloads, no GPU needed.

The translations are gibberish (random weights) — that is fine and expected.
What this proves is that every moving part actually works on THIS machine with
THIS version of transformers: tokenizer plumbing, language tokens, the data
split, LoRA attach, the collator, checkpoint save/resume, adapter merge,
metrics, and table generation.

    python selftest.py                      # ~1-3 min on CPU
    python selftest.py --keep               # leave .selftest/ for inspection
    python selftest.py --data /path.csv

Exit code 0 = safe to launch the real runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / ".selftest"

PASS, FAIL, INFO = "  PASS  ", "  FAIL  ", "  ..    "
_results: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    _results.append((name, ok, detail))
    return ok


def parse_args():
    p = argparse.ArgumentParser(description="Offline end-to-end self-test")
    p.add_argument("--data", default=None, help="tri-parallel CSV")
    p.add_argument("--keep", action="store_true",
                   help="keep .selftest/ afterwards for inspection")
    p.add_argument("--rows", type=int, default=80,
                   help="rows per language to use (default 80)")
    return p.parse_args()


# ============================================================
# BUILD A TINY OFFLINE MODEL
# ============================================================

def build_tiny_model(corpus_texts: list[str], out_dir: Path) -> Path:
    """Train a small SPM tokenizer on real text and wrap it in an NLLB
    tokenizer + a tiny M2M100 model (NLLB's architecture).

    Using the project's own text means the tokenizer sees Devanagari and Ol
    Chiki, so script handling is genuinely exercised.
    """
    import sentencepiece as spm
    from transformers import M2M100Config, M2M100ForConditionalGeneration, NllbTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    spm_dir = out_dir / "spm"
    spm_dir.mkdir(exist_ok=True)

    corpus_file = spm_dir / "corpus.txt"
    corpus_file.write_text("\n".join(corpus_texts), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(spm_dir / "sp"),
        vocab_size=1000,
        model_type="unigram",
        character_coverage=0.9995,
        bos_id=0, pad_id=1, eos_id=2, unk_id=3,
        minloglevel=2,
    )

    tokenizer = NllbTokenizer(vocab_file=str(spm_dir / "sp.model"))
    tokenizer.save_pretrained(str(out_dir))

    cfg = M2M100Config(
        vocab_size=len(tokenizer),
        d_model=64,
        encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=128, decoder_ffn_dim=128,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
    model = M2M100ForConditionalGeneration(cfg)
    model.save_pretrained(str(out_dir))
    return out_dir


def write_test_config(model_dir: Path, data_path: Path, out: Path) -> Path:
    """A config pointing at the tiny local model, mirroring the real one."""
    import yaml

    base = yaml.safe_load((HERE / "config.yaml").read_text())
    base["data"]["path"] = str(data_path)
    base["output"]["root"] = str(out / "runs")
    base["output"]["tables"] = str(out / "tables")
    base["models"] = {
        "tiny": {"hf_id": str(model_dir), "family": "nllb", "params": "0.1M"}
    }
    base["infer"]["batch_size"] = 8
    base["infer"]["num_beams"] = 1
    base["infer"]["max_source_length"] = 64
    base["infer"]["max_target_length"] = 64
    base["infer"]["checkpoint_every_batches"] = 2
    base["finetune"].update({
        "epochs": 1, "batch_size": 4, "grad_accum": 1, "eval_steps": 10,
        "num_workers": 0, "early_stopping_patience": 1, "lr": 0.001,
        # Force CPU: this is a machinery test, and Apple MPS cannot do
        # scaled_dot_product_attention with dropout, which would fail here for
        # reasons that have nothing to do with the code.
        "use_cpu": True,
    })
    base["metrics"]["bootstrap_rounds"] = 50
    cfg_path = out / "selftest_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
    return cfg_path


def make_mini_corpus(src: Path, dest: Path, rows: int) -> tuple[Path, list[str]]:
    """Slice a few rows per language so the test is fast but real."""
    import pandas as pd

    df = pd.read_csv(src)
    keep = df.groupby("language", group_keys=False).head(rows).reset_index(drop=True)
    keep.to_csv(dest, index=False)
    texts = (keep["tribal"].astype(str).tolist()
             + keep["english"].astype(str).tolist())
    return dest, texts


# ============================================================
# RUN A STAGE
# ============================================================

def run(cmd: list[str], log: Path, timeout: int = 1800) -> tuple[bool, str]:
    """Run a stage, capture everything to a log, return (ok, tail)."""
    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DISABLED"] = "true"
    env["HF_HUB_OFFLINE"] = "1"          # nothing here should hit the network
    env["CUDA_VISIBLE_DEVICES"] = ""     # CPU only: testing plumbing, not speed
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.write_text("TIMEOUT")
        return False, f"timed out after {timeout}s"
    log.write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-12:])
        return False, tail
    return True, ""


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print(" TRIBALSUITE MT — OFFLINE SELF-TEST")
    print("=" * 70)
    print(" Random weights, so translations are gibberish. This tests the")
    print(" machinery, not the model.\n")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    logs = WORK / "logs"
    logs.mkdir()

    sys.path.insert(0, str(HERE))

    # ---- locate data ----
    try:
        import common as C

        cfg = C.load_config()
        C.setup_logging(WORK, "selftest")
        src = C.find_data(cfg, args.data)
        step("find tri-parallel CSV", True, str(src))
    except Exception as e:
        step("find tri-parallel CSV", False, f"{type(e).__name__}: {e}")
        print("\nPass --data /path/to/D1_parallel.csv")
        return 1

    mini, texts = make_mini_corpus(src, WORK / "mini.csv", args.rows)
    step("build mini corpus", True, f"{len(texts) // 2} pairs")

    # ---- tiny model ----
    try:
        model_dir = build_tiny_model(texts, WORK / "tiny_model")
        step("build tiny NLLB-architecture model", True, str(model_dir.name))
    except Exception as e:
        import traceback
        (logs / "build_model.log").write_text(traceback.format_exc())
        step("build tiny NLLB-architecture model", False,
             f"{type(e).__name__}: {e}")
        return summarise()

    cfg_path = write_test_config(model_dir, mini, WORK)
    step("write self-test config", True, cfg_path.name)

    py = sys.executable
    common = ["--config", str(cfg_path), "--data", str(mini)]

    # ---- 1. language-token preflight (dry run) ----
    ok, tail = run([py, "infer.py", "--model", "tiny", *common, "--dry_run"],
                   logs / "dry_run.log")
    if not step("preflight / dry run", ok, tail):
        return summarise()

    # ---- 2. zero-shot inference ----
    ok, tail = run([py, "infer.py", "--model", "tiny", *common,
                    "--limit", "16"], logs / "zeroshot.log")
    if not step("zero-shot inference", ok, tail):
        return summarise()

    pred = WORK / "runs" / "tiny" / "zeroshot" / "predictions.csv"
    ok = pred.exists()
    if ok:
        import pandas as pd
        n = len(pd.read_csv(pred))
        step("predictions written", True, f"{n} rows")
    else:
        step("predictions written", False, f"missing {pred}")
        return summarise()

    # ---- 3. resume is a no-op when already complete ----
    ok, tail = run([py, "infer.py", "--model", "tiny", *common,
                    "--limit", "16"], logs / "resume.log")
    txt = (logs / "resume.log").read_text()
    step("resume detects completed work", ok and "already" in txt.lower(),
         "skipped finished groups" if "already" in txt.lower() else tail)

    # ---- 3b. log archive + index ----
    arch = HERE / "logs"
    archived = sorted(arch.glob("*_infer_tiny_*.log"))
    idx = arch / "index.csv"
    ok_arch = bool(archived) and idx.exists()
    detail = (f"{len(archived)} archived, index.csv present" if ok_arch
              else f"archived={len(archived)} index={idx.exists()}")
    if ok_arch:
        import csv as _csv
        rows = list(_csv.DictReader(open(idx)))
        fin = [r for r in rows if r["event"] == "finished" and r["status"] == "ok"]
        ok_arch = bool(fin)
        detail += f", {len(fin)} finished-ok rows"
    step("logs archived + indexed", ok_arch, detail)

    latest = arch / "latest.log"
    step("logs/latest.log points at newest",
         latest.exists() or (arch / "latest.txt").exists(),
         "symlink" if latest.is_symlink() else "pointer file")

    # ---- 4. source-token sweep ----
    ok, tail = run([py, "infer.py", "--model", "tiny", *common, "--sweep_src",
                    "--split", "dev", "--limit", "8", "--tag", "sweep"],
                   logs / "sweep.log")
    step("source-token sweep (self-healing)", ok, tail)

    # ---- 5. LoRA fine-tuning ----
    ok, tail = run([py, "finetune.py", "--model", "tiny", *common, "--smoke"],
                   logs / "finetune_lora.log", timeout=2400)
    if not step("LoRA fine-tuning", ok, tail):
        return summarise()

    best = WORK / "runs" / "tiny" / "finetune" / "best"
    adapter = (best / "adapter_config.json").exists()
    step("LoRA adapter saved", adapter,
         "adapter_config.json present" if adapter else f"missing in {best}")

    # ---- 6. inference with the fine-tuned checkpoint ----
    ok, tail = run([py, "infer.py", "--model", "tiny", *common,
                    "--ckpt", str(best), "--limit", "16"],
                   logs / "infer_ft.log")
    if not step("inference with LoRA checkpoint (merge)", ok, tail):
        return summarise()

    # ---- 6b. extend epochs WITHOUT restarting ----
    # The important assertion: the second run must report RESUMING at a
    # non-zero step, and must end at more steps than the first run did.
    import json as _json
    m1 = _json.loads((WORK / "runs" / "tiny" / "finetune"
                      / "train_metrics.json").read_text())
    ok, tail = run([py, "finetune.py", "--model", "tiny", *common,
                    "--extend", "--epochs", "3",
                    "--set", "finetune.max_train_samples=50"],
                   logs / "finetune_extend.log", timeout=2400)
    if ok:
        m2 = _json.loads((WORK / "runs" / "tiny" / "finetune"
                          / "train_metrics.json").read_text())
        txt = (logs / "finetune_extend.log").read_text()
        resumed = "RESUMING at step" in txt
        grew = m2.get("steps", 0) > m1.get("steps", 0)
        step("extend epochs resumes (no restart)", resumed and grew,
             f"steps {m1.get('steps')} -> {m2.get('steps')}, "
             f"epochs {m1.get('epochs_completed')} -> "
             f"{m2.get('epochs_completed')}"
             + ("" if resumed else "  [did NOT log RESUMING]")
             + ("" if grew else "  [step count did not increase]"))
    else:
        step("extend epochs resumes (no restart)", False, tail)

    # ---- 6c. training statistics files ----
    tl = WORK / "runs" / "tiny" / "finetune" / "training_log.csv"
    tm = WORK / "runs" / "tiny" / "finetune" / "train_metrics.json"
    if tl.exists() and tm.exists():
        import pandas as pd
        curve = pd.read_csv(tl)
        stats = _json.loads(tm.read_text())
        need = ["initial_loss", "final_loss", "loss_variance", "train_minutes",
                "sec_per_step", "steps", "epochs_completed"]
        missing = [k for k in need if k not in stats]
        step("training statistics captured", not missing,
             f"{len(curve)} curve points, {len(stats)} metrics"
             + (f", MISSING {missing}" if missing else ""))
    else:
        step("training statistics captured", False,
             f"missing {tl.name} or {tm.name}")

    ist = WORK / "runs" / "tiny" / "zeroshot" / "inference_stats.json"
    if ist.exists():
        s = _json.loads(ist.read_text())
        step("inference statistics captured", "per_group" in s,
             f"{len(s.get('per_group', []))} groups timed, "
             f"{s.get('overall_sentences_per_sec')} sent/s")
    else:
        step("inference statistics captured", False, f"missing {ist}")

    # ---- 7. full fine-tune path ----
    ok, tail = run([py, "finetune.py", "--model", "tiny", *common, "--smoke",
                    "--tag", "finetune_full",
                    "--set", "finetune.mode=full", "finetune.lr=3e-5"],
                   logs / "finetune_full.log", timeout=2400)
    step("full fine-tuning path", ok, tail)

    # ---- 8. frozen-layers variant ----
    ok, tail = run([py, "finetune.py", "--model", "tiny", *common, "--smoke",
                    "--tag", "finetune_frozen",
                    "--set", "finetune.mode=full",
                    "finetune.freeze_encoder=true", "finetune.lr=3e-5"],
                   logs / "finetune_frozen.log", timeout=2400)
    step("frozen-layers fine-tuning path", ok, tail)

    # ---- 9. evaluate ----
    ok, tail = run([py, "evaluate.py", "--all", "--config", str(cfg_path),
                    "--name", "selftest"], logs / "evaluate.log")
    if not step("evaluate / metrics", ok, tail):
        return summarise()

    tables = WORK / "tables"
    for ext in ("tsv", "csv", "tex"):
        f = tables / f"selftest.{ext}"
        step(f"table written (.{ext})", f.exists() and f.stat().st_size > 0,
             str(f.name))

    # ---- 10. consolidation ----
    ok, tail = run([py, "consolidate.py", "--config", str(cfg_path),
                    "--out", str(WORK / "consolidated")],
                   logs / "consolidate.log")
    step("consolidate into CSVs", ok, tail)
    cons = WORK / "consolidated"
    for f in ("training_stats.csv", "loss_curves.csv", "inference_stats.csv"):
        step(f"consolidated {f}", (cons / f).exists(), str(f))

    # ---- 11. by_group sweep scoring ----
    ok, tail = run([py, "evaluate.py", "--run",
                    str(WORK / "runs" / "tiny" / "sweep"),
                    "--config", str(cfg_path), "--by_group",
                    "--name", "selftest_sweep", "--no_bootstrap"],
                   logs / "evaluate_sweep.log")
    step("sweep comparison table", ok, tail)

    return summarise(keep=args.keep)


def summarise(keep: bool = True) -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    failed = [(n, d) for n, ok, d in _results if not ok]

    print("\n" + "=" * 70)
    print(f" {passed}/{total} checks passed")
    if failed:
        print("\n FAILURES:")
        for name, detail in failed:
            print(f"   - {name}")
            for ln in str(detail).splitlines()[-6:]:
                print(f"       {ln}")
        print(f"\n Full logs: {WORK / 'logs'}/")
        print("=" * 70)
        return 1

    print("\n Everything works. Safe to run for real:")
    print("     bash run_all.sh")
    print(f"\n (self-test artifacts in {WORK}/ — delete when done)"
          if keep else "")
    print("=" * 70)
    if not keep and WORK.exists():
        shutil.rmtree(WORK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
