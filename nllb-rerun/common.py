#!/usr/bin/env python3
"""
common.py — shared plumbing for the NLLB / seq2seq MT experiments.

Everything that more than one script needs lives here: config loading, data
discovery, the deterministic split, model/tokenizer setup, metrics, logging,
and crash handling. The individual scripts stay short and readable.

Nothing here talks to the network except `load_model_and_tokenizer`, which
pulls from the HuggingFace cache (set HF_HOME to a shared path on the server).
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("tribalmt")

# Canonical column names we normalise everything onto.
REQUIRED_COLS = ["language", "english", "tribal"]


# ============================================================
# CONFIG
# ============================================================

def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml. Falls back to the copy next to this file."""
    import yaml

    cfg_path = Path(path) if path else HERE / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def deep_get(cfg: dict, dotted: str, default=None):
    """cfg lookup by 'a.b.c'. Returns `default` if any level is missing."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `--set a.b.c=value` overrides. Values are parsed as YAML scalars."""
    import yaml

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got: {ov}")
        key, raw = ov.split("=", 1)
        val = yaml.safe_load(raw)
        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
        LOG.info("config override: %s = %r", key, val)
    return cfg


# ============================================================
# LOGGING + CRASH HANDLING
# ============================================================

ARCHIVE_DIR = HERE / "logs"
_ARCHIVE_PATH: Path | None = None
_START_TIME: float | None = None


def setup_logging(run_dir: Path, name: str = "run", label: str | None = None,
                  archive: bool = True) -> Path:
    """Console + two file logs. Returns the per-run log path.

    Three destinations, because they answer different questions:

      1. console — what is happening right now
      2. `<run_dir>/<name>.log` — appended across invocations, so the full
         history of THIS run directory lives in one file
      3. `logs/<timestamp>_<name>_<label>.log` — one file per invocation, never
         overwritten, so an old run's log is still there after you re-run

    `logs/latest.log` is symlinked to the newest archive file, which is what
    `bash watch.sh` tails.
    """
    import time
    from datetime import datetime

    global _ARCHIVE_PATH, _START_TIME
    _START_TIME = time.time()

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"{name}.log"

    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    LOG.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)

    if archive:
        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            _ARCHIVE_PATH = ARCHIVE_DIR / f"{stamp}_{name}{suffix}.log"
            afh = logging.FileHandler(_ARCHIVE_PATH, mode="w", encoding="utf-8")
            afh.setFormatter(fmt)
            LOG.addHandler(afh)

            # `logs/latest.log` -> newest archive, for `tail -f`.
            latest = ARCHIVE_DIR / "latest.log"
            try:
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(_ARCHIVE_PATH.name)
            except OSError:
                # Some shared filesystems disallow symlinks; write a pointer.
                (ARCHIVE_DIR / "latest.txt").write_text(_ARCHIVE_PATH.name)

            _index_append("started", name, label, run_dir)
        except Exception as e:  # never let logging setup kill a run
            print(f"warning: could not set up log archive: {e}", file=sys.stderr)

    LOG.propagate = False
    return log_file


def _index_append(event: str, script: str, label: str | None,
                  run_dir: Path, status: str = "", duration_min: str = "") -> None:
    """Append a row to logs/index.csv.

    Append-only on purpose: it survives a kill -9 mid-write, and `grep` is
    enough to answer "what did I run last Tuesday and did it finish?".
    """
    import csv
    from datetime import datetime

    idx = ARCHIVE_DIR / "index.csv"
    new = not idx.exists()
    try:
        with open(idx, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "event", "script", "label",
                            "duration_min", "status", "log_file", "run_dir",
                            "command"])
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, script,
                label or "", duration_min, status,
                _ARCHIVE_PATH.name if _ARCHIVE_PATH else "",
                str(run_dir), " ".join(sys.argv),
            ])
    except Exception:
        pass


def finish_logging(status: str, run_dir: Path | None = None) -> None:
    """Record how an invocation ended, and where its log went."""
    import time

    dur = ""
    if _START_TIME:
        dur = f"{(time.time() - _START_TIME) / 60:.2f}"
    _index_append("finished", Path(sys.argv[0]).stem, None,
                  run_dir or Path("."), status, dur)
    if _ARCHIVE_PATH:
        LOG.info("log archived: %s", _ARCHIVE_PATH)


def log_environment(run_dir: Path) -> None:
    """Record what we are running on. Cheap, and saves hours when a run
    behaves differently on two nodes."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.environ.get("HF_HOME"),
    }
    for pkg in ("torch", "transformers", "peft", "sacrebleu", "datasets"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = "not installed"

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_mem_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 1
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as e:  # pragma: no cover
        info["torch_probe_error"] = str(e)

    if shutil.which("nvidia-smi"):
        try:
            info["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        except Exception:
            pass

    (run_dir / "environment.json").write_text(json.dumps(info, indent=2))
    LOG.info("environment: %s", json.dumps(
        {k: info[k] for k in ("hostname", "torch", "transformers", "cuda_available")
         if k in info}))


def run_guarded(main_fn, run_dir_getter) -> None:
    """Run `main_fn`, and on any exception dump a full traceback to
    <run_dir>/error.log before exiting non-zero.

    Keeps stack traces out of the scheduler's stderr soup and puts them
    somewhere you can find later.
    """
    try:
        main_fn()
    except KeyboardInterrupt:
        LOG.warning("interrupted by user — partial results are on disk, "
                    "re-run the same command to resume")
        _safe_finish("interrupted", run_dir_getter)
        sys.exit(130)
    except Exception:
        tb = traceback.format_exc()
        LOG.error("FAILED\n%s", tb)
        try:
            rd = run_dir_getter()
            if rd:
                Path(rd).mkdir(parents=True, exist_ok=True)
                (Path(rd) / "error.log").write_text(tb)
                LOG.error("traceback written to %s", Path(rd) / "error.log")
        except Exception:
            pass
        _safe_finish("FAILED", run_dir_getter)
        sys.exit(1)
    else:
        _safe_finish("ok", run_dir_getter)


def _safe_finish(status: str, run_dir_getter) -> None:
    try:
        rd = run_dir_getter()
        finish_logging(status, Path(rd) if rd else None)
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ============================================================
# DATA
# ============================================================

def _anchor(p: str | Path) -> Path:
    """Resolve a possibly-relative path against THIS directory, not the cwd.

    Anchoring to the package rather than the working directory is what lets the
    folder be moved, copied or rsynced to another machine and still run, and it
    means a script behaves the same whether invoked from here or from elsewhere.
    """
    p = Path(p).expanduser()
    return p if p.is_absolute() else (HERE / p)


def raw_data_dir(cfg: dict) -> Path:
    """Where the raw corpus CSVs are expected. Fixed by config, not the cwd."""
    return _anchor(deep_get(cfg, "data.raw_dir", "raw_data"))


def find_data(cfg: dict, explicit: str | None = None) -> Path:
    """Locate the tri-parallel CSV that prepare_data.py generates.

    Order: --data flag, config `data.path`, $TRIBAL_DATA, then each
    `data.search_hints` entry. Relative paths anchor to this directory first and
    the cwd second, so the normal case needs no configuration at all.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if deep_get(cfg, "data.path"):
        candidates.append(_anchor(deep_get(cfg, "data.path")))
    if os.environ.get("TRIBAL_DATA"):
        candidates.append(Path(os.environ["TRIBAL_DATA"]).expanduser())
    for hint in deep_get(cfg, "data.search_hints", []) or []:
        candidates.append(_anchor(hint))
        candidates.append(Path.cwd() / hint)

    for c in candidates:
        if c.is_file():
            LOG.info("data: %s", c.resolve())
            return c.resolve()

    # Not found. The most likely reason is simply that prepare_data.py has not
    # been run yet, so say that first rather than listing override flags.
    raw = raw_data_dir(cfg)
    langs = deep_get(cfg, "data.languages", []) or []
    present = sorted(p.name for p in raw.glob("*.csv")) if raw.is_dir() else []

    if present:
        hint = (f"Raw data IS present in {raw} ({', '.join(present)}).\n"
                f"You just need to generate the held-out file:\n\n"
                f"    python prepare_data.py\n")
    else:
        hint = (f"No raw data found either. Put one CSV per language in:\n\n"
                f"    {raw}/\n"
                + "".join(f"      {l}.csv   columns: English,Hindi,{l}\n"
                          for l in langs)
                + "\nthen run:\n\n    python prepare_data.py\n")

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"\nNo D1_parallel.csv found.\n\n{hint}\n"
        f"Paths checked:\n  {tried}\n\n"
        f"To use a file kept somewhere else instead, either pass "
        f"--data /path/to/D1_parallel.csv, set $TRIBAL_DATA, or set data.path "
        f"in config.yaml."
    )


def load_parallel(path: Path, languages: list[str]) -> pd.DataFrame:
    """Load and normalise the parallel corpus.

    Accepts either the processed long format (language, english, hindi, tribal)
    or a per-language wide file (English, Hindi, <LangName>). Column matching
    is case-insensitive so server-side copies with different casing still work.
    """
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}

    # Wide format: English,Hindi,Bhili  ->  reshape to long.
    if "language" not in lower:
        lang_cols = [l for l in languages if l.lower() in lower]
        if len(lang_cols) == 1:
            lang = lang_cols[0]
            df = df.rename(columns={
                lower["english"]: "english",
                lower[lang.lower()]: "tribal",
            })
            if "hindi" in lower:
                df = df.rename(columns={lower["hindi"]: "hindi"})
            df["language"] = lang
            LOG.info("detected wide-format file for %s", lang)
        else:
            raise ValueError(
                f"{path} has no 'language' column and no single recognised "
                f"language column. Columns present: {list(df.columns)}"
            )
    else:
        ren = {}
        for want in ("language", "english", "tribal", "hindi"):
            if want in lower:
                ren[lower[want]] = want
        df = df.rename(columns=ren)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=["english", "tribal"])
    df["english"] = df["english"].astype(str).str.strip()
    df["tribal"] = df["tribal"].astype(str).str.strip()
    df = df[(df["english"] != "") & (df["tribal"] != "")]
    df = df[df["language"].isin(languages)].reset_index(drop=True)
    if len(df) < before:
        LOG.info("dropped %d rows (empty/NaN/other language)", before - len(df))

    if df.empty:
        raise ValueError(
            f"No usable rows left after filtering to {languages}. "
            f"Languages in file: {sorted(pd.read_csv(path).get('language', pd.Series()).unique())}"
        )

    # Stable global id so predictions can be joined back and resumed safely.
    df["gid"] = (
        df["language"].astype(str) + "_" +
        df.groupby("language").cumcount().astype(str)
    )

    LOG.info("loaded %d pairs: %s", len(df),
             df["language"].value_counts().to_dict())
    return df


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Deterministic per-language 70/10/20 split.

    Per-language (not global) so every language is represented in train, dev
    and test in proportion. Seeded, so the same rows land in the same split on
    every machine and every re-run — this is what makes zero-shot and
    fine-tuned numbers comparable to each other.
    """
    frac_train = deep_get(cfg, "data.split.train", 0.70)
    frac_dev = deep_get(cfg, "data.split.dev", 0.10)
    seed = deep_get(cfg, "data.seed", 42)

    parts = []
    for lang, g in df.groupby("language", sort=True):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        n_tr = int(round(n * frac_train))
        n_dv = int(round(n * frac_dev))
        split = np.array(["test"] * n, dtype=object)
        split[:n_tr] = "train"
        split[n_tr:n_tr + n_dv] = "dev"
        g["split"] = split
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    LOG.info("split sizes:\n%s",
             out.groupby(["language", "split"]).size().unstack(fill_value=0))
    return out


def get_eval_set(df: pd.DataFrame, cfg: dict, split: str = "test") -> pd.DataFrame:
    """The evaluation slice, with the optional per-language cap applied
    deterministically (head of the already-shuffled split)."""
    sub = df[df["split"] == split].copy()
    cap = deep_get(cfg, "data.max_test_per_language")
    if cap:
        sub = sub.groupby("language", group_keys=False).head(int(cap))
        LOG.warning("max_test_per_language=%s — SMOKE RUN, do not report these "
                    "numbers as final", cap)
    return sub.reset_index(drop=True)


# ============================================================
# MODELS
# ============================================================

@dataclass
class ModelSpec:
    key: str
    hf_id: str
    family: str
    params: str
    target_token: str
    source_tokens: dict[str, str]
    sweep_tokens: dict[str, list[str]]


def resolve_model(cfg: dict, key: str) -> ModelSpec:
    models = deep_get(cfg, "models", {}) or {}
    if key not in models:
        raise ValueError(
            f"Unknown model '{key}'. Available: {sorted(models)}\n"
            f"Add new ones under `models:` in {cfg.get('_config_path')}"
        )
    m = models[key]
    fam = m["family"]
    toks = deep_get(cfg, f"lang_tokens.{fam}")
    if not toks:
        raise ValueError(f"No lang_tokens entry for family '{fam}' in config")
    return ModelSpec(
        key=key,
        hf_id=m["hf_id"],
        family=fam,
        params=m.get("params", "?"),
        target_token=toks["target"],
        source_tokens=toks["source"],
        sweep_tokens=toks.get("sweep", {}),
    )


def pick_dtype():
    """bf16 where supported (Ampere+), else fp16 on CUDA, else fp32."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32, "fp32 (cpu)"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def from_pretrained_seq2seq(hf_id: str, dtype, trust_remote_code: bool = False):
    """Load a seq2seq model, tolerating the `torch_dtype` -> `dtype` rename.

    transformers <4.56 wants `torch_dtype`; newer versions warn on it and
    prefer `dtype`. Try the new name, fall back to the old one.
    """
    from transformers import AutoModelForSeq2SeqLM

    try:
        return AutoModelForSeq2SeqLM.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code, dtype=dtype)
    except TypeError:
        LOG.info("transformers does not accept dtype= — using torch_dtype=")
        return AutoModelForSeq2SeqLM.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code, torch_dtype=dtype)


def load_model_and_tokenizer(spec: ModelSpec, src_lang: str,
                             for_training: bool = False):
    """Load tokenizer + seq2seq model with the source language token set.

    NLLB/mBART/M2M100 all need the source language fixed on the *tokenizer*
    and the target language forced at generation time, so this returns both
    plus the resolved forced-BOS id.
    """
    import torch
    from transformers import AutoTokenizer

    trust = spec.family == "indictrans2"
    LOG.info("loading %s (%s, %s params) src_lang=%s",
             spec.key, spec.hf_id, spec.params, src_lang)

    tok_kwargs: dict[str, Any] = {"trust_remote_code": trust}
    if spec.family in ("nllb", "mbart50", "m2m100"):
        tok_kwargs["src_lang"] = src_lang
        tok_kwargs["tgt_lang"] = spec.target_token

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **tok_kwargs)
    validate_lang_token(tokenizer, src_lang, spec)
    validate_lang_token(tokenizer, spec.target_token, spec)

    dtype, dtype_name = pick_dtype()
    # Train in fp32/bf16 master weights; fp16 weights + fp16 grads diverge.
    load_dtype = torch.float32 if (for_training and dtype == torch.float16) else dtype

    model = from_pretrained_seq2seq(spec.hf_id, load_dtype, trust)
    forced_bos = resolve_lang_token_id(tokenizer, spec.target_token)

    # Set the target-language BOS on generation_config ONLY. transformers 5
    # raises if you mutate generation fields on model.config ("This strategy to
    # control generation is not supported anymore"), and generate() is also
    # given forced_bos_token_id explicitly, so this is just a sensible default.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_bos_token_id = forced_bos

    LOG.info("loaded in %s | forced_bos(%s)=%s | %.0fM params",
             dtype_name, spec.target_token, forced_bos,
             sum(p.numel() for p in model.parameters()) / 1e6)
    return model, tokenizer, forced_bos


def resolve_lang_token_id(tokenizer, code: str) -> int:
    """Get the id for a language code across transformers versions.

    `lang_code_to_id` was removed in newer releases; convert_tokens_to_ids is
    the stable path. Raises with the available codes if the token is unknown.
    """
    tid = tokenizer.convert_tokens_to_ids(code)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is not None and tid != unk and tid >= 0:
        return tid

    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(mapping, dict) and code in mapping:
        return mapping[code]

    raise ValueError(
        f"Language token {code!r} is not in this tokenizer's vocabulary.\n"
        f"Available language codes: {available_lang_tokens(tokenizer)[:80]} ..."
    )


def available_lang_tokens(tokenizer) -> list[str]:
    """Best-effort list of language codes a tokenizer knows."""
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(mapping, dict) and mapping:
        return sorted(mapping)
    extra = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    return sorted(t for t in extra if "_" in t or len(t) <= 7)


def validate_lang_token(tokenizer, code: str, spec: ModelSpec) -> None:
    """Fail at startup, not 40 minutes into a run, if a code is wrong."""
    try:
        resolve_lang_token_id(tokenizer, code)
    except ValueError as e:
        raise ValueError(
            f"[{spec.key}] bad language token {code!r}.\n{e}\n"
            f"Fix it under lang_tokens.{spec.family} in config.yaml."
        ) from None


def filter_known_tokens(tokenizer, codes: list[str], spec: ModelSpec,
                        label: str) -> list[str]:
    """Keep only the language codes this tokenizer actually knows.

    Used for sweep candidate lists: a hopeful guess in config.yaml should
    degrade to a warning, not kill a queued job. Raises only if nothing at
    all survives, since then there is no experiment to run.
    """
    good, bad = [], []
    for c in codes:
        try:
            resolve_lang_token_id(tokenizer, c)
            good.append(c)
        except ValueError:
            bad.append(c)
    if bad:
        LOG.warning("[%s/%s] dropping unknown language token(s) %s — "
                    "not in this tokenizer's vocabulary", spec.key, label, bad)
    if not good:
        raise ValueError(
            f"[{spec.key}] none of the candidate tokens {codes} exist for "
            f"{label}. Valid codes include: "
            f"{available_lang_tokens(tokenizer)[:60]}\n"
            f"Fix lang_tokens.{spec.family} in config.yaml."
        )
    return good


# ============================================================
# TRANSFORMERS VERSION TOLERANCE
# ============================================================

def available_reporters() -> list[str]:
    """Which Trainer loggers we can actually use.

    The Trainer raises outright if you ask for tensorboard and it is not
    installed. Losing a queued training job to a *logging* dependency is
    exactly the kind of avoidable failure worth guarding against, so probe
    first and fall back to no reporting.
    """
    import importlib.util

    for mod, name in (("tensorboard", "tensorboard"),
                      ("tensorboardX", "tensorboard")):
        if importlib.util.find_spec(mod) is not None:
            return [name]
    LOG.warning("tensorboard not installed — training will run but without "
                "loss curves. Install with: pip install tensorboard")
    return []


def build_training_args(cls, desired: dict, n_train: int = 0,
                        warmup_ratio: float | None = None):
    """Construct TrainingArguments across transformers 4.x and 5.x.

    The API keeps moving: `evaluation_strategy` became `eval_strategy` in
    4.41, and 5.x dropped `group_by_length` and `warmup_ratio` outright. Rather
    than pin a version, we filter the kwargs against the real signature and
    translate what we can, logging every substitution so the run log records
    exactly which settings took effect.
    """
    import inspect

    sig = set(inspect.signature(cls.__init__).parameters)
    kwargs = dict(desired)

    # eval_strategy (>=4.41) vs evaluation_strategy (<4.41)
    if "eval_strategy" in kwargs and "eval_strategy" not in sig:
        if "evaluation_strategy" in sig:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
            LOG.info("compat: eval_strategy -> evaluation_strategy")
        else:
            kwargs.pop("eval_strategy")

    # warmup_ratio removed in 5.x -> convert to absolute warmup_steps
    if warmup_ratio:
        if "warmup_ratio" in sig:
            kwargs["warmup_ratio"] = warmup_ratio
        elif "warmup_steps" in sig and n_train:
            bs = kwargs.get("per_device_train_batch_size", 8)
            ga = kwargs.get("gradient_accumulation_steps", 1)
            epochs = kwargs.get("num_train_epochs", 1)
            total = max(1, int(n_train / max(bs * ga, 1) * epochs))
            kwargs["warmup_steps"] = max(1, int(total * warmup_ratio))
            LOG.info("compat: warmup_ratio=%.3f -> warmup_steps=%d (of ~%d)",
                     warmup_ratio, kwargs["warmup_steps"], total)

    # use_cpu was called no_cuda before 4.34
    if "use_cpu" in kwargs and "use_cpu" not in sig:
        if "no_cuda" in sig:
            kwargs["no_cuda"] = kwargs.pop("use_cpu")
            LOG.info("compat: use_cpu -> no_cuda")
        else:
            kwargs.pop("use_cpu")

    # group_by_length removed in 5.x -> sortish_sampler has the same intent
    if kwargs.get("group_by_length") and "group_by_length" not in sig:
        kwargs.pop("group_by_length", None)
        if "sortish_sampler" in sig:
            kwargs["sortish_sampler"] = True
            LOG.info("compat: group_by_length -> sortish_sampler "
                     "(length bucketing, same purpose)")
        else:
            LOG.warning("compat: no length-bucketing option in this "
                        "transformers version — training will be somewhat "
                        "slower due to padding waste")

    dropped = [k for k in kwargs if k not in sig]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        LOG.warning("compat: this transformers version ignores %s", dropped)

    return cls(**kwargs)


# ============================================================
# METRICS
# ============================================================

_SPBLEU_TOKENIZER: str | None = None


def _spbleu_tokenizer() -> str:
    """Prefer the FLORES-200 SPM tokenizer for spBLEU; degrade loudly."""
    global _SPBLEU_TOKENIZER
    if _SPBLEU_TOKENIZER:
        return _SPBLEU_TOKENIZER
    import sacrebleu

    errors = []
    for name in ("flores200", "flores101"):
        try:
            sacrebleu.corpus_bleu(["a"], [["a"]], tokenize=name)
            _SPBLEU_TOKENIZER = name
            LOG.info("spBLEU tokenizer: %s", name)
            return name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__} {str(e).strip()[:120]}")
    LOG.warning(
        "FLORES tokenizer unavailable — falling back to 13a. spBLEU will NOT "
        "be comparable to the paper's numbers.\n"
        "  Almost always this is a missing sentencepiece:  "
        "pip install sentencepiece\n"
        "  (NLLB's tokenizer needs it too, so install it regardless.)\n"
        "  probe errors: %s", " | ".join(errors))
    _SPBLEU_TOKENIZER = "13a"
    return "13a"


FAILURE_PLACEHOLDERS = {"ERROR", "NAN", "NONE", "NULL", "<PAD>"}


def as_text(values, treat_placeholders_as_empty: bool = False) -> list[str]:
    """Coerce any pandas/py sequence to a list of plain `str`.

    Needed because pandas >= 3 keeps missing values missing through
    `.astype(str)` instead of rendering them as the literal "nan", so a CSV
    round-trip of an empty cell hands you a float back. Metrics libraries then
    crash deep inside a loop, which is a miserable way to lose a long run.

    With `treat_placeholders_as_empty`, API-failure markers like "ERROR"
    collapse to "" so they score zero and stay countable instead of being
    silently dropped.
    """
    out: list[str] = []
    for x in list(values):
        if x is None:
            s = ""
        elif isinstance(x, float) and x != x:      # NaN
            s = ""
        else:
            s = str(x).strip()
        if treat_placeholders_as_empty and s.upper() in FAILURE_PLACEHOLDERS:
            s = ""
        out.append(s)
    return out


def compute_metrics(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """chrF++ (primary), spBLEU, BLEU, ROUGE-L — same definitions as the paper."""
    import sacrebleu
    from rouge_score import rouge_scorer

    hyps, refs = as_text(hyps), as_text(refs)
    if not hyps:
        return {"n": 0}

    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    spbleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=_spbleu_tokenizer()).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = 100.0 * sum(
        rs.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)
    ) / len(hyps)

    return {
        "n": len(hyps),
        "chrf++": round(chrf, 2),
        "spBLEU": round(spbleu, 2),
        "BLEU": round(bleu, 2),
        "ROUGE-L": round(rouge_l, 2),
    }


def bootstrap_ci(hyps: list[str], refs: list[str], rounds: int = 1000,
                 seed: int = 42, metric: str = "chrf++") -> dict[str, float]:
    """Percentile bootstrap CI on a corpus-level metric.

    Reviewers asked for variance; this is cheaper than multi-seed reruns and is
    the same family of test the paper already uses for classification.

    chrF++ and BLEU are both computed from additive per-segment sufficient
    statistics, so we extract those once and resample the statistics rather
    than re-tokenising the corpus 1000 times. On the full test split that is
    the difference between ~75 minutes and a few seconds.
    """
    hyps, refs = as_text(hyps), as_text(refs)
    if rounds <= 0 or len(hyps) < 2:
        return {}

    from sacrebleu.metrics import BLEU, CHRF

    m = (CHRF(word_order=2) if metric == "chrf++"
         else BLEU(tokenize=_spbleu_tokenizer()))

    rng = np.random.default_rng(seed)
    n = len(hyps)

    try:
        stats = m._extract_corpus_statistics(hyps, [refs])
        scores = np.empty(rounds)
        for k in range(rounds):
            idx = rng.integers(0, n, n)
            scores[k] = m._aggregate_and_compute([stats[i] for i in idx]).score
    except (AttributeError, TypeError) as e:
        # sacrebleu changed its internals — fall back to the slow but always
        # correct path, and say so rather than silently taking 75 minutes.
        LOG.warning("fast bootstrap unavailable (%s) — using the slow path; "
                    "reduce metrics.bootstrap_rounds if this drags", e)
        scores = np.empty(rounds)
        for k in range(rounds):
            idx = rng.integers(0, n, n)
            scores[k] = m.corpus_score([hyps[i] for i in idx],
                                       [[refs[i] for i in idx]]).score

    return {
        f"{metric}_mean": round(float(scores.mean()), 2),
        f"{metric}_ci_lo": round(float(np.percentile(scores, 2.5)), 2),
        f"{metric}_ci_hi": round(float(np.percentile(scores, 97.5)), 2),
    }


# ============================================================
# IO
# ============================================================

def atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file + rename so a crash mid-write cannot corrupt
    an existing results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def runs_root(cfg: dict) -> Path:
    """Anchored, so runs land next to the code however you invoke it."""
    return _anchor(deep_get(cfg, "output.root", "runs"))


def tables_dir(cfg: dict) -> Path:
    return _anchor(deep_get(cfg, "output.tables", "tables"))


def run_dir(cfg: dict, model_key: str, setting: str) -> Path:
    d = runs_root(cfg) / model_key / setting
    d.mkdir(parents=True, exist_ok=True)
    return d
