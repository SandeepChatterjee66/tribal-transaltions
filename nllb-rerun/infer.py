#!/usr/bin/env python3
"""
infer.py — tribal -> English inference for any registered seq2seq model.

Used for BOTH zero-shot and fine-tuned evaluation, so the two are guaranteed
to run on identical inputs with identical decoding settings. That comparability
is the whole point of the table.

  # zero-shot, all four languages
  python infer.py --model nllb-600m

  # a fine-tuned checkpoint
  python infer.py --model nllb-600m --ckpt runs/nllb-600m/finetune/best

  # which source-language proxy token is fairest? (dev split, small sample)
  python infer.py --model nllb-600m --sweep_src --split dev --limit 200

Resumable: re-running the same command picks up where a crash left off.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import common as C
from common import LOG

_RUN_DIR: Path | None = None
_T_START = time.time()


def parse_args():
    p = argparse.ArgumentParser(
        description="Tribal->English inference (zero-shot or fine-tuned)")
    p.add_argument("--model", required=True,
                   help="model key from config.yaml `models:`")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None, help="path to tri-parallel CSV")
    p.add_argument("--ckpt", default=None,
                   help="fine-tuned checkpoint dir; omit for zero-shot")
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap items per language (smoke tests only)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_beams", type=int, default=None)
    p.add_argument("--tag", default=None,
                   help="output subdir name; default zeroshot/finetuned")
    p.add_argument("--sweep_src", action="store_true",
                   help="try every candidate source token from config sweep list")
    p.add_argument("--overwrite", action="store_true",
                   help="ignore existing predictions and start fresh")
    p.add_argument("--dry_run", action="store_true",
                   help="validate everything, load nothing, translate nothing")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   help="config overrides, e.g. --set infer.num_beams=1")
    return p.parse_args()


def translate(model, tokenizer, forced_bos, texts, batch_size,
              max_src, max_tgt, num_beams, on_progress=None,
              adapter=None, src_lang=None, tgt_lang=None):
    """Batched beam decode with automatic OOM back-off.

    Yields (start_index, hypotheses) per batch so the caller can checkpoint.
    `adapter` is the IndicTrans2 pre/post-processor when that family is in use;
    for every other model it stays None and this is a plain HF decode loop.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    i = 0
    bs = batch_size
    while i < len(texts):
        batch = [str(t) for t in texts[i:i + bs]]
        try:
            model_in = (adapter.preprocess(batch, src_lang, tgt_lang)
                        if adapter else batch)
            enc = tokenizer(model_in, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_src).to(device)
            gen_kwargs = {"max_new_tokens": max_tgt, "num_beams": num_beams}
            # IndicTrans2 encodes the target in the input tags, so forcing a
            # BOS language token would corrupt the output.
            if forced_bos is not None and adapter is None:
                gen_kwargs["forced_bos_token_id"] = forced_bos
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            hyps = tokenizer.batch_decode(out, skip_special_tokens=True)
            if adapter:
                hyps = adapter.postprocess(hyps, tgt_lang)
            hyps = [str(h).strip() for h in hyps]
            yield i, hyps
            i += bs
            if on_progress:
                on_progress(i, len(texts))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise RuntimeError(
                    f"OOM even at batch_size=1 on item {i}. Lower "
                    f"infer.max_source_length or use a smaller model."
                ) from None
            bs = max(1, bs // 2)
            LOG.warning("CUDA OOM — halving batch size to %d and retrying", bs)


def run_one(spec, cfg, df, src_token, out_dir, args, model_cache: dict):
    """Translate one (language, source-token) group, resuming if partial."""
    pred_path = out_dir / "predictions.csv"

    done: set[str] = set()
    rows: list[dict] = []
    if pred_path.exists() and not args.overwrite:
        prev = pd.read_csv(pred_path)
        prev = prev[prev["hyp"].notna() & (prev["hyp"].astype(str) != "")]
        rows = prev.to_dict("records")
        done = set(prev["gid"].astype(str))
        LOG.info("resuming: %d already translated", len(done))

    todo = df[~df["gid"].astype(str).isin(done)].reset_index(drop=True)
    if todo.empty:
        LOG.info("nothing to do — %s already complete", pred_path)
        return pd.DataFrame(rows)

    LOG.info("%d to translate (src_lang=%s)", len(todo), src_token)

    key = (spec.key, args.ckpt or "base")
    if key not in model_cache:
        model, tokenizer, forced_bos = C.load_model_and_tokenizer(
            spec, src_token, for_training=False)
        if args.ckpt:
            model = load_checkpoint(model, args.ckpt)
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")

        adapter = None
        if spec.family == "indictrans2":
            from indictrans2 import IndicTrans2Adapter
            adapter = IndicTrans2Adapter()

        model_cache.clear()          # only ever hold one model in memory
        model_cache[key] = (model, tokenizer, forced_bos, adapter)
    model, tokenizer, forced_bos, adapter = model_cache[key]

    # src_lang can change between groups without reloading the model.
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_token

    bs = args.batch_size or C.deep_get(cfg, "infer.batch_size", 32)
    max_src = C.deep_get(cfg, "infer.max_source_length", 192)
    max_tgt = C.deep_get(cfg, "infer.max_target_length", 192)
    beams = args.num_beams or C.deep_get(cfg, "infer.num_beams", 4)
    ckpt_every = C.deep_get(cfg, "infer.checkpoint_every_batches", 20)

    texts = todo["tribal"].tolist()
    t0 = time.time()
    n_batches = 0
    log_every = max(1, C.deep_get(cfg, "infer.log_every_sentences", 200))
    out_tokens = 0

    def progress(done_n, total_n):
        if done_n % log_every < bs or done_n >= total_n:
            el = time.time() - t0
            rate = done_n / max(el, 1e-6)
            eta = (total_n - done_n) / max(rate, 1e-6)
            mem = ""
            try:
                import torch
                if torch.cuda.is_available():
                    mem = f" | {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            except Exception:
                pass
            LOG.info("  %6d/%-6d (%5.1f%%) | %6.1f sent/s | elapsed %5.1fm | "
                     "ETA %5.1fm%s", done_n, total_n,
                     100 * done_n / max(total_n, 1), rate, el / 60,
                     eta / 60, mem)

    for start, hyps in translate(model, tokenizer, forced_bos, texts, bs,
                                 max_src, max_tgt, beams, progress,
                                 adapter=adapter, src_lang=src_token,
                                 tgt_lang=spec.target_token):
        chunk = todo.iloc[start:start + len(hyps)]
        for (_, r), h in zip(chunk.iterrows(), hyps):
            rows.append({
                "gid": r["gid"], "language": r["language"],
                "tribal": r["tribal"], "english": r["english"],
                "hyp": h, "src_lang": src_token,
            })
            out_tokens += len(str(h).split())
        n_batches += 1
        if n_batches % ckpt_every == 0:
            C.atomic_write(pd.DataFrame(rows), pred_path)
            LOG.info("  ... checkpointed %d rows to disk", len(rows))

    out = pd.DataFrame(rows)
    C.atomic_write(out, pred_path)

    elapsed = time.time() - t0
    n_new = len(todo)
    empty = sum(1 for r in rows if not str(r["hyp"]).strip())
    timing = {
        "language": str(todo["language"].iloc[0]),
        "src_lang": src_token,
        "n_translated_now": n_new,
        "n_total_in_file": len(out),
        "seconds": round(elapsed, 1),
        "minutes": round(elapsed / 60, 2),
        "sentences_per_sec": round(n_new / max(elapsed, 1e-6), 2),
        "sec_per_sentence": round(elapsed / max(n_new, 1), 4),
        "output_tokens": out_tokens,
        "output_tokens_per_sec": round(out_tokens / max(elapsed, 1e-6), 1),
        "batch_size": bs,
        "num_beams": beams,
        "empty_outputs": empty,
        "empty_pct": round(100 * empty / max(len(out), 1), 2),
    }
    try:
        import torch
        if torch.cuda.is_available():
            timing["peak_gpu_mem_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass
    C.save_json(timing, out_dir / "timing.json")

    LOG.info("  done: %d rows in %.1f min | %.1f sent/s | %.0f out-tok/s | "
             "empty %.1f%%", len(out), elapsed / 60,
             timing["sentences_per_sec"], timing["output_tokens_per_sec"],
             timing["empty_pct"])
    if empty:
        LOG.warning("  %d empty output(s) — re-run this command to retry only "
                    "those rows", empty)
    return out


def load_checkpoint(model, ckpt: str):
    """Attach a LoRA adapter or load a fully fine-tuned checkpoint."""
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")

    if (ckpt_path / "adapter_config.json").exists():
        from peft import PeftModel
        LOG.info("loading LoRA adapter from %s", ckpt_path)
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        model = model.merge_and_unload()   # merge for faster inference
        LOG.info("adapter merged into base weights")
        return model

    LOG.info("loading full checkpoint from %s", ckpt_path)
    return C.from_pretrained_seq2seq(str(ckpt_path), model.dtype)


def main():
    global _RUN_DIR
    args = parse_args()
    cfg = C.apply_overrides(C.load_config(args.config), args.overrides)

    setting = args.tag or ("finetuned" if args.ckpt else "zeroshot")
    _RUN_DIR = C.run_dir(cfg, args.model, setting)
    C.setup_logging(_RUN_DIR, "infer", label=f"{args.model}_{setting}")
    C.log_environment(_RUN_DIR)
    C.set_seed(C.deep_get(cfg, "data.seed", 42))

    LOG.info("=" * 62)
    LOG.info(" INFERENCE — %s [%s]", args.model, setting)
    LOG.info("=" * 62)

    spec = C.resolve_model(cfg, args.model)
    languages = args.languages or C.deep_get(cfg, "data.languages")
    if args.limit:
        cfg.setdefault("data", {})["max_test_per_language"] = args.limit

    data_path = C.find_data(cfg, args.data)
    df = C.load_parallel(data_path, languages)
    df = C.make_splits(df, cfg)
    eval_df = C.get_eval_set(df, cfg, args.split)
    C.save_json({"config": {k: v for k, v in cfg.items()
                            if not k.startswith("_")},
                 "args": vars(args)}, _RUN_DIR / "resolved_config.json")

    # Preflight: tokenizer + language codes, before any GPU work. This is the
    # step that catches a wrong language code in seconds instead of an hour in.
    LOG.info("preflight: validating language tokens")
    from transformers import AutoTokenizer
    trust = spec.family == "indictrans2"
    if trust:
        from indictrans2 import check_codes
        check_codes(spec, languages)
    probe = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=trust)

    C.validate_lang_token(probe, spec.target_token, spec)
    resolved_sweeps: dict[str, list[str]] = {}
    for lang in languages:
        if lang not in spec.source_tokens:
            raise ValueError(
                f"No source token configured for {lang!r} under "
                f"lang_tokens.{spec.family}.source in config.yaml")
        # The configured default must be valid — that is not a guess.
        C.validate_lang_token(probe, spec.source_tokens[lang], spec)
        if args.sweep_src:
            cands = spec.sweep_tokens.get(lang) or [spec.source_tokens[lang]]
            resolved_sweeps[lang] = C.filter_known_tokens(
                probe, cands, spec, f"sweep:{lang}")
        else:
            resolved_sweeps[lang] = [spec.source_tokens[lang]]
    LOG.info("preflight OK — target=%s, sources=%s",
             spec.target_token,
             {l: v for l, v in resolved_sweeps.items()})

    if args.dry_run:
        LOG.info("dry run: %d eval items across %s. Exiting before model load.",
                 len(eval_df), eval_df["language"].unique().tolist())
        return

    model_cache: dict = {}
    all_rows = []

    for lang in languages:
        sub = eval_df[eval_df["language"] == lang].reset_index(drop=True)
        if sub.empty:
            LOG.warning("no %s rows in the %s split — skipping", lang, args.split)
            continue

        for src_token in resolved_sweeps[lang]:
            LOG.info("-" * 62)
            LOG.info("%s | src_lang=%s | n=%d", lang, src_token, len(sub))
            sub_dir = (_RUN_DIR / "by_group" / f"{lang}__{src_token}"
                       if args.sweep_src else _RUN_DIR / "by_group" / lang)
            out = run_one(spec, cfg, sub, src_token, sub_dir, args, model_cache)
            all_rows.append(out)

    if all_rows:
        merged = pd.concat(all_rows, ignore_index=True)
        C.atomic_write(merged, _RUN_DIR / "predictions.csv")

        # Roll every per-group timing.json into one summary.
        per_group, total_sec, total_n = [], 0.0, 0
        for tf in sorted((_RUN_DIR / "by_group").glob("*/timing.json")):
            try:
                t = json.loads(tf.read_text())
            except Exception:
                continue
            t["group"] = tf.parent.name
            per_group.append(t)
            total_sec += t.get("seconds", 0)
            total_n += t.get("n_translated_now", 0)

        wall = time.time() - _T_START
        summary = {
            "model": args.model,
            "setting": setting,
            "checkpoint": args.ckpt,
            "split": args.split,
            "n_predictions": len(merged),
            "n_translated_this_invocation": total_n,
            "translate_minutes": round(total_sec / 60, 2),
            "wall_clock_minutes": round(wall / 60, 2),
            "overall_sentences_per_sec": round(total_n / max(total_sec, 1e-6), 2),
            "num_beams": args.num_beams or C.deep_get(cfg, "infer.num_beams", 4),
            "batch_size": args.batch_size or C.deep_get(cfg, "infer.batch_size", 32),
            "per_group": per_group,
        }
        C.save_json(summary, _RUN_DIR / "inference_stats.json")

        LOG.info("=" * 62)
        LOG.info("INFERENCE SUMMARY — %s [%s]", args.model, setting)
        LOG.info("  predictions        : %d", len(merged))
        LOG.info("  translated now     : %d", total_n)
        LOG.info("  translate time     : %.1f min", total_sec / 60)
        LOG.info("  wall clock         : %.1f min (incl. model load)", wall / 60)
        if total_n:
            LOG.info("  throughput         : %.1f sent/s",
                     total_n / max(total_sec, 1e-6))
        for t in per_group:
            LOG.info("    %-22s n=%-6s %6.1f sent/s  %5.1fm  empty %.1f%%",
                     t.get("group", "?"), t.get("n_translated_now", 0),
                     t.get("sentences_per_sec", 0), t.get("minutes", 0),
                     t.get("empty_pct", 0))
        LOG.info("  stats file         : %s", _RUN_DIR / "inference_stats.json")
        LOG.info("  next               : python evaluate.py --run %s", _RUN_DIR)
        LOG.info("=" * 62)


if __name__ == "__main__":
    C.run_guarded(main, lambda: _RUN_DIR)
