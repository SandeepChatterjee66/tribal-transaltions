#!/usr/bin/env python3
"""
finetune.py — fine-tune a seq2seq MT model on TribalCorp D1 (tribal -> English).

Trains on the SAME D1 train split the encoder experiments use, and never
touches the D1 test split, so the resulting numbers sit in the same table as
TribalBERT without an asterisk.

  # default: LoRA, all 4 languages in one model (fast)
  python finetune.py --model nllb-600m

  # one model per language (matches the per-language encoder protocol)
  python finetune.py --model nllb-600m --per_language

  # full fine-tune (slower, needs more memory)
  python finetune.py --model nllb-600m --set finetune.mode=full finetune.lr=3e-5

  # 5-minute sanity check before committing the GPU
  python finetune.py --model nllb-600m --smoke

Resumable: re-run the same command and it continues from the last checkpoint.
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


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune seq2seq MT on TribalCorp")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--per_language", action="store_true",
                   help="train one model per language instead of one pooled model")
    p.add_argument("--tag", default=None, help="output subdir name")
    p.add_argument("--epochs", type=int, default=None,
                   help="override finetune.epochs (use with --extend to train "
                        "longer from an existing checkpoint)")
    p.add_argument("--extend", action="store_true",
                   help="continue an already-finished run to a higher --epochs "
                        "instead of skipping it. Picks up from the last "
                        "checkpoint; does NOT restart from epoch 1.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (50 train / 13 dev per language, 1 epoch) to "
                        "prove the pipeline works before spending real GPU time")
    p.add_argument("--verbose", action="store_true",
                   help="log every step instead of every 25")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    return p.parse_args()


class Seq2SeqCollator:
    """Pads a batch and always supplies `decoder_input_ids`.

    Two things conspire to make this necessary rather than optional:

      1. `label_smoothing_factor` makes the HF Trainer pop `labels` out of the
         batch and compute the loss itself, so the model never gets the chance
         to derive decoder inputs from labels.
      2. transformers 5 removed `prepare_decoder_input_ids_from_labels` from
         M2M100/NLLB, so `DataCollatorForSeq2Seq(model=...)` silently stops
         producing `decoder_input_ids` too.

    Together those leave the decoder with neither input ids nor embeddings and
    the model raises a confusing "cannot specify both" error. Shifting the
    labels here fixes it in every version and keeps label smoothing available.
    """

    def __init__(self, tokenizer, model, label_pad_token_id: int = -100):
        from transformers import DataCollatorForSeq2Seq

        self.inner = DataCollatorForSeq2Seq(
            tokenizer, label_pad_token_id=label_pad_token_id, padding=True)
        self.label_pad_token_id = label_pad_token_id

        cfg = getattr(getattr(model, "base_model", model), "config", None) \
            or model.config
        self.pad_token_id = cfg.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = (
            getattr(cfg, "decoder_start_token_id", None)
            or getattr(cfg, "eos_token_id", None)
            or tokenizer.eos_token_id
        )
        if self.pad_token_id is None or self.decoder_start_token_id is None:
            raise ValueError(
                "Cannot build decoder inputs: the model config has no "
                f"pad_token_id ({self.pad_token_id}) or "
                f"decoder_start_token_id ({self.decoder_start_token_id})."
            )
        LOG.info("collator: pad=%s decoder_start=%s",
                 self.pad_token_id, self.decoder_start_token_id)

    def __call__(self, features):
        import torch

        batch = self.inner(features)
        if "decoder_input_ids" in batch or "labels" not in batch:
            return batch

        labels = batch["labels"]
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.decoder_start_token_id
        # -100 is a loss-ignore marker, never a real token id.
        shifted.masked_fill_(shifted == self.label_pad_token_id,
                             self.pad_token_id)
        batch["decoder_input_ids"] = shifted
        return batch


def build_dataset(df, tokenizer, spec, max_src, max_tgt, desc, seed=42):
    """Tokenise into a HuggingFace Dataset, one language group at a time.

    Grouping by language is what lets each example carry its own source
    language token (NLLB puts that token in the input sequence, so it has to
    be set on the tokenizer before encoding).

    The `length` column feeds group_by_length, which buckets similar-length
    inputs together and cuts padding waste substantially — the single cheapest
    speedup available here.
    """
    from datasets import Dataset, concatenate_datasets

    parts = []
    for lang, g in df.groupby("language", sort=True):
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = spec.source_tokens[lang]
            tokenizer.tgt_lang = spec.target_token

        ds = Dataset.from_pandas(
            g[["tribal", "english"]].reset_index(drop=True),
            preserve_index=False)

        def tok_fn(batch):
            enc = tokenizer(
                [str(t) for t in batch["tribal"]],
                text_target=[str(e) for e in batch["english"]],
                truncation=True,
                max_length=max_src,
            )
            enc["labels"] = [lab[:max_tgt] for lab in enc["labels"]]
            enc["length"] = [len(x) for x in enc["input_ids"]]
            return enc

        ds = ds.map(tok_fn, batched=True, batch_size=512,
                    remove_columns=["tribal", "english"],
                    desc=f"tokenise {desc}:{lang}")
        parts.append(ds)
        LOG.info("  %s: %d examples, mean src len %.1f tokens",
                 lang, len(ds), sum(ds["length"]) / max(len(ds), 1))

    out = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    # Shuffle so language groups are interleaved rather than seen in blocks.
    return out.shuffle(seed=seed)


def attach_lora(model, cfg):
    from peft import LoraConfig, TaskType, get_peft_model

    targets = C.deep_get(cfg, "finetune.lora.target_modules",
                         ["q_proj", "k_proj", "v_proj", "out_proj"])
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    usable = [t for t in targets if t in present]
    if not usable:
        raise ValueError(
            f"None of the configured LoRA target_modules {targets} exist in "
            f"this model. Some candidates present: "
            f"{sorted(m for m in present if len(m) <= 12)[:30]}\n"
            f"Fix finetune.lora.target_modules in config.yaml."
        )
    if set(usable) != set(targets):
        LOG.warning("LoRA targets present: %s (requested %s)", usable, targets)

    lcfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=C.deep_get(cfg, "finetune.lora.r", 16),
        lora_alpha=C.deep_get(cfg, "finetune.lora.alpha", 32),
        lora_dropout=C.deep_get(cfg, "finetune.lora.dropout", 0.05),
        target_modules=usable,
    )
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("LoRA attached: %.2fM trainable / %.0fM total (%.2f%%)",
             trainable / 1e6, total / 1e6, 100 * trainable / total)
    return model


def freeze_encoder(model):
    """Freeze the whole encoder, training only the decoder (+ cross-attention).

    The third fine-tuning variant you wanted to compare. The reasoning: the
    encoder has to represent an unseen tribal language, the decoder only has to
    produce English it already knows well. Freezing the encoder tests whether
    the gains come from adapting the *reader* or the *writer* — and it trains
    in roughly half the time.
    """
    frozen = 0
    base = getattr(model, "base_model", model)
    enc = getattr(base, "encoder", None) or getattr(
        getattr(base, "model", base), "encoder", None)
    if enc is None:
        LOG.warning("could not locate an encoder on this model — "
                    "freeze_encoder had no effect")
        return model
    for p in enc.parameters():
        p.requires_grad = False
        frozen += p.numel()
    LOG.info("froze encoder: %.1fM parameters", frozen / 1e6)
    return model


def freeze_embeddings(model):
    """NLLB's embedding matrix is ~256k x d — by far the largest parameter
    block. Freezing it is the single biggest speed/memory win for full FT and
    costs almost nothing in quality when the target language (English) is
    already well covered."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ("shared.weight", "embed_tokens",
                                   "embed_positions")):
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        LOG.info("froze %.1fM embedding parameters", frozen / 1e6)
    return model


def make_stats_callback(out_dir: Path, total_examples: int, tokens_per_example: float):
    """A TrainerCallback that makes the run transparent while it happens.

    Emits, on every logging step: step, epoch, loss, learning rate, grad norm,
    samples/s, tokens/s, GPU memory, elapsed, and **ETA**. Appends every row to
    `training_log.csv` as it goes, so you can tail or plot the loss curve mid-run
    rather than waiting for the end.
    """
    import time

    import torch
    from transformers import TrainerCallback

    log_csv = out_dir / "training_log.csv"

    class StatsCallback(TrainerCallback):
        def on_train_begin(self, targs, state, control, **kw):
            self.t0 = time.time()
            self.rows = []
            self.start_step = state.global_step
            if state.global_step:
                LOG.info("RESUMING at step %d of %d (%.1f%% already done) — "
                         "not restarting from scratch",
                         state.global_step, state.max_steps,
                         100 * state.global_step / max(state.max_steps, 1))
            LOG.info("total optimisation steps planned: %d over %.1f epochs",
                     state.max_steps, targs.num_train_epochs)
            if not log_csv.exists():
                log_csv.write_text(
                    "step,epoch,loss,eval_loss,lr,grad_norm,samples_per_s,"
                    "tokens_per_s,gpu_mem_gb,elapsed_min,eta_min\n")

        def on_log(self, targs, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" not in logs and "eval_loss" not in logs:
                return
            now = time.time()
            elapsed = now - self.t0
            steps_done = max(state.global_step - self.start_step, 1)
            sec_per_step = elapsed / steps_done
            remaining = max(state.max_steps - state.global_step, 0)
            eta = remaining * sec_per_step

            eff_batch = (targs.per_device_train_batch_size
                         * targs.gradient_accumulation_steps
                         * max(targs.world_size, 1))
            samples_s = eff_batch / sec_per_step if sec_per_step else 0.0
            tokens_s = samples_s * tokens_per_example

            mem = 0.0
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9

            row = {
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0), 3),
                "loss": logs.get("loss", ""),
                "eval_loss": logs.get("eval_loss", ""),
                "lr": logs.get("learning_rate", ""),
                "grad_norm": logs.get("grad_norm", ""),
                "samples_per_s": round(samples_s, 2),
                "tokens_per_s": round(tokens_s, 1),
                "gpu_mem_gb": round(mem, 2),
                "elapsed_min": round(elapsed / 60, 2),
                "eta_min": round(eta / 60, 1),
            }
            self.rows.append(row)
            with open(log_csv, "a") as f:
                f.write(",".join(str(row[k]) for k in [
                    "step", "epoch", "loss", "eval_loss", "lr", "grad_norm",
                    "samples_per_s", "tokens_per_s", "gpu_mem_gb",
                    "elapsed_min", "eta_min"]) + "\n")

            if "eval_loss" in logs:
                LOG.info("  [eval] step %d/%d  epoch %.2f  eval_loss=%.4f  "
                         "elapsed %.1fm  ETA %.1fm",
                         state.global_step, state.max_steps, row["epoch"],
                         logs["eval_loss"], elapsed / 60, eta / 60)
            else:
                LOG.info("  step %d/%d (%.0f%%)  epoch %.2f  loss=%.4f  "
                         "lr=%.2e  %.1f samp/s  %.0f tok/s  mem %.1fGB  "
                         "elapsed %.1fm  ETA %.1fm",
                         state.global_step, state.max_steps,
                         100 * state.global_step / max(state.max_steps, 1),
                         row["epoch"], logs.get("loss", float("nan")),
                         float(logs.get("learning_rate") or 0),
                         samples_s, tokens_s, mem, elapsed / 60, eta / 60)

        def on_epoch_end(self, targs, state, control, **kw):
            LOG.info("--- finished epoch %.2f / %.1f  (step %d/%d) ---",
                     float(state.epoch or 0), targs.num_train_epochs,
                     state.global_step, state.max_steps)

    return StatsCallback()


def loss_curve_stats(log_csv: Path) -> dict:
    """Summarise the loss curve: convergence, stability, and where it plateaued.

    Deliberately mirrors the encoder tables already in the paper (training
    convergence, loss variance, computational efficiency) so the MT rows can be
    reported the same way.
    """
    if not log_csv.exists():
        return {}
    try:
        df = pd.read_csv(log_csv)
    except Exception:
        return {}
    train = df[pd.to_numeric(df["loss"], errors="coerce").notna()].copy()
    if train.empty:
        return {}
    train["loss"] = pd.to_numeric(train["loss"])

    first, last = float(train["loss"].iloc[0]), float(train["loss"].iloc[-1])
    drop = first - last
    # Step at which 90% of the total loss reduction had been achieved.
    step_90 = None
    if drop > 0:
        target = first - 0.9 * drop
        hit = train[train["loss"] <= target]
        if not hit.empty:
            step_90 = int(hit["step"].iloc[0])

    out = {
        "initial_loss": round(first, 4),
        "final_loss": round(last, 4),
        "min_loss": round(float(train["loss"].min()), 4),
        "loss_reduction": round(drop, 4),
        "loss_variance": round(float(train["loss"].var()), 6),
        "loss_std": round(float(train["loss"].std()), 4),
        "step_to_90pct_reduction": step_90,
        "n_logged_points": int(len(train)),
        "mean_samples_per_s": round(
            float(pd.to_numeric(train["samples_per_s"], errors="coerce").mean()), 2),
        "mean_tokens_per_s": round(
            float(pd.to_numeric(train["tokens_per_s"], errors="coerce").mean()), 1),
        "peak_gpu_mem_gb": round(
            float(pd.to_numeric(train["gpu_mem_gb"], errors="coerce").max()), 2),
    }
    ev = df[pd.to_numeric(df["eval_loss"], errors="coerce").notna()].copy()
    if not ev.empty:
        ev["eval_loss"] = pd.to_numeric(ev["eval_loss"])
        out["best_eval_loss"] = round(float(ev["eval_loss"].min()), 4)
        out["best_eval_step"] = int(ev.loc[ev["eval_loss"].idxmin(), "step"])
        out["final_eval_loss"] = round(float(ev["eval_loss"].iloc[-1]), 4)
        out["n_evals"] = int(len(ev))
    return out


def train_one(spec, cfg, train_df, dev_df, out_dir, args):
    import torch
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    done_marker = out_dir / "DONE"
    if done_marker.exists() and not args.smoke and not args.extend:
        prev = json.loads(done_marker.read_text() or "{}")
        LOG.info("already trained to %.1f epochs (%s exists) — skipping.",
                 prev.get("epochs_completed", 0), done_marker)
        LOG.info("To train LONGER from here without restarting:")
        LOG.info("    python finetune.py --model %s --extend --epochs 6 %s",
                 args.model, f"--tag {args.tag}" if args.tag else "")
        return out_dir / "best"

    # Any language works for the initial tokenizer state; per-example src_lang
    # is set in the dataset, which is what actually matters.
    first_lang = train_df["language"].iloc[0]
    model, tokenizer, _ = C.load_model_and_tokenizer(
        spec, spec.source_tokens[first_lang], for_training=True)

    mode = C.deep_get(cfg, "finetune.mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError(f"finetune.mode must be 'lora' or 'full', got {mode!r}")

    if C.deep_get(cfg, "finetune.freeze_embeddings", True):
        model = freeze_embeddings(model)
    if C.deep_get(cfg, "finetune.freeze_encoder", False):
        model = freeze_encoder(model)
    if mode == "lora":
        model = attach_lora(model, cfg)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train_params == 0:
        raise ValueError(
            "Nothing is trainable — you have frozen everything. Check "
            "finetune.freeze_embeddings / freeze_encoder / mode in config.yaml.")
    LOG.info("trainable parameters: %.2fM", n_train_params / 1e6)

    max_src = C.deep_get(cfg, "infer.max_source_length", 192)
    max_tgt = C.deep_get(cfg, "infer.max_target_length", 192)
    seed = C.deep_get(cfg, "data.seed", 42)
    LOG.info("tokenising train split")
    train_ds = build_dataset(train_df, tokenizer, spec, max_src, max_tgt,
                             "train", seed)
    LOG.info("tokenising dev split")
    dev_ds = build_dataset(dev_df, tokenizer, spec, max_src, max_tgt,
                           "dev", seed)

    collator = Seq2SeqCollator(tokenizer, model, label_pad_token_id=-100)

    dtype, dtype_name = C.pick_dtype()
    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16

    # Epoch budget: --epochs beats config, and --smoke beats both.
    epochs = (1 if args.smoke
              else (args.epochs or C.deep_get(cfg, "finetune.epochs", 3)))
    steps = 10 if args.smoke else C.deep_get(cfg, "finetune.eval_steps", 500)

    # Rough token count, for tokens/sec reporting.
    try:
        mean_len = float(sum(train_ds["length"])) / max(len(train_ds), 1)
    except Exception:
        mean_len = 0.0
    desired = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=float(C.deep_get(cfg, "finetune.lr", 3e-4)),
        per_device_train_batch_size=C.deep_get(cfg, "finetune.batch_size", 16),
        per_device_eval_batch_size=C.deep_get(cfg, "finetune.batch_size", 16),
        gradient_accumulation_steps=C.deep_get(cfg, "finetune.grad_accum", 2),
        weight_decay=C.deep_get(cfg, "finetune.weight_decay", 0.01),
        label_smoothing_factor=C.deep_get(cfg, "finetune.label_smoothing", 0.1),
        max_grad_norm=C.deep_get(cfg, "finetune.max_grad_norm", 1.0),
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=bool(C.deep_get(cfg, "finetune.use_cpu", False)),
        gradient_checkpointing=C.deep_get(
            cfg, "finetune.gradient_checkpointing", False),
        group_by_length=C.deep_get(cfg, "finetune.group_by_length", True),
        length_column_name="length",
        dataloader_num_workers=C.deep_get(cfg, "finetune.num_workers", 4),
        eval_strategy="steps",
        eval_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=C.deep_get(cfg, "finetune.save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1 if (args.smoke or args.verbose) else C.deep_get(
            cfg, "finetune.logging_steps", 25),
        logging_dir=str(out_dir / "logs"),
        report_to=C.available_reporters(),
        seed=seed,
        # Keep True: the sampler reads `length` from the dataset before the
        # Trainer strips columns the model's forward() does not accept.
        remove_unused_columns=True,
    )
    # transformers 4.x and 5.x disagree about several of these names.
    targs = C.build_training_args(
        Seq2SeqTrainingArguments, desired, n_train=len(train_ds),
        warmup_ratio=C.deep_get(cfg, "finetune.warmup_ratio", 0.03))

    patience = C.deep_get(cfg, "finetune.early_stopping_patience", 3)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    callbacks.append(make_stats_callback(out_dir, len(train_ds), mean_len))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    eff_batch = (targs.per_device_train_batch_size
                 * targs.gradient_accumulation_steps)
    LOG.info("-" * 62)
    LOG.info("TRAINING PLAN")
    LOG.info("  examples      : %d train / %d dev", len(train_ds), len(dev_ds))
    LOG.info("  mean src len  : %.1f tokens", mean_len)
    LOG.info("  mode          : %s (%.2fM trainable params)",
             mode, n_train_params / 1e6)
    LOG.info("  precision     : %s", dtype_name)
    LOG.info("  effective batch: %d (%d x %d accum)", eff_batch,
             targs.per_device_train_batch_size,
             targs.gradient_accumulation_steps)
    LOG.info("  epochs        : %s", epochs)
    LOG.info("  ~steps/epoch  : %d", max(1, len(train_ds) // eff_batch))
    LOG.info("  ~total steps  : %d", max(1, len(train_ds) // eff_batch) * int(epochs))
    LOG.info("  lr            : %g", targs.learning_rate)
    LOG.info("  eval/save every: %d steps", steps)
    LOG.info("  live log      : %s", out_dir / "training_log.csv")
    LOG.info("-" * 62)

    # Resume from the newest checkpoint if one exists. This is what makes
    # --extend work: weights, optimiser state, LR scheduler and data order are
    # all restored, so training continues rather than starting over.
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1])) \
        if (out_dir / "checkpoints").exists() else []
    resume = str(ckpts[-1]) if ckpts else None
    if resume:
        LOG.info("RESUMING from %s (step %s)", resume,
                 Path(resume).name.split("-")[1])
        if args.extend:
            LOG.info("--extend: raising the budget to %s epochs and continuing "
                     "from that step. The LR schedule is rebuilt for the new "
                     "total, so the tail of the curve differs slightly from a "
                     "single %s-epoch run — note that if you report both.",
                     epochs, epochs)
    elif args.extend:
        LOG.warning("--extend given but no checkpoint exists in %s — this will "
                    "train from scratch for %s epochs",
                    out_dir / "checkpoints", epochs)

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    mins = (time.time() - t0) / 60

    best = out_dir / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))

    final_eval = {k: round(float(v), 4) for k, v in trainer.evaluate().items()
                  if isinstance(v, (int, float))}

    steps_done = int(result.global_step)
    total_examples_seen = steps_done * eff_batch
    metrics = {
        # --- what ran ---
        "mode": mode,
        "epochs_requested": float(epochs),
        "epochs_completed": round(float(trainer.state.epoch or 0), 3),
        "steps": steps_done,
        "max_steps_planned": int(trainer.state.max_steps),
        "early_stopped": steps_done < int(trainer.state.max_steps),
        "resumed_from": resume,
        "extended": bool(args.extend),
        "per_language": False,
        # --- data ---
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "mean_src_tokens": round(mean_len, 1),
        "effective_batch": eff_batch,
        "examples_seen": total_examples_seen,
        "tokens_seen_approx": int(total_examples_seen * mean_len),
        # --- cost ---
        "train_minutes": round(mins, 1),
        "train_hours": round(mins / 60, 2),
        "sec_per_step": round(mins * 60 / max(steps_done, 1), 3),
        "examples_per_sec": round(total_examples_seen / max(mins * 60, 1e-6), 2),
        # --- quality ---
        "train_loss": round(float(result.training_loss), 4),
        "trainable_params_M": round(n_train_params / 1e6, 2),
        "precision": dtype_name,
        **final_eval,
        # --- curve ---
        **loss_curve_stats(out_dir / "training_log.csv"),
    }
    if torch.cuda.is_available():
        metrics["peak_gpu_mem_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)

    C.save_json(metrics, out_dir / "train_metrics.json")
    done_marker.write_text(json.dumps(metrics, indent=2))

    LOG.info("=" * 62)
    LOG.info("TRAINING SUMMARY")
    for k in ("epochs_completed", "steps", "early_stopped", "train_minutes",
              "sec_per_step", "examples_per_sec", "mean_tokens_per_s",
              "initial_loss", "final_loss", "loss_reduction", "loss_variance",
              "step_to_90pct_reduction", "best_eval_loss", "best_eval_step",
              "peak_gpu_mem_gb"):
        if k in metrics:
            LOG.info("  %-24s %s", k, metrics[k])
    LOG.info("  full loss curve          %s", out_dir / "training_log.csv")
    LOG.info("  all statistics           %s", out_dir / "train_metrics.json")
    LOG.info("=" * 62)
    LOG.info("checkpoint: %s", best)
    LOG.info("to train longer from here: python finetune.py --model %s "
             "--extend --epochs %d", args.model, int(epochs) * 2)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def main():
    global _RUN_DIR
    args = parse_args()
    cfg = C.apply_overrides(C.load_config(args.config), args.overrides)

    # DEFAULT IS ONE POOLED MODEL FOR ALL FOUR LANGUAGES.
    # Per-language is opt-in, via either `--per_language` or
    # `finetune.per_language: true` in config.yaml. The flag wins if given.
    per_language = bool(args.per_language
                        or C.deep_get(cfg, "finetune.per_language", False))

    setting = args.tag or ("finetune_per_lang" if per_language else "finetune")
    _RUN_DIR = C.run_dir(cfg, args.model, setting)
    C.setup_logging(_RUN_DIR, "finetune", label=f"{args.model}_{setting}")
    C.log_environment(_RUN_DIR)
    C.set_seed(C.deep_get(cfg, "data.seed", 42))

    LOG.info("=" * 62)
    LOG.info(" FINE-TUNE — %s [%s]", args.model, setting)
    LOG.info("=" * 62)
    LOG.info("scope: %s", "one model PER LANGUAGE" if per_language
             else "ONE POOLED MODEL for all languages (default)")

    spec = C.resolve_model(cfg, args.model)
    languages = args.languages or C.deep_get(cfg, "data.languages")

    data_path = C.find_data(cfg, args.data)
    df = C.load_parallel(data_path, languages)
    df = C.make_splits(df, cfg)

    train_all = df[df["split"] == "train"].reset_index(drop=True)
    dev_all = df[df["split"] == "dev"].reset_index(drop=True)

    cap = C.deep_get(cfg, "finetune.max_train_samples")
    if args.smoke:
        train_all = train_all.groupby("language", group_keys=False).head(50)
        dev_all = dev_all.groupby("language", group_keys=False).head(13)
        LOG.warning("SMOKE RUN — results are meaningless, this only proves "
                    "the pipeline runs end to end")
    elif cap:
        train_all = train_all.groupby("language", group_keys=False).head(int(cap))
        LOG.warning("max_train_samples=%s per language", cap)

    C.save_json({"config": {k: v for k, v in cfg.items()
                            if not k.startswith("_")},
                 "args": vars(args),
                 "n_train": len(train_all), "n_dev": len(dev_all)},
                _RUN_DIR / "resolved_config.json")

    if args.dry_run:
        LOG.info("dry run: %d train / %d dev. Exiting before model load.",
                 len(train_all), len(dev_all))
        return

    if per_language:
        for lang in languages:
            tr = train_all[train_all["language"] == lang]
            dv = dev_all[dev_all["language"] == lang]
            if tr.empty:
                LOG.warning("no train rows for %s — skipping", lang)
                continue
            LOG.info("-" * 62)
            LOG.info("language: %s (%d train / %d dev)", lang, len(tr), len(dv))
            train_one(spec, cfg, tr, dv, _RUN_DIR / lang, args)
        LOG.info("=" * 62)
        LOG.info(" next — one inference call per language:")
        for lang in languages:
            LOG.info("   python infer.py --model %s --ckpt %s/%s/best "
                     "--languages %s --tag ft_per_lang",
                     args.model, _RUN_DIR, lang, lang)
        LOG.info("=" * 62)
    else:
        # One model, all languages. Each example still carries its own source
        # language token, so the model knows which language it is reading.
        best = train_one(spec, cfg, train_all, dev_all, _RUN_DIR, args)
        LOG.info("=" * 62)
        LOG.info(" one pooled model trained on %d examples across %s",
                 len(train_all), sorted(set(train_all["language"])))
        LOG.info(" next:")
        LOG.info("   python infer.py --model %s --ckpt %s --tag ft_lora",
                 args.model, best)
        LOG.info("   python evaluate.py --all --name mt_all")
        LOG.info("=" * 62)


if __name__ == "__main__":
    C.run_guarded(main, lambda: _RUN_DIR)
