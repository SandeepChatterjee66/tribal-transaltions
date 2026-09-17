#!/usr/bin/env python3
"""
consolidate.py — gather everything from runs/ into flat TSV/CSV tables.

Walks every runs/<model>/<setting>/ directory and produces one place to look
for results, one place to look for raw translations, and a paper-shaped table.

    python consolidate.py                     # -> consolidated/
    python consolidate.py --out /scratch/out
    python consolidate.py --no_translations   # skip the big file

EVERY table below is written as both `.tsv` and `.csv`. TSV is the primary
format; the CSV is a convenience copy. `paper_table.tex` is the only
LaTeX output, and it is secondary.

Outputs in consolidated/ (each as .tsv AND .csv):

  paper_table            wide: rows = model x setting, cols = lang x metric
  zeroshot_vs_ft         per language: zero-shot, fine-tuned, and the delta
  all_scores             one row per model x setting x language, with CIs
  training_stats         per training run: convergence, cost, throughput
  loss_curves            every logged training point, all runs
  inference_stats        per language decode cost
  src_token_sweep        every source-token candidate scored (if swept)
  all_translations       every prediction from every run, stacked
  run_manifest           what ran where: node, GPU, versions, timings, status
  paper_table.tex        LaTeX version of paper_table (secondary)
  README_RESULTS.md      plain-language summary of what is in each file

Safe to run at any time, including mid-experiment — anything unfinished is
reported as such rather than skipped silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import common as C
from common import LOG

LANG_SHORT = {"Bhili": "bhb", "Gondi": "gon", "Mundari": "unr", "Santali": "sat"}
METRICS = ["chrf++", "spBLEU", "ROUGE-L", "BLEU"]


def parse_args():
    p = argparse.ArgumentParser(description="Consolidate all runs into CSVs")
    p.add_argument("--config", default=None)
    p.add_argument("--runs", default=None, help="override output.root")
    p.add_argument("--out", default="consolidated")
    p.add_argument("--no_translations", action="store_true",
                   help="skip all_translations.csv (it can get large)")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    return p.parse_args()


def write_table(df: pd.DataFrame, out: Path, name: str) -> None:
    """Write every table as BOTH .tsv and .csv.

    TSV is the primary format — tabs survive commas inside translated text,
    which CSV quoting handles but which makes the file annoying to eyeball or
    paste into a spreadsheet. The .csv is there for anything that insists on it.
    """
    df.to_csv(out / f"{name}.tsv", sep="\t", index=False)
    df.to_csv(out / f"{name}.csv", index=False)
    LOG.info("%-24s %6d rows  -> %s.{tsv,csv}", name, len(df), name)


def order_languages(langs) -> list[str]:
    """Corpus order (bhb, gon, unr, sat) with MACRO last.

    Alphabetical would put MACRO between Gondi and Mundari, which reads as if
    it were another language.
    """
    canonical = list(LANG_SHORT)
    langs = set(langs)
    out = [l for l in canonical if l in langs]
    out += sorted(l for l in langs if l not in canonical and l != "MACRO")
    if "MACRO" in langs:
        out.append("MACRO")
    return out


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Read every run directory. Returns (scores, manifest, run_dirs)."""
    score_rows, manifest_rows, run_dirs = [], [], []

    for run_dir in sorted(runs_root.glob("*/*")):
        if not run_dir.is_dir():
            continue
        model, setting = run_dir.parent.name, run_dir.name
        pred = run_dir / "predictions.csv"
        metrics = run_dir / "metrics.json"
        env = read_json(run_dir / "environment.json")
        train = read_json(run_dir / "train_metrics.json")

        status = "ok"
        n_pred = 0
        trained = bool(train) or (run_dir / "DONE").exists()
        if (run_dir / "error.log").exists():
            status = "CRASHED"
        elif not pred.exists():
            # A training-only directory holds a checkpoint, not predictions —
            # say so rather than making it look like a failure.
            status = ("trained — run infer.py with this checkpoint"
                      if trained else "no predictions")
        else:
            try:
                n_pred = len(pd.read_csv(pred))
            except Exception as e:
                status = f"unreadable predictions: {type(e).__name__}"
        if pred.exists() and not metrics.exists():
            status = "not scored (run evaluate.py)"

        infer_stats = read_json(run_dir / "inference_stats.json")
        gpus = env.get("gpus") or []
        manifest_rows.append({
            "model": model, "setting": setting, "status": status,
            "n_predictions": n_pred,
            "hostname": env.get("hostname", ""),
            "gpu": gpus[0]["name"] if gpus else "",
            "torch": env.get("torch", ""),
            "transformers": env.get("transformers", ""),
            "train_minutes": train.get("train_minutes", ""),
            "epochs_completed": train.get("epochs_completed", ""),
            "train_loss": train.get("train_loss", ""),
            "ft_mode": train.get("mode", ""),
            "n_train": train.get("n_train", ""),
            "infer_minutes": infer_stats.get("translate_minutes", ""),
            "sent_per_sec": infer_stats.get("overall_sentences_per_sec", ""),
            "path": str(run_dir),
        })
        run_dirs.append(run_dir)

        m = read_json(metrics)
        for lang, v in (m.get("per_language") or {}).items():
            score_rows.append({
                "model": model, "setting": setting, "language": lang,
                "code": LANG_SHORT.get(lang, lang[:3].lower()),
                "n": v.get("n", 0),
                "empty_pct": v.get("empty_pct", 0.0),
                **{k: v.get(k) for k in METRICS},
                "chrf++_ci_lo": v.get("chrf++_ci_lo"),
                "chrf++_ci_hi": v.get("chrf++_ci_hi"),
                "src_lang": ",".join(v.get("src_lang", [])),
            })
        macro = m.get("macro") or {}
        if macro:
            score_rows.append({
                "model": model, "setting": setting, "language": "MACRO",
                "code": "avg", "n": macro.get("n_total", 0),
                "empty_pct": macro.get("empty_pct", 0.0),
                **{k: macro.get(k) for k in METRICS},
                "chrf++_ci_lo": None, "chrf++_ci_hi": None, "src_lang": "",
            })

    return pd.DataFrame(score_rows), pd.DataFrame(manifest_rows), run_dirs


def wide_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Rows = model x setting, columns = <lang>_<metric>. Paper shape."""
    sub = scores[scores["language"] != "MACRO"]
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot_table(index=["model", "setting"], columns="code",
                          values=METRICS, aggfunc="first")
    piv.columns = [f"{code}_{metric}" for metric, code in piv.columns]
    order = [f"{c}_{m}" for c in ("bhb", "gon", "unr", "sat")
             for m in ("ROUGE-L", "chrf++", "spBLEU")]
    piv = piv[[c for c in order if c in piv.columns]]

    macro = scores[scores["language"] == "MACRO"].set_index(["model", "setting"])
    for m in ("chrf++", "spBLEU", "ROUGE-L"):
        if m in macro.columns:
            piv[f"MACRO_{m}"] = macro[m]
    return piv.reset_index()


def zeroshot_vs_ft(scores: pd.DataFrame) -> pd.DataFrame:
    """The comparison the reviewers actually asked for, as a flat table."""
    rows = []
    zs = scores[scores["setting"] == "zeroshot"]
    # `sweep` runs are zero-shot probes on the dev split, not fine-tuned
    # systems — comparing them here would be meaningless.
    ft_settings = {s for s in set(scores["setting"])
                   if s != "zeroshot" and "sweep" not in s}
    for setting in sorted(ft_settings):
        ft = scores[scores["setting"] == setting]
        for model in sorted(set(ft["model"])):
            for lang in order_languages(set(ft["language"])):
                a = zs[(zs.model == model) & (zs.language == lang)]
                b = ft[(ft.model == model) & (ft.language == lang)]
                if a.empty or b.empty:
                    continue
                row = {"model": model, "ft_setting": setting, "language": lang}
                for m in ("chrf++", "spBLEU", "ROUGE-L"):
                    av, bv = a[m].iloc[0], b[m].iloc[0]
                    row[f"{m}_zeroshot"] = av
                    row[f"{m}_finetuned"] = bv
                    row[f"{m}_delta"] = (round(bv - av, 2)
                                         if pd.notna(av) and pd.notna(bv) else None)
                rows.append(row)
    return pd.DataFrame(rows)


def sweep_table(run_dirs: list[Path], cfg: dict) -> pd.DataFrame:
    """Score each by_group/<lang>__<token> partial so the proxy-token choice
    is auditable rather than asserted."""
    rows = []
    for rd in run_dirs:
        bg = rd / "by_group"
        if not bg.exists():
            continue
        for sub in sorted(bg.iterdir()):
            pred = sub / "predictions.csv"
            if not sub.is_dir() or not pred.exists() or "__" not in sub.name:
                continue
            try:
                df = pd.read_csv(pred)
            except Exception:
                continue
            lang, token = sub.name.split("__", 1)
            hyps = C.as_text(df["hyp"], treat_placeholders_as_empty=True)
            refs = C.as_text(df["english"])
            m = C.compute_metrics(hyps, refs)
            rows.append({
                "model": rd.parent.name, "setting": rd.name,
                "language": lang, "src_token": token,
                "n": m.get("n", 0),
                "empty_pct": round(100 * sum(1 for h in hyps if not h)
                                   / max(len(hyps), 1), 2),
                **{k: m.get(k) for k in METRICS},
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model", "language", "chrf++"],
                            ascending=[True, True, False])
        df["rank_in_language"] = df.groupby(
            ["model", "language"]).cumcount() + 1
    return df


def training_stats_table(run_dirs: list[Path]) -> pd.DataFrame:
    """One row per training run: convergence, stability, cost.

    Mirrors the convergence / loss-variance / efficiency tables the paper
    already reports for the encoders, so the MT rows can go in the same shape.
    """
    keys = ["mode", "epochs_requested", "epochs_completed", "steps",
            "max_steps_planned", "early_stopped", "extended", "n_train",
            "n_dev", "effective_batch", "mean_src_tokens", "examples_seen",
            "tokens_seen_approx", "trainable_params_M", "precision",
            "train_minutes", "train_hours", "sec_per_step", "examples_per_sec",
            "mean_samples_per_s", "mean_tokens_per_s", "peak_gpu_mem_gb",
            "initial_loss", "final_loss", "min_loss", "loss_reduction",
            "loss_variance", "loss_std", "step_to_90pct_reduction",
            "best_eval_loss", "best_eval_step", "final_eval_loss", "n_evals",
            "train_loss"]
    rows = []
    for rd in run_dirs:
        m = read_json(rd / "train_metrics.json")
        if not m:
            continue
        rows.append({"model": rd.parent.name, "setting": rd.name,
                     **{k: m.get(k) for k in keys}})
    return pd.DataFrame(rows)


def loss_curves_table(run_dirs: list[Path]) -> pd.DataFrame:
    """Every logged training point from every run, stacked. Plot straight from
    this: one line per (model, setting)."""
    frames = []
    for rd in run_dirs:
        f = rd / "training_log.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df.insert(0, "model", rd.parent.name)
        df.insert(1, "setting", rd.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def inference_stats_table(run_dirs: list[Path]) -> pd.DataFrame:
    """Per-language decoding cost for every inference run."""
    rows = []
    for rd in run_dirs:
        s = read_json(rd / "inference_stats.json")
        if not s:
            continue
        for g in s.get("per_group", []):
            rows.append({
                "model": rd.parent.name, "setting": rd.name,
                "group": g.get("group"), "language": g.get("language"),
                "src_lang": g.get("src_lang"),
                "n": g.get("n_total_in_file"),
                "minutes": g.get("minutes"),
                "sentences_per_sec": g.get("sentences_per_sec"),
                "sec_per_sentence": g.get("sec_per_sentence"),
                "output_tokens_per_sec": g.get("output_tokens_per_sec"),
                "batch_size": g.get("batch_size"),
                "num_beams": g.get("num_beams"),
                "peak_gpu_mem_gb": g.get("peak_gpu_mem_gb"),
                "empty_pct": g.get("empty_pct"),
            })
    return pd.DataFrame(rows)


def stack_translations(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for rd in run_dirs:
        pred = rd / "predictions.csv"
        if not pred.exists():
            continue
        try:
            df = pd.read_csv(pred)
        except Exception as e:
            LOG.warning("skipping unreadable %s: %s", pred, e)
            continue
        df.insert(0, "model", rd.parent.name)
        df.insert(1, "setting", rd.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tex_escape(s) -> str:
    r"""Escape LaTeX specials in a cell.

    Run names like `ft_lora` and `nllb-1.3b` go straight into the table, and a
    bare underscore in text mode is a hard compile error ("Missing $ inserted"),
    so the generated .tex would not build if pasted into the paper.
    """
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def write_latex(wide: pd.DataFrame, path: Path) -> None:
    if wide.empty:
        path.write_text("% no results yet\n")
        return
    cols = [c for c in wide.columns if c not in ("model", "setting")
            and not c.startswith("MACRO")]
    lines = [
        "% generated by consolidate.py — underscores are escaped, so this",
        "% compiles as-is. Wrap in \\begin{table} yourself.",
        "\\begin{tabular}{ll" + "r" * len(cols) + "}",
        "\\toprule",
        "Model & Setting & " + " & ".join(
            tex_escape(c.replace("_", " ")) for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, r in wide.iterrows():
        cells = [tex_escape(r["model"]), tex_escape(r["setting"])]
        cells += ["--" if pd.isna(r[c]) else f"{r[c]:.2f}" for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines))


RESULTS_README = """# Consolidated results

Generated by `consolidate.py`. Re-run it any time to refresh.

**Every table is written twice: `<name>.tsv` and `<name>.csv`.** Same contents.
TSV is the primary format. Only `paper_table` additionally has a `.tex`.

| File | What it is |
|---|---|
| `paper_table.tsv` / `.csv` | Wide layout: rows = model x setting, columns = language x metric. **This is the table for the paper.** |
| `zeroshot_vs_ft.tsv` / `.csv` | Per language: zero-shot, fine-tuned, and the delta. |
| `all_scores.tsv` / `.csv` | One row per model x setting x language, with bootstrap CIs. The source of truth. |
| `src_token_sweep.tsv` / `.csv` | Every source-language proxy token scored and ranked. |
| `all_translations.tsv` / `.csv` | Every prediction from every run, stacked. For error analysis. |
| `run_manifest.tsv` / `.csv` | What ran where: node, GPU, versions, wall-clock, status. |
| `paper_table.tex` | LaTeX version of `paper_table`, underscores escaped so it compiles. Secondary. |
| `training_stats.csv` | Per training run: epochs, steps, wall-clock, throughput, peak GPU memory, initial/final/min loss, loss variance, step to 90% of the loss drop, best eval loss. |
| `loss_curves.csv` | Every logged training point from every run. Plot one line per `(model, setting)`. |
| `inference_stats.csv` | Per language: decode minutes, sentences/sec, output tokens/sec, peak GPU memory, beams, batch size. |

## Read these two columns first

**`empty_pct`** — the share of outputs that were blank or `ERROR`. Anything
above ~2% means the score is being dragged down by generation failures rather
than translation quality. Re-run those rows (`infer.py` only redoes what is
missing) before putting the number in the paper.

**`status`** in `run_manifest.csv` — `CRASHED` means there is an `error.log` in
that run directory. `not scored` means predictions exist but `evaluate.py` has
not been run on them yet.

## Caveats to carry into the writeup

- `src_lang` records the source-language token used. Bhili, Gondi and Mundari
  are in none of these models, so those rows use a Devanagari proxy. Say so.
- NLLB's only Santali is `sat_Beng` (Bengali script) while TribalCorp's Santali
  is Ol Chiki. IndicTrans2 is the only model here with `sat_Olck`.
- `chrf++_ci_lo/hi` is a 1000-round bootstrap CI. Overlapping intervals mean
  the difference is not resolvable at this sample size.
"""


def main():
    args = parse_args()
    cfg = C.apply_overrides(C.load_config(args.config), args.overrides)
    out = C._anchor(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C.setup_logging(out, "consolidate")

    runs_root = C._anchor(args.runs) if args.runs else C.runs_root(cfg)
    if not runs_root.exists():
        raise SystemExit(f"No runs directory at {runs_root}. Run infer.py first.")

    LOG.info("scanning %s", runs_root)
    scores, manifest, run_dirs = collect(runs_root)

    if manifest.empty:
        raise SystemExit(f"No run directories found under {runs_root}.")

    write_table(manifest, out, "run_manifest")
    bad = manifest[manifest["status"] != "ok"]
    if not bad.empty:
        LOG.warning("%d run(s) need attention:\n%s", len(bad),
                    bad[["model", "setting", "status"]].to_string(index=False))

    if scores.empty:
        LOG.warning("no metrics.json anywhere — run: python evaluate.py --all")
    else:
        scores = scores.sort_values(["model", "setting", "language"])
        write_table(scores, out, "all_scores")

        wide = wide_table(scores)
        write_table(wide, out, "paper_table")
        write_latex(wide, out / "paper_table.tex")   # secondary

        cmp_df = zeroshot_vs_ft(scores)
        if not cmp_df.empty:
            write_table(cmp_df, out, "zeroshot_vs_ft")
        else:
            LOG.info("zeroshot_vs_ft.csv skipped — need both settings scored")

    ts = training_stats_table(run_dirs)
    if not ts.empty:
        write_table(ts, out, "training_stats")
        cols = [c for c in ("model", "setting", "epochs_completed", "steps",
                            "train_minutes", "initial_loss", "final_loss",
                            "best_eval_loss", "peak_gpu_mem_gb")
                if c in ts.columns]
        LOG.info("training summary:\n%s", ts[cols].to_string(index=False))

    curves = loss_curves_table(run_dirs)
    if not curves.empty:
        write_table(curves, out, "loss_curves")

    inf = inference_stats_table(run_dirs)
    if not inf.empty:
        write_table(inf, out, "inference_stats")

    sweep = sweep_table(run_dirs, cfg)
    if not sweep.empty:
        write_table(sweep, out, "src_token_sweep")
        best = sweep[sweep["rank_in_language"] == 1]
        LOG.info("best source token per language:\n%s",
                 best[["model", "language", "src_token", "chrf++"]]
                 .to_string(index=False))

    if not args.no_translations:
        tr = stack_translations(run_dirs)
        if not tr.empty:
            write_table(tr, out, "all_translations")

    (out / "README_RESULTS.md").write_text(RESULTS_README)

    LOG.info("=" * 62)
    LOG.info(" consolidated -> %s/", out)
    for f in sorted(out.glob("*")):
        if f.is_file():
            LOG.info("   %-24s %8.1f KB", f.name, f.stat().st_size / 1024)
    LOG.info("=" * 62)


if __name__ == "__main__":
    C.run_guarded(main, lambda: Path(parse_args().out))
