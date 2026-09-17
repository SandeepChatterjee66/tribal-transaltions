#!/usr/bin/env python3
"""
evaluate.py — score predictions and emit the paper table.

  # one run
  python evaluate.py --run runs/nllb-600m/zeroshot

  # every run under runs/ into one combined table
  python evaluate.py --all

  # compare the source-token sweep to justify the proxy choice
  python evaluate.py --run runs/nllb-600m/zeroshot --by_group

Outputs, per run and combined, into tables/:
  <name>_metrics.json   full per-language numbers + bootstrap CIs
  <name>.tsv / .csv     tidy table
  <name>.tex            LaTeX booktabs body, paste-ready

Also reports the share of empty/failed outputs per language. A silently high
empty rate is what makes a baseline look artificially weak, so it is printed
next to every score rather than buried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import common as C
from common import LOG

_OUT: Path | None = None

LANG_SHORT = {"Bhili": "bhb", "Gondi": "gon", "Mundari": "unr", "Santali": "sat"}
METRIC_ORDER = ["chrf++", "spBLEU", "ROUGE-L", "BLEU"]


def parse_args():
    p = argparse.ArgumentParser(description="Score MT predictions")
    p.add_argument("--run", default=None, help="a runs/<model>/<setting> dir")
    p.add_argument("--all", action="store_true",
                   help="score every run found under output.root")
    p.add_argument("--config", default=None)
    p.add_argument("--by_group", action="store_true",
                   help="also score each by_group/ subdir separately "
                        "(use with --sweep_src runs)")
    p.add_argument("--name", default=None, help="output basename")
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    return p.parse_args()


def clean(series) -> list[str]:
    """Normalise hypotheses. Empty and API-error placeholders become ''
    so they score zero and get counted, never silently dropped."""
    return C.as_text(series, treat_placeholders_as_empty=True)


def score_frame(df: pd.DataFrame, cfg: dict, bootstrap: bool) -> dict:
    """Per-language + macro metrics for one predictions frame."""
    rounds = 0 if not bootstrap else C.deep_get(cfg, "metrics.bootstrap_rounds", 1000)
    seed = C.deep_get(cfg, "metrics.seed", 42)

    out: dict = {"per_language": {}}
    for lang, g in df.groupby("language", sort=True):
        hyps = clean(g["hyp"])
        refs = C.as_text(g["english"])
        n_empty = sum(1 for h in hyps if h == "")

        m = C.compute_metrics(hyps, refs)
        m["empty_outputs"] = n_empty
        m["empty_pct"] = round(100.0 * n_empty / max(len(hyps), 1), 2)
        if "src_lang" in g.columns:
            m["src_lang"] = sorted(set(g["src_lang"].astype(str)))
        if rounds:
            m.update(C.bootstrap_ci(hyps, refs, rounds, seed, "chrf++"))
        out["per_language"][lang] = m

        if m["empty_pct"] > 2.0:
            LOG.warning("%s: %.1f%% of outputs are empty/ERROR — scores are "
                        "depressed by generation failures, not translation "
                        "quality. Re-run those items before reporting.",
                        lang, m["empty_pct"])

    langs = list(out["per_language"])
    for metric in METRIC_ORDER:
        vals = [out["per_language"][l][metric] for l in langs
                if metric in out["per_language"][l]]
        if vals:
            out.setdefault("macro", {})[metric] = round(sum(vals) / len(vals), 2)
    out["macro"]["n_total"] = int(len(df))
    out["macro"]["empty_pct"] = round(
        100.0 * sum(out["per_language"][l]["empty_outputs"] for l in langs)
        / max(len(df), 1), 2)
    return out


def find_runs(cfg: dict) -> list[Path]:
    root = C.runs_root(cfg)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/*/predictions.csv"))


def tidy_rows(model: str, setting: str, scored: dict) -> list[dict]:
    rows = []
    for lang, m in scored["per_language"].items():
        rows.append({
            "model": model,
            "setting": setting,
            "language": lang,
            "code": LANG_SHORT.get(lang, lang[:3].lower()),
            "n": m.get("n", 0),
            "empty_%": m.get("empty_pct", 0.0),
            **{k: m.get(k) for k in METRIC_ORDER},
            "chrf++_ci": (f"[{m['chrf++_ci_lo']}, {m['chrf++_ci_hi']}]"
                          if "chrf++_ci_lo" in m else ""),
            "src_lang": ",".join(m.get("src_lang", [])),
        })
    macro = scored.get("macro", {})
    rows.append({
        "model": model, "setting": setting, "language": "MACRO", "code": "avg",
        "n": macro.get("n_total", 0), "empty_%": macro.get("empty_pct", 0.0),
        **{k: macro.get(k) for k in METRIC_ORDER},
        "chrf++_ci": "", "src_lang": "",
    })
    return rows


def tex_escape(s) -> str:
    r"""Escape LaTeX specials. Run names like `ft_lora` contain underscores,
    which are a hard compile error in text mode."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def write_latex(df: pd.DataFrame, path: Path, primary: str = "chrf++") -> None:
    """Wide LaTeX table: one row per model/setting, columns = language x metric.
    Matches the layout of the existing MT tables in the paper."""
    langs = [l for l in LANG_SHORT if l in set(df["language"])]
    metrics = ["ROUGE-L", "chrf++", "spBLEU"]

    header = ["Model"]
    for l in langs:
        header += [f"{LANG_SHORT[l]} {m}" for m in metrics]
    lines = [
        "% generated by evaluate.py — paste into the paper",
        "% primary metric: " + primary,
        "\\begin{tabular}{l" + "r" * (len(langs) * len(metrics)) + "}",
        "\\toprule",
        " & ".join(tex_escape(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    for (model, setting), g in df[df["language"] != "MACRO"].groupby(
            ["model", "setting"], sort=True):
        cells = [tex_escape(f"{model} ({setting})")]
        for l in langs:
            row = g[g["language"] == l]
            for m in metrics:
                cells.append(f"{row[m].iloc[0]:.2f}" if len(row) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines))


def main():
    global _OUT
    args = parse_args()
    cfg = C.apply_overrides(C.load_config(args.config), args.overrides)
    _OUT = C.tables_dir(cfg)
    _OUT.mkdir(parents=True, exist_ok=True)
    C.setup_logging(_OUT, "evaluate", label=args.name or "all")

    runs = find_runs(cfg) if args.all else (
        [Path(args.run)] if args.run else [])
    if not runs:
        raise SystemExit(
            "Nothing to score. Pass --run runs/<model>/<setting> or --all.\n"
            "Run infer.py first.")

    LOG.info("scoring %d run(s)", len(runs))
    all_rows: list[dict] = []
    combined: dict = {}

    for rd in runs:
        pred = rd / "predictions.csv"
        if not pred.exists():
            LOG.warning("no predictions.csv in %s — skipping", rd)
            continue
        model, setting = rd.parent.name, rd.name
        df = pd.read_csv(pred)
        LOG.info("-" * 62)
        LOG.info("%s [%s] — %d predictions", model, setting, len(df))

        scored = score_frame(df, cfg, not args.no_bootstrap)
        combined[f"{model}/{setting}"] = scored
        C.save_json(scored, rd / "metrics.json")
        all_rows += tidy_rows(model, setting, scored)

        for lang, m in scored["per_language"].items():
            LOG.info("  %-9s n=%-6d chrF++=%-6.2f spBLEU=%-6.2f "
                     "ROUGE-L=%-6.2f empty=%.1f%%",
                     lang, m.get("n", 0), m.get("chrf++", 0), m.get("spBLEU", 0),
                     m.get("ROUGE-L", 0), m.get("empty_pct", 0))
        LOG.info("  MACRO     chrF++=%.2f", scored["macro"].get("chrf++", 0))

        if args.by_group and (rd / "by_group").exists():
            grp_rows = []
            for sub in sorted((rd / "by_group").iterdir()):
                p = sub / "predictions.csv"
                if not p.exists():
                    continue
                gdf = pd.read_csv(p)
                gs = score_frame(gdf, cfg, False)
                for lang, m in gs["per_language"].items():
                    grp_rows.append({
                        "group": sub.name, "language": lang,
                        "src_lang": ",".join(m.get("src_lang", [])),
                        "n": m.get("n", 0),
                        **{k: m.get(k) for k in METRIC_ORDER},
                    })
            if grp_rows:
                gdf = pd.DataFrame(grp_rows).sort_values(
                    ["language", "chrf++"], ascending=[True, False])
                out = _OUT / f"{model}_{setting}_src_sweep.tsv"
                gdf.to_csv(out, sep="\t", index=False)
                LOG.info("  source-token sweep -> %s", out)
                LOG.info("  best source token per language:\n%s",
                         gdf.groupby("language").head(1).to_string(index=False))

    if not all_rows:
        raise SystemExit("No runs produced scores.")

    tidy = pd.DataFrame(all_rows)
    name = args.name or ("all_models" if len(runs) > 1 else
                         f"{runs[0].parent.name}_{runs[0].name}")
    tidy.to_csv(_OUT / f"{name}.tsv", sep="\t", index=False)
    tidy.to_csv(_OUT / f"{name}.csv", index=False)
    C.save_json(combined, _OUT / f"{name}_metrics.json")
    write_latex(tidy, _OUT / f"{name}.tex",
                C.deep_get(cfg, "metrics.primary", "chrf++"))

    LOG.info("=" * 62)
    LOG.info(" tables written to %s/", _OUT)
    for ext in ("tsv", "csv", "tex"):
        LOG.info("   %s.%s", name, ext)
    LOG.info("=" * 62)
    print()
    print(tidy.to_string(index=False))


if __name__ == "__main__":
    C.run_guarded(main, lambda: _OUT)
