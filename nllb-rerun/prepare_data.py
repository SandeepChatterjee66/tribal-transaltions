#!/usr/bin/env python3
"""
prepare_data.py — turn raw_data/ into the files everything else reads.

Drop the four corpus CSVs into `raw_data/` and run this. No paths to configure:
every location is derived from where this file lives, so the whole directory can
be moved or copied to another machine and still work.

    raw_data/Bhili.csv        columns: English,Hindi,Bhili
    raw_data/Gondi.csv        columns: English,Hindi,Gondi
    raw_data/Mundari.csv      columns: English,Hindi,Mundari
    raw_data/Santali.csv      columns: English,Hindi,Santali

                    |
                    v

    data/processed/D1_parallel.csv     15% held out  <- what the MT stage uses
    data/processed/D0_mlm.csv          85% for continued pretraining
    data/processed/dataset_stats.json  row counts and provenance

THE SPLIT
---------
85/15 per language at seed 42, reproducing the partition used for the published
experiments. To stay byte-identical with that partition, three details matter and
are preserved here: the RNG is seeded ONCE before the loop (not per language),
the languages are processed in a fixed order, and each language's rows are
shuffled with `random.shuffle` on the full row list. Changing any of those
reshuffles the corpus and the held-out set stops matching.

D0 is emitted long-form with the English, Hindi and tribal sides as separate
rows, then shuffled, for masked-language-model pretraining. D1 keeps the sides
aligned as triplets.

USAGE
    python prepare_data.py                  # normal use, nothing to configure
    python prepare_data.py --force          # regenerate over existing outputs
    python prepare_data.py --skip_d0        # only D1 (all the MT stage needs)
    python prepare_data.py --verify         # check existing outputs, write nothing

Part of the TribalSuite release.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================
# PATHS — all relative to this file, so the directory is portable
# ============================================================

HERE = Path(__file__).resolve().parent

RAW_DIR = HERE / "raw_data"
OUT_DIR = HERE / "data" / "processed"
D1_PATH = OUT_DIR / "D1_parallel.csv"
D0_PATH = OUT_DIR / "D0_mlm.csv"
STATS_PATH = OUT_DIR / "dataset_stats.json"

# Fixed order. The shuffle draws from one seeded stream, so reordering these
# changes every split.
LANGUAGES = ["Bhili", "Gondi", "Mundari", "Santali"]

SEED = 42
D1_RATIO = 0.15

# csv fields can be long in this corpus; raise the limit rather than crash.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--raw_dir", default=str(RAW_DIR),
                   help=f"where the raw CSVs live (default: {RAW_DIR.name}/)")
    p.add_argument("--out_dir", default=str(OUT_DIR),
                   help="where to write the processed files")
    p.add_argument("--languages", nargs="+", default=LANGUAGES,
                   help="languages to include, in split order")
    p.add_argument("--d1_ratio", type=float, default=D1_RATIO,
                   help="fraction held out as D1 (default 0.15)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip_d0", action="store_true",
                   help="write only D1; the MT stage needs nothing else")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing outputs")
    p.add_argument("--verify", action="store_true",
                   help="report on existing outputs and exit")
    return p.parse_args()


# ============================================================
# CHECKS
# ============================================================

def check_raw(raw_dir: Path, languages: list[str]) -> dict[str, Path]:
    """Locate one CSV per language and validate its header before reading
    hundreds of thousands of rows."""
    if not raw_dir.is_dir():
        raise SystemExit(
            f"\nNo raw data directory at:\n    {raw_dir}\n\n"
            f"Create it and drop the four corpus CSVs in:\n"
            + "".join(f"    {raw_dir.name}/{l}.csv\n" for l in languages))

    found, missing = {}, []
    for lang in languages:
        # Tolerate case variations in the filename.
        hits = [p for p in raw_dir.glob("*.csv") if p.stem.lower() == lang.lower()]
        if hits:
            found[lang] = hits[0]
        else:
            missing.append(lang)

    if missing:
        present = sorted(p.name for p in raw_dir.glob("*.csv"))
        raise SystemExit(
            f"\nMissing raw file(s) for: {', '.join(missing)}\n"
            f"Looked in: {raw_dir}\n"
            f"Found there: {present or 'nothing'}\n\n"
            f"Expected one CSV per language, named after it:\n"
            + "".join(f"    {raw_dir.name}/{l}.csv\n" for l in missing))

    for lang, path in found.items():
        with open(path, encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        cols = {c.strip().lower() for c in header}
        need = {"english", lang.lower()}
        if not need <= cols:
            raise SystemExit(
                f"\n{path.name} is missing required column(s) "
                f"{sorted(need - cols)}.\n"
                f"Header found: {header}\n"
                f"Expected: English,Hindi,{lang}")
        if "hindi" not in cols:
            print(f"   note: {path.name} has no Hindi column — D1 will carry an "
                  f"empty hindi field (unused by the MT stage)")

    return found


def verify(out_dir: Path, languages: list[str]) -> int:
    """Report on existing outputs without touching them."""
    d1 = out_dir / "D1_parallel.csv"
    print("\n" + "=" * 70)
    print(" VERIFYING EXISTING OUTPUTS")
    print("=" * 70)

    if not d1.exists():
        print(f"\n   {d1} does not exist. Run: python prepare_data.py")
        return 1

    import collections
    counts = collections.Counter()
    empty = 0
    with open(d1, encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        for row in r:
            counts[row.get("language", "?")] += 1
            if not (row.get("english", "").strip()
                    and row.get("tribal", "").strip()):
                empty += 1

    print(f"\n   file    : {d1}")
    print(f"   columns : {cols}")
    print(f"   rows    : {sum(counts.values()):,}")
    for lang in languages:
        print(f"     {lang:9s} {counts.get(lang, 0):,}")
    other = set(counts) - set(languages)
    if other:
        print(f"   unexpected languages: {sorted(other)}")
    if empty:
        print(f"   ⚠️  {empty:,} rows with an empty english or tribal side")

    required = {"language", "english", "tribal"}
    if not required <= set(cols):
        print(f"\n   ❌ missing required column(s): {sorted(required - set(cols))}")
        return 1

    stats = out_dir / "dataset_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        print(f"\n   generated with seed={s.get('seed')} "
              f"d1_ratio={s.get('d1_ratio')}")

    print("\n   ✅ D1 looks usable. Next: python check_env.py")
    return 0


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.verify:
        raise SystemExit(verify(out_dir, args.languages))

    d1_path = out_dir / "D1_parallel.csv"
    d0_path = out_dir / "D0_mlm.csv"

    if d1_path.exists() and not args.force:
        print(f"\n⏭️  {d1_path} already exists.")
        print("    Pass --force to regenerate, or --verify to inspect it.")
        print("    Nothing else needs configuring — check_env.py will find it.")
        return

    print("\n" + "=" * 70)
    print(" TRIBALSUITE — DATA PREPARATION")
    print("=" * 70)
    print(f"  raw data   : {raw_dir}")
    print(f"  output     : {out_dir}")
    print(f"  languages  : {', '.join(args.languages)}  (split order)")
    print(f"  seed       : {args.seed}")
    print(f"  D1 ratio   : {args.d1_ratio}  (D0 gets {1 - args.d1_ratio:.2f})")
    print("=" * 70)

    files = check_raw(raw_dir, args.languages)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed ONCE, before the loop. Each language's shuffle continues the same
    # stream, which is what the published partition did.
    random.seed(args.seed)

    d0_rows: list[dict] = []
    d1_rows: list[dict] = []
    stats: dict = defaultdict(int)
    per_lang: dict = {}

    for lang in args.languages:
        path = files[lang]
        print(f"\n📄 {path.name}")

        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"   rows read         : {len(rows):,}")

        random.shuffle(rows)

        split_idx = int(len(rows) * (1 - args.d1_ratio))
        d0_chunk, d1_chunk = rows[:split_idx], rows[split_idx:]
        print(f"   -> D0 (pretrain)  : {len(d0_chunk):,}")
        print(f"   -> D1 (held out)  : {len(d1_chunk):,}")

        if not args.skip_d0:
            for r in d0_chunk:
                eng = (r.get("English") or "").strip()
                hin = (r.get("Hindi") or "").strip()
                tri = (r.get(lang) or "").strip()
                if eng:
                    d0_rows.append({"text": eng, "language": "English"})
                    stats["D0_English"] += 1
                if hin:
                    d0_rows.append({"text": hin, "language": "Hindi"})
                    stats["D0_Hindi"] += 1
                if tri:
                    d0_rows.append({"text": tri, "language": lang})
                    stats[f"D0_{lang}"] += 1

        kept = 0
        for r in d1_chunk:
            eng = (r.get("English") or "").strip()
            hin = (r.get("Hindi") or "").strip()
            tri = (r.get(lang) or "").strip()
            # Both sides are needed for a usable evaluation pair.
            if eng and tri:
                d1_rows.append({"language": lang, "english": eng,
                                "hindi": hin, "tribal": tri})
                kept += 1
        stats[f"D1_{lang}"] = kept
        dropped = len(d1_chunk) - kept
        if dropped:
            print(f"   dropped from D1   : {dropped:,} (empty english or tribal)")

        per_lang[lang] = {
            "raw_rows": len(rows),
            "d0_rows": len(d0_chunk),
            "d1_rows_raw": len(d1_chunk),
            "d1_rows_kept": kept,
            "source_file": path.name,
        }

    # ---- write D1 ----
    print(f"\n💾 writing {d1_path.name}")
    tmp = d1_path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["language", "english", "hindi", "tribal"])
        w.writeheader()
        w.writerows(d1_rows)
    tmp.replace(d1_path)
    print(f"   {len(d1_rows):,} rows")

    # ---- write D0 ----
    if not args.skip_d0:
        print(f"\n🔀 shuffling D0 into a mixed-language corpus")
        random.shuffle(d0_rows)
        print(f"💾 writing {d0_path.name}")
        tmp = d0_path.with_suffix(".csv.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text", "language"])
            w.writeheader()
            w.writerows(d0_rows)
        tmp.replace(d0_path)
        print(f"   {len(d0_rows):,} rows")
    else:
        print("\n⏭️  D0 skipped (--skip_d0)")

    # ---- stats ----
    payload = {
        "seed": args.seed,
        "d1_ratio": args.d1_ratio,
        "languages": args.languages,
        "raw_dir": str(raw_dir),
        "totals": {"D0_rows": len(d0_rows), "D1_rows": len(d1_rows)},
        "per_language": per_lang,
        "counts": dict(stats),
        "note": ("D0 contains the English and Hindi sides as well as the tribal "
                 "side; continued pretraining therefore sees all three."),
    }
    STATS = out_dir / "dataset_stats.json"
    STATS.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ---- summary ----
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)
    print(f"{'Language':<10} {'raw':>9} {'D0':>9} {'D1':>9}")
    print("-" * 40)
    for lang in args.languages:
        s = per_lang[lang]
        print(f"{lang:<10} {s['raw_rows']:>9,} {s['d0_rows']:>9,} "
              f"{s['d1_rows_kept']:>9,}")
    print("-" * 40)
    print(f"{'TOTAL':<10} {sum(s['raw_rows'] for s in per_lang.values()):>9,} "
          f"{len(d0_rows):>9,} {len(d1_rows):>9,}")

    print(f"\n   D1     : {d1_path}")
    if not args.skip_d0:
        print(f"   D0     : {d0_path}")
    print(f"   stats  : {STATS}")
    print("\n   Nothing to configure — the other scripts find these by default.")
    print("   Next: python check_env.py")


if __name__ == "__main__":
    main()
