#!/bin/bash
# ============================================================
# run_zeroshot.sh
#
# Zero-shot tribal -> English inference for every model in MODELS,
# then one combined table.
#
# USAGE:
#   bash run_zeroshot.sh
#   bash run_zeroshot.sh --smoke              # 50 items/lang, greedy, fast
#   MODELS="nllb-600m nllb-1.3b" bash run_zeroshot.sh
#   DATA=/scratch/tribal/D1_parallel.csv bash run_zeroshot.sh
#
# Resume-safe: re-running skips work already on disk.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- CONFIG (override from the environment) ----
MODELS="${MODELS:-nllb-600m nllb-1.3b mbart50}"
DATA="${DATA:-}"
LOG_DIR="${LOG_DIR:-logs}"

# Find an interpreter: honour $PY, else python, else python3.
if [ -z "${PY:-}" ]; then
    if command -v python >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "ERROR: no python or python3 on PATH" >&2; exit 1; fi
fi


EXTRA=""
if [ "${1:-}" = "--smoke" ]; then
    EXTRA="--limit 50 --num_beams 1"
    echo "!! SMOKE MODE — 50 items/language, greedy decoding, DO NOT REPORT"
fi

DATA_FLAG=""
[ -n "$DATA" ] && DATA_FLAG="--data $DATA"

mkdir -p "$LOG_DIR"

echo ""
echo "======================================================"
echo " TRIBALSUITE — ZERO-SHOT MT (open-weight seq2seq)"
echo "======================================================"
echo " Models : $MODELS"
echo " Data   : ${DATA:-<auto-discover>}"
echo " Extra  : ${EXTRA:-none}"
echo "======================================================"
echo ""

# ---- PREFLIGHT: fail fast before any model download ----
for MODEL in $MODELS; do
    echo ">> preflight: $MODEL"
    "$PY" infer.py --model "$MODEL" $DATA_FLAG --dry_run
done
echo "✅ preflight passed for all models"
echo ""

# ---- INFERENCE ----
for MODEL in $MODELS; do
    echo ""
    echo "======================================================"
    echo " 🌍 zero-shot: $MODEL"
    echo "======================================================"
    "$PY" infer.py --model "$MODEL" $DATA_FLAG $EXTRA \
        2>&1 | tee -a "$LOG_DIR/zeroshot_${MODEL}.log"
    echo "✅ finished: $MODEL"
done

# ---- SCORE ----
echo ""
echo "======================================================"
echo " 📊 scoring"
echo "======================================================"
"$PY" evaluate.py --all --name zeroshot_all 2>&1 | tee -a "$LOG_DIR/evaluate.log"

echo ""
echo "======================================================"
echo " 🎉 ZERO-SHOT COMPLETE"
echo "======================================================"
echo " Tables : tables/zeroshot_all.{tsv,csv,tex}"
echo " Preds  : runs/<model>/zeroshot/predictions.csv"
echo "======================================================"
