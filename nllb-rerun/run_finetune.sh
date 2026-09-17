#!/bin/bash
# ============================================================
# run_finetune.sh
#
# Fine-tune each model on the D1 train split, run inference with the
# resulting checkpoint, then build the combined zero-shot vs fine-tuned table.
#
# USAGE:
#   bash run_finetune.sh
#   bash run_finetune.sh --smoke                    # ~5 min, proves it runs
#   MODELS="nllb-600m" bash run_finetune.sh
#   MODE=full LR=3e-5 bash run_finetune.sh          # full FT instead of LoRA
#   PER_LANGUAGE=1 bash run_finetune.sh             # one model per language
#
# Resume-safe: re-running resumes from the last checkpoint.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- CONFIG (override from the environment) ----
MODELS="${MODELS:-nllb-600m nllb-1.3b}"
DATA="${DATA:-}"
MODE="${MODE:-lora}"
LR="${LR:-}"
PER_LANGUAGE="${PER_LANGUAGE:-0}"
LANGUAGES="${LANGUAGES:-Bhili Gondi Mundari Santali}"
LOG_DIR="${LOG_DIR:-logs}"

# Find an interpreter: honour $PY, else python, else python3.
if [ -z "${PY:-}" ]; then
    if command -v python >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "ERROR: no python or python3 on PATH" >&2; exit 1; fi
fi


SMOKE=""
if [ "${1:-}" = "--smoke" ]; then
    SMOKE="--smoke"
    echo "!! SMOKE MODE — tiny run, DO NOT REPORT"
fi

DATA_FLAG=""
[ -n "$DATA" ] && DATA_FLAG="--data $DATA"

FREEZE_ENCODER="${FREEZE_ENCODER:-0}"

SETS="finetune.mode=$MODE"
[ -n "$LR" ] && SETS="$SETS finetune.lr=$LR"

PL_FLAG=""
SETTING="finetune"
if [ "$PER_LANGUAGE" = "1" ]; then
    PL_FLAG="--per_language"
    SETTING="finetune_per_lang"
fi

# Distinct tag per variant so LoRA / full / frozen-encoder results land in
# separate run dirs and end up as separate rows in the consolidated table.
TAG_FLAG=""
INFER_TAG="ft_lora"
if [ "$FREEZE_ENCODER" = "1" ]; then
    SETS="$SETS finetune.freeze_encoder=true"
    SETTING="finetune_frozen"
    TAG_FLAG="--tag finetune_frozen"
    INFER_TAG="ft_frozen"
elif [ "$MODE" = "full" ]; then
    SETTING="finetune_full"
    TAG_FLAG="--tag finetune_full"
    INFER_TAG="ft_full"
fi

mkdir -p "$LOG_DIR"

echo ""
echo "======================================================"
echo " TRIBALSUITE — SEQ2SEQ FINE-TUNING"
echo "======================================================"
echo " Models       : $MODELS"
echo " Mode         : $MODE"
echo " Per-language : $PER_LANGUAGE"
echo " Freeze enc.  : $FREEZE_ENCODER"
echo " Run dir      : runs/<model>/$SETTING"
echo " Data         : ${DATA:-<auto-discover>}"
echo "======================================================"
echo ""

for MODEL in $MODELS; do
    echo ""
    echo "======================================================"
    echo " 🔥 fine-tuning: $MODEL"
    echo "======================================================"
    "$PY" finetune.py --model "$MODEL" $DATA_FLAG $PL_FLAG $SMOKE $TAG_FLAG \
        --set $SETS \
        2>&1 | tee -a "$LOG_DIR/finetune_${MODEL}.log"

    echo ""
    echo " 🌍 inference with fine-tuned $MODEL"
    if [ "$PER_LANGUAGE" = "1" ]; then
        for LANG in $LANGUAGES; do
            CKPT="runs/$MODEL/$SETTING/$LANG/best"
            if [ ! -d "$CKPT" ]; then
                echo " ⚠️  missing checkpoint $CKPT — skipping $LANG"
                continue
            fi
            "$PY" infer.py --model "$MODEL" --ckpt "$CKPT" \
                --languages "$LANG" --tag "$INFER_TAG" $DATA_FLAG \
                2>&1 | tee -a "$LOG_DIR/infer_ft_${MODEL}_${LANG}.log"
        done
    else
        "$PY" infer.py --model "$MODEL" \
            --ckpt "runs/$MODEL/$SETTING/best" --tag "$INFER_TAG" $DATA_FLAG \
            2>&1 | tee -a "$LOG_DIR/infer_ft_${MODEL}.log"
    fi
    echo "✅ finished: $MODEL"
done

echo ""
echo "======================================================"
echo " 📊 scoring (zero-shot + fine-tuned together)"
echo "======================================================"
"$PY" evaluate.py --all --name mt_all 2>&1 | tee -a "$LOG_DIR/evaluate.log"

echo ""
echo "======================================================"
echo " 🎉 FINE-TUNING COMPLETE"
echo "======================================================"
echo " Tables : tables/mt_all.{tsv,csv,tex}"
echo " Ckpts  : runs/<model>/$SETTING/best"
echo ""
echo " TensorBoard:"
echo "   tensorboard --logdir runs"
echo "======================================================"
