#!/bin/bash
# ============================================================
# run_all.sh — THE ONE COMMAND.
#
# Runs the whole thing in the right order: check -> zero-shot -> LoRA
# fine-tune -> score -> consolidate. Stops at the first real failure and
# tells you which log to read.
#
#   bash run_all.sh          run in the foreground
#   bash run_all.sh --bg     detach and auto-tail (survives a dropped ssh;
#                            Ctrl-C stops watching, not the job)
#
# Everything is resume-safe: if it dies, re-run the same command.
#
# Environment knobs (all optional):
#   TRIBAL_DATA=/path/D1_parallel.csv   where the corpus is
#   MODELS="nllb-600m nllb-1.3b"        which models
#   SKIP_SELFTEST=1                     skip the offline self-test
#   SKIP_FINETUNE=1                     zero-shot only
#   SWEEP=1                             also sweep source-language tokens
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS="${MODELS:-nllb-600m nllb-1.3b}"
# Find an interpreter: honour $PY, else python, else python3.
if [ -z "${PY:-}" ]; then
    if command -v python >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "ERROR: no python or python3 on PATH" >&2; exit 1; fi
fi
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

# ---- --bg: detach, then tail so you watch without holding the job ----
# Survives a dropped SSH connection. Ctrl-C stops the tail, not the run.
if [ "${1:-}" = "--bg" ]; then
    shift
    STAMP=$(date +%Y%m%d_%H%M%S)
    MASTER="$LOG_DIR/run_all_${STAMP}.log"
    echo "$MASTER" > "$LOG_DIR/run_all_latest.txt"
    echo "Launching in the background."
    echo "  master log : $MASTER"
    echo "  pid file   : $LOG_DIR/run_all.pid"
    nohup env MODELS="$MODELS" PY="$PY" LOG_DIR="$LOG_DIR" \
        bash "$0" "$@" > "$MASTER" 2>&1 &
    echo $! > "$LOG_DIR/run_all.pid"
    echo "  pid        : $(cat "$LOG_DIR/run_all.pid")"
    echo ""
    echo "Now following the log. Ctrl-C stops WATCHING; the job keeps going."
    echo "Re-attach any time with:  bash watch.sh"
    echo "Stop the job with:        kill \$(cat $LOG_DIR/run_all.pid)"
    echo "----------------------------------------------------------------------"
    sleep 2
    tail -f "$MASTER"
    exit 0
fi

DATA_FLAG=""
[ -n "${TRIBAL_DATA:-}" ] && DATA_FLAG="--data $TRIBAL_DATA"

STARTED=$(date +%s)

banner() {
    echo ""
    echo "======================================================================"
    echo " $1"
    echo "======================================================================"
}

die() {
    echo ""
    echo "!! FAILED at: $1"
    echo "!! Read: $2"
    echo "!! Then re-run 'bash run_all.sh' — completed work is skipped."
    exit 1
}

# ---- Generate D1 from raw_data/ if it is not there yet ----
# Deliberately automatic: the only thing a user should have to do is drop the
# corpus CSVs into raw_data/. Nothing to configure, no paths to edit.
if [ ! -f data/processed/D1_parallel.csv ] && [ -z "${TRIBAL_DATA:-}" ]; then
    banner "STEP 0/6  preparing data from raw_data/"
    $PY prepare_data.py 2>&1 | tee "$LOG_DIR/00_prepare_data.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        die "data preparation" "$LOG_DIR/00_prepare_data.log"
    fi
else
    banner "STEP 0/6  data already prepared — skipping"
fi

banner "STEP 1/6  environment check"
$PY check_env.py $DATA_FLAG 2>&1 | tee "$LOG_DIR/00_check_env.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    die "environment check" "$LOG_DIR/00_check_env.log"
fi

if [ "${SKIP_SELFTEST:-0}" != "1" ]; then
    banner "STEP 2/6  offline self-test (no GPU, no downloads)"
    echo "Proves the pipeline works before we spend real GPU time."
    $PY selftest.py $DATA_FLAG 2>&1 | tee "$LOG_DIR/01_selftest.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        die "self-test" "$LOG_DIR/01_selftest.log  (and .selftest/logs/)"
    fi
else
    banner "STEP 2/6  offline self-test — SKIPPED (SKIP_SELFTEST=1)"
fi

banner "STEP 3/6  zero-shot inference"
for MODEL in $MODELS; do
    echo ""
    echo ">>> zero-shot: $MODEL"
    $PY infer.py --model "$MODEL" $DATA_FLAG \
        2>&1 | tee "$LOG_DIR/02_zeroshot_${MODEL}.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && \
        die "zero-shot $MODEL" "runs/$MODEL/zeroshot/error.log"

    if [ "${SWEEP:-0}" = "1" ]; then
        echo ">>> source-token sweep (dev split): $MODEL"
        $PY infer.py --model "$MODEL" $DATA_FLAG --sweep_src \
            --split dev --limit 300 --tag sweep \
            2>&1 | tee "$LOG_DIR/02_sweep_${MODEL}.log"
    fi
done

if [ "${SKIP_FINETUNE:-0}" != "1" ]; then
    banner "STEP 4/6  LoRA fine-tuning + inference"
    for MODEL in $MODELS; do
        echo ""
        echo ">>> fine-tune (LoRA): $MODEL"
        $PY finetune.py --model "$MODEL" $DATA_FLAG \
            --set finetune.mode=lora \
            2>&1 | tee "$LOG_DIR/03_finetune_${MODEL}.log"
        [ "${PIPESTATUS[0]}" -ne 0 ] && \
            die "fine-tune $MODEL" "runs/$MODEL/finetune/error.log"

        echo ">>> inference with fine-tuned $MODEL"
        $PY infer.py --model "$MODEL" --ckpt "runs/$MODEL/finetune/best" \
            --tag ft_lora $DATA_FLAG \
            2>&1 | tee "$LOG_DIR/03_infer_ft_${MODEL}.log"
        [ "${PIPESTATUS[0]}" -ne 0 ] && \
            die "fine-tuned inference $MODEL" "runs/$MODEL/ft_lora/error.log"
    done
else
    banner "STEP 4/6  fine-tuning — SKIPPED (SKIP_FINETUNE=1)"
fi

banner "STEP 5/6  scoring"
$PY evaluate.py --all --name mt_all 2>&1 | tee "$LOG_DIR/04_evaluate.log"
[ "${PIPESTATUS[0]}" -ne 0 ] && die "scoring" "$LOG_DIR/04_evaluate.log"

banner "STEP 6/6  consolidating into CSVs"
$PY consolidate.py 2>&1 | tee "$LOG_DIR/05_consolidate.log"
[ "${PIPESTATUS[0]}" -ne 0 ] && die "consolidation" "$LOG_DIR/05_consolidate.log"

ELAPSED=$(( ($(date +%s) - STARTED) / 60 ))

banner "DONE in ${ELAPSED} min"
cat <<'EOF'
Results — every table is written as BOTH .tsv and .csv:
  consolidated/paper_table.tsv     <- the table for the paper
  consolidated/zeroshot_vs_ft.tsv  <- zero-shot vs fine-tuned, with deltas
  consolidated/all_scores.tsv      <- every number, with bootstrap CIs
  consolidated/training_stats.tsv  <- loss, convergence, wall-clock, throughput
  consolidated/inference_stats.tsv <- per-language decode cost
  consolidated/run_manifest.tsv    <- what ran where, and anything that failed
  consolidated/paper_table.tex     <- LaTeX version (secondary)
  consolidated/README_RESULTS.md   <- what each file contains

Logs (kept forever, one file per invocation):
  logs/index.csv                   every run, with status and duration
  logs/<timestamp>_<script>_*.log  the log of that specific invocation
  bash watch.sh --list             browse them
  bash watch.sh <model>            re-read a specific one

Before reporting anything, check the empty_pct column in all_scores.csv.
Above ~2% means generation failures are dragging the score down; re-run those
items first (infer.py only redoes what is missing).

Optional follow-ups, once LoRA numbers look right:
  MODE=full LR=3e-5 bash run_finetune.sh          # full fine-tune
  FREEZE_ENCODER=1 MODE=full bash run_finetune.sh # decoder-only
  SWEEP=1 bash run_all.sh                         # justify the proxy tokens
EOF
