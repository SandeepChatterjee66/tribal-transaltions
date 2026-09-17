#!/bin/bash
# ============================================================
# watch.sh — see what is happening, live.
#
#   bash watch.sh                 tail the newest log (follows a running job)
#   bash watch.sh --list          every invocation ever, newest first
#   bash watch.sh --stats         live training progress: loss, ETA, throughput
#   bash watch.sh --errors        every run that crashed, with its traceback
#   bash watch.sh nllb-1.3b       tail the newest log matching a name
#   bash watch.sh --files         where all the logs are
#
# Ctrl-C stops watching. It does NOT stop the job.
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOGS="logs"
# Find an interpreter: honour $PY, else python, else python3.
if [ -z "${PY:-}" ]; then
    if command -v python >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "ERROR: no python or python3 on PATH" >&2; exit 1; fi
fi

newest_log() {
    # Prefer the symlink written by setup_logging; fall back to mtime order.
    if [ -L "$LOGS/latest.log" ] && [ -e "$LOGS/latest.log" ]; then
        readlink "$LOGS/latest.log" | sed "s|^|$LOGS/|"
    elif [ -f "$LOGS/latest.txt" ]; then
        echo "$LOGS/$(cat "$LOGS/latest.txt")"
    else
        ls -t "$LOGS"/*.log 2>/dev/null | grep -v 'latest.log' | head -1
    fi
}

case "${1:-}" in

--list|-l)
    if [ ! -f "$LOGS/index.csv" ]; then
        echo "No runs recorded yet ($LOGS/index.csv does not exist)."
        exit 0
    fi
    echo "Every invocation, newest first:"
    echo
    $PY - <<'EOF'
import csv, sys
rows = list(csv.DictReader(open("logs/index.csv")))
starts = {}
for r in rows:
    if r["event"] == "started":
        starts[r["log_file"]] = r
    else:
        s = starts.get(r["log_file"], {})
        s["status"] = r["status"]
        s["duration_min"] = r["duration_min"]
print(f"{'WHEN':<20} {'SCRIPT':<11} {'WHAT':<28} {'MIN':>6}  {'STATUS':<11} LOG")
print("-" * 110)
for r in reversed(list(starts.values())):
    print(f"{r['timestamp']:<20} {r['script']:<11} {r['label'][:28]:<28} "
          f"{(r.get('duration_min') or '-'):>6}  "
          f"{(r.get('status') or 'running'):<11} {r['log_file']}")
EOF
    ;;

--stats|-s)
    LOG_CSV=$(ls -t runs/*/*/training_log.csv 2>/dev/null | head -1)
    if [ -z "$LOG_CSV" ]; then
        echo "No training_log.csv yet — nothing has started training."
        exit 0
    fi
    echo "Following: $LOG_CSV   (Ctrl-C to stop)"
    echo
    while true; do
        $PY - "$LOG_CSV" <<'EOF'
import sys
import pandas as pd
try:
    df = pd.read_csv(sys.argv[1])
except Exception as e:
    print("waiting for data...", e); raise SystemExit
tr = df[pd.to_numeric(df["loss"], errors="coerce").notna()].copy()
ev = df[pd.to_numeric(df["eval_loss"], errors="coerce").notna()].copy()
if tr.empty:
    print("waiting for the first logged step..."); raise SystemExit
tr["loss"] = pd.to_numeric(tr["loss"])
last = tr.iloc[-1]
print(f"\033[2J\033[H", end="")          # clear screen
print(f"file          {sys.argv[1]}")
print(f"step          {last['step']} / epoch {last['epoch']}")
print(f"loss          {last['loss']:.4f}   (start {tr['loss'].iloc[0]:.4f}, "
      f"min {tr['loss'].min():.4f})")
if not ev.empty:
    ev["eval_loss"] = pd.to_numeric(ev["eval_loss"])
    print(f"eval_loss     {ev['eval_loss'].iloc[-1]:.4f}   "
          f"(best {ev['eval_loss'].min():.4f} @ step "
          f"{int(ev.loc[ev['eval_loss'].idxmin(), 'step'])})")
print(f"throughput    {last['samples_per_s']} samp/s   "
      f"{last['tokens_per_s']} tok/s")
print(f"gpu memory    {last['gpu_mem_gb']} GB")
print(f"elapsed       {last['elapsed_min']} min")
print(f"ETA           {last['eta_min']} min")
print()
print("last 15 logged steps:")
cols = [c for c in ("step", "epoch", "loss", "eval_loss", "lr",
                    "samples_per_s", "eta_min") if c in tr.columns]
print(df[cols].tail(15).to_string(index=False))
EOF
        sleep 10
    done
    ;;

--errors|-e)
    FOUND=0
    for f in runs/*/*/error.log; do
        [ -f "$f" ] || continue
        FOUND=1
        echo "======================================================================"
        echo " CRASHED: $(dirname "$f")"
        echo "======================================================================"
        tail -20 "$f"
        echo
    done
    [ "$FOUND" = "0" ] && echo "No error.log anywhere — nothing has crashed."
    ;;

--files|-f)
    echo "Archived logs (one file per invocation, never overwritten):"
    ls -lht "$LOGS"/*.log 2>/dev/null | head -25 || echo "  none yet"
    echo
    echo "Index of every invocation:"
    echo "  $LOGS/index.csv"
    echo
    echo "Per-run logs (appended across invocations):"
    ls -lht runs/*/*/{infer,finetune}.log 2>/dev/null | head -20 || echo "  none yet"
    echo
    echo "Live loss curves:"
    ls -lht runs/*/*/training_log.csv 2>/dev/null | head -10 || echo "  none yet"
    ;;

--help|-h)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    ;;

"")
    F=$(newest_log)
    if [ -z "$F" ] || [ ! -f "$F" ]; then
        echo "No logs yet. Start something first, e.g.:"
        echo "    bash run_all.sh"
        exit 0
    fi
    echo "Following: $F   (Ctrl-C stops watching, not the job)"
    echo "----------------------------------------------------------------------"
    tail -n 40 -f "$F"
    ;;

*)
    # Treat the argument as a filename fragment: model name, setting, etc.
    F=$(ls -t "$LOGS"/*"$1"*.log 2>/dev/null | head -1)
    if [ -z "$F" ]; then
        echo "No archived log matching '$1'. Available:"
        ls -t "$LOGS"/*.log 2>/dev/null | head -20 || echo "  none yet"
        exit 1
    fi
    echo "Following: $F   (Ctrl-C stops watching, not the job)"
    echo "----------------------------------------------------------------------"
    tail -n 40 -f "$F"
    ;;
esac
