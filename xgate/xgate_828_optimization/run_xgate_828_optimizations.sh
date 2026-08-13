#!/usr/bin/env bash
# Run both long optimizations sequentially.  Launch this script (not Codex)
# using: ./run_xgate_828_optimizations.sh start
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
BASE_OUT="$ROOT/xgate_828_optimization"
LATEST_FILE="$BASE_OUT/latest_run"
COMMAND="${1:-start}"

# An explicitly supplied directory is used as-is.  Otherwise every start gets
# a timestamped directory; status/stop resolve to the most recently started run.
if [[ -n "${XGATE_828_OUT:-}" ]]; then
  OUT="$XGATE_828_OUT"
elif [[ "$COMMAND" == "start" ]]; then
  OUT="$BASE_OUT/$(date '+%Y%m%d_%H%M%S')_$$"
else
  if [[ ! -f "$LATEST_FILE" ]]; then
    echo "No previous run found in $BASE_OUT" >&2
    exit 1
  fi
  OUT="$(cat "$LATEST_FILE")"
fi
PYTHON="${XGATE_PYTHON:-/home/zlqed/anaconda3/envs/zp2q/bin/python}"
PIDFILE="$OUT/runner.pid"
LOG="$OUT/runner.log"
mkdir -p "$OUT"

run_jobs() {
  cd "$ROOT"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/.matplotlib}"
  mkdir -p "$MPLCONFIGDIR"
  echo "runner $$ started $(date --iso-8601=seconds)"
  "$PYTHON" -u Xgate_fidelity_optimize.py --task direct --output-dir "$OUT" "$@"
  "$PYTHON" -u Xgate_fidelity_optimize.py --task population-limited --output-dir "$OUT" "$@"
  echo "runner $$ finished $(date --iso-8601=seconds)"
}

case "${1:-start}" in
  start)
    shift || true
    mkdir -p "$BASE_OUT"
    printf '%s\n' "$OUT" > "$LATEST_FILE"
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Already running as PID $(cat "$PIDFILE")"
      exit 1
    fi
    # setsid + nohup detaches the runner from the invoking terminal/session.
    XGATE_828_OUT="$OUT" nohup setsid bash "$SELF" foreground "$@" >>"$LOG" 2>&1 </dev/null &
    echo $! >"$PIDFILE"
    echo "Started PID $!"
    echo "Output: $OUT"
    echo "Log: $LOG"
    ;;
  foreground)
    shift
    trap 'rm -f "$PIDFILE"' EXIT
    run_jobs "$@"
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "RUNNING PID $(cat "$PIDFILE")"
    else
      echo "NOT RUNNING"
    fi
    for f in "$OUT"/*.status; do [[ -e "$f" ]] && echo "$(basename "$f"): $(cat "$f")"; done
    tail -n 30 "$LOG" 2>/dev/null || true
    ;;
  stop)
    [[ -f "$PIDFILE" ]] && kill -- "-$(cat "$PIDFILE")" 2>/dev/null || true
    ;;
  *) echo "Usage: $0 {start|status|stop} [optimizer arguments]"; exit 2 ;;
esac
