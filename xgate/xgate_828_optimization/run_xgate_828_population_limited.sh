#!/usr/bin/env bash
# Start one detached, population-limited 828 ns X-gate optimization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$"
OUT="${XGATE_828_OUT:-$ROOT/xgate_828_optimization/population_limited_$RUN_ID}"
PYTHON="${XGATE_PYTHON:-/home/zlqed/anaconda3/envs/zp2q/bin/python}"

mkdir -p "$OUT/.matplotlib"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/.matplotlib}"

cd "$ROOT"
nohup setsid "$PYTHON" -u Xgate_fidelity_optimize.py \
  --task population-limited --output-dir "$OUT" "$@" \
  >"$OUT/population-limited.log" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$OUT/population-limited.pid"

echo "Started population-limited optimization: PID $PID"
echo "Output: $OUT"
echo "Log: $OUT/population-limited.log"
