#!/usr/bin/env bash
# Run from a regular server shell (outside the Codex tool sandbox).
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/zlqed/anaconda3/envs/zp2q/bin/python
OUT="$PROJECT_DIR/cz/data/separate_coherence_cz_182ns_3to300_aux3"
mkdir -p "$OUT"
cd "$PROJECT_DIR"
nohup "$PYTHON" -u cz/scripts/run_2q_fidelity_separate_coherence.py \
  </dev/null >>"$OUT/run.log" 2>&1 &
pid=$!
printf 'Started PID %s\nLog: %s/run.log\n' "$pid" "$OUT"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'Process exited during startup; check the log:'
  tail -25 "$OUT/run.log"
  exit 1
fi
printf 'Monitor: tail -f "%s/run.log"\n' "$OUT"
