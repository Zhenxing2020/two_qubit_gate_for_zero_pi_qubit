#!/usr/bin/env bash
# Resume the leakage-inclusive CNOT optimization and, once all 33 points are
# present, run the three noisy data sets and build the figure-data table.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPT_DIR="$ROOT/cnot/data/fidelity_optimize"
NOISE_DIR="$ROOT/cnot/data/fidelity_qutip"
CSV="$OPT_DIR/cnot_leakage_optimized_all33.csv"
NPZ="$OPT_DIR/cnot_leakage_optimized_all33.npz"
OPT_LOG="$OPT_DIR/cnot_leakage_optimized_all33.log"
PULSES="$OPT_DIR/cnot_leakage_optimized_pulses.csv"
FINAL="$ROOT/figure/data/data_cnot_fidelity_leakage_optimized.txt"
LOCK="$OPT_DIR/cnot_leakage_pipeline.lock"

mkdir -p "$OPT_DIR" "$NOISE_DIR"
exec 9>"$LOCK"
flock -n 9 || { echo "Another CNOT pipeline already holds $LOCK"; exit 0; }
echo $$ > "$OPT_DIR/cnot_leakage_pipeline.pid"
trap 'rm -f "$OPT_DIR/cnot_leakage_pipeline.pid"' EXIT

source /home/zlqed/anaconda3/etc/profile.d/conda.sh
conda activate zp2q
export MPLCONFIGDIR=/tmp/mpl
export NUMEXPR_MAX_THREADS=128
mkdir -p "$MPLCONFIGDIR"

row_count() {
  python - "$CSV" <<'PY'
import csv, pathlib, sys
p = pathlib.Path(sys.argv[1])
print(sum(1 for _ in csv.DictReader(p.open())) if p.exists() else 0)
PY
}

rows="$(row_count)"
echo "[$(date -Is)] optimized rows before resume: $rows/33"
if (( rows < 33 )); then
  cd "$ROOT/cnot"
  python -u scripts/optimize_cnot_fidelity.py \
    --pulse-file ../figure/data/data_cnot_fidelity_npz.txt \
    --workers 60 --truc-optimize 220 --truc-large 1000 \
    --resume --result-file "$NPZ" --csv-file "$CSV" \
    >> "$OPT_LOG" 2>&1
fi

rows="$(row_count)"
if (( rows != 33 )); then
  echo "[$(date -Is)] ERROR: optimization ended with $rows/33 rows" >&2
  exit 1
fi

python - "$CSV" "$PULSES" <<'PY'
import pandas as pd, sys
src, dst = sys.argv[1:]
d = pd.read_csv(src)
if len(d) != 33 or list(d.idx.astype(int)) != list(range(33)):
    raise SystemExit("optimization CSV is not a complete ordered 33-row table")
d.rename(columns={'p0':'tg','p1':'drive_amp_1','p2':'drive_amp_2',
                  'p3':'detune_1','p4':'detune_2'})[
    ['tg','drive_amp_1','drive_amp_2','detune_1','detune_2']
].to_csv(dst, index=False)
PY

for t1 in 170 30 3; do
  log="$NOISE_DIR/cnot_noise_leakage_optimized_n220_T1=${t1}us.log"
  # A completed log has one final 33-element fidelity_noise_list. Re-run an
  # incomplete log from scratch so the finalizer cannot consume partial data.
  if ! python - "$log" <<'PY'
import pathlib, re, sys
p=pathlib.Path(sys.argv[1])
if not p.exists(): raise SystemExit(1)
s=p.read_text(errors='replace')
b=re.findall(r'fidelity_noise_list\s*=\s*np\.array\(\[(.*?)\]\)',s,re.S)
if not b or len(re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?',b[-1])) != 33:
    raise SystemExit(1)
PY
  then
    : > "$log"
    cd "$ROOT/cz"
    # Noisy n=220 superoperators are memory intensive. Run gate times
    # sequentially; eight concurrent gate times previously exhausted RAM.
    python -u scripts/run_2q_fidelity.py \
      --gate cnot --n-truc-list 220 \
      --calculate-ideal false --calculate-noise true \
      --use-qt-fidelity false --t1 "$t1" \
      --pulse-file "$PULSES" --parallel-jobs 1 --tg-parallel false \
      --num-cpus-noisy 1 \
      > "$log" 2>&1
  fi
done

cd "$ROOT"
python cnot/scripts/finalize_cnot_leakage_data.py \
  --optimization-csv "$CSV" \
  --noise-170 "$NOISE_DIR/cnot_noise_leakage_optimized_n220_T1=170us.log" \
  --noise-30 "$NOISE_DIR/cnot_noise_leakage_optimized_n220_T1=30us.log" \
  --noise-3 "$NOISE_DIR/cnot_noise_leakage_optimized_n220_T1=3us.log" \
  --output "$FINAL"
echo "[$(date -Is)] CNOT pipeline complete: $FINAL"
