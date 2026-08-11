#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/home/zlqed/two_qubit_gate/zeropi/two_qubit_gate_clean"
cd "$PROJECT_DIR"
source /home/zlqed/anaconda3/etc/profile.d/conda.sh
conda activate zp2q
export MPLCONFIGDIR=/tmp/mpl
export NUMEXPR_MAX_THREADS=128
export PYTHONUNBUFFERED=1
RUN_LOG="$PROJECT_DIR/cz/data/fidelity_qutip/cz_n200_leakage_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$RUN_LOG")"
echo "$RUN_LOG" > "$PROJECT_DIR/cz/data/fidelity_qutip/cz_n200_leakage_current.logpath"
python cz/scripts/run_2q_fidelity.py \
  --gate cz \
  --n-truc-list 200 \
  --calculate-ideal true \
  --calculate-noise false \
  --use-qt-fidelity false \
  --pulse-file ../figure/data/data_cz_fidelity_leakage.txt \
  --tg-parallel true \
  --parallel-jobs 10 2>&1 | tee "$RUN_LOG"
python - "$RUN_LOG" "$PROJECT_DIR/figure/data/data_cz_fidelity_leakage.txt" <<'PY'
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
log_path, table_path = map(Path, sys.argv[1:])
text = log_path.read_text()
blocks = re.findall(r"f_ideal_200\s*=\s*np\.array\(\[\s*(.*?)\s*\]\)", text, re.S)
values = []
for block in blocks:
    nums = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", block)
    values.extend(float(x) for x in nums)
table = pd.read_csv(table_path)
if len(values) != len(table):
    raise RuntimeError(f"Expected {len(table)} n=200 results, found {len(values)} in {log_path}")
f = np.asarray(values)
table["f_leakage_200"] = f
table["fidelity_leakage_200"] = 1.0 - 10.0**f
tmp = table_path.with_suffix(table_path.suffix + ".tmp")
table.to_csv(tmp, index=False, float_format="%.8f")
tmp.replace(table_path)
print(f"Appended f_leakage_200 and fidelity_leakage_200 to {table_path}")
PY
echo "DONE $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
