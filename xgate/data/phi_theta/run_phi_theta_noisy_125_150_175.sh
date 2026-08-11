#!/usr/bin/env bash
set -u
PROJECT=/home/zlqed/two_qubit_gate/zeropi/two_qubit_gate_clean
OUT="$PROJECT/xgate/data/phi_theta"
cd "$PROJECT"
source /home/zlqed/anaconda3/etc/profile.d/conda.sh
conda activate zp2q
export MPLCONFIGDIR=/tmp/mpl
export NUMEXPR_MAX_THREADS=128
export PYTHONUNBUFFERED=1
pids=()
for n in 125 150 175; do
  nohup python xgate/scripts/run_xgate_fidelity.py \
    --drive-theta true --drive-phi true \
    --n-full "$n" --tg-list 0 --t1 3 \
    --use-qt-fidelity false \
    --calculate-ideal false --calculate-noise true \
    --parallel-jobs 1 \
    > "$OUT/phi_theta_noise_3us_n${n}.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$OUT/phi_theta_noise_3us_n${n}.pid"
  echo "n=$n pid=$pid"
  pids+=("$pid")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
echo "all jobs exited; status=$status"
exit "$status"
