# Fidelity workflow and tmux rerun guide

This file summarizes the fidelity work performed in this repository and gives
commands for restarting the unfinished calculations from a normal SSH shell.

## Current status — 2026-08-03

### Completed optimization and reference calculations

```text
Phi leakage-inclusive optimization:             19/19
Theta corrected optimization, max_step=3e-4:    18/18
CZ leakage-inclusive optimization, n_truc=60:   30/30
CZ ideal leakage-inclusive, n_truc=60:          30/30
```

Earlier five-point Phi noisy calculations and the first 25 CZ noisy points are
also complete. Earlier Theta five-point noisy results used the obsolete
`max_step=1e-3` pulse optimization and should not be used as final data.

### Current missing-only noisy schedule

The initial attempt to run every point again was stopped to avoid duplicate
work. The active schedule calculates only missing results:

```text
Phi, T1=170 us: 14 missing indices
Phi, T1=30 us:  14 missing indices
Phi, T1=3 us:   all 19 indices (no earlier result)

Theta corrected, T1=170 us: all 18 indices
Theta corrected, T1=30 us:  all 18 indices
Theta corrected, T1=3 us:   all 18 indices

CZ, T1=170 us: indices [25,26,28,29]
CZ, T1=30 us:  indices [25,26,27,28,29]
CZ, T1=3 us:   indices [25,26,27,28,29]
```

Worker limits were adjusted as follows:

```text
Theta: 3 tasks x 40 workers = 120
CZ:    3 tasks x 16 workers = 48
Phi:   3 tasks x 20 workers = 60
Configured total = 228 workers (<240)
```

Phi uses the newly added option:

```bash
--parallel-jobs 5
```

With four QuTiP workers per outer job, this gives approximately 20 workers per
Phi task. Current status file:

```text
xgate/data/fidelity_qutip/worker_limited_228_20260803_021625.status
```

The active logs use timestamp `20260803_021625` and contain `workers20`,
`workers40`, or `workers16` in their filenames.

### Optimized pulse tables

Full optimized pulse tables used by the noisy jobs:

```text
xgate/data/fidelity_optimize/phi_optimized_all19.csv
xgate/data/fidelity_optimize/theta_optimized_corrected_all18.csv
cz/data/fidelity_optimize/cz_optimized_all30.csv
```

The corrected five-point Theta table is also available:

```text
xgate/data/fidelity_optimize/theta_optimized_five_points_corrected_mstep3e4.csv
```

### Exported leakage-inclusive fidelity tables

Optimized pulse parameters and ideal leakage-inclusive fidelity were exported
in the same comma-separated style as the existing `figure/data` tables:

```text
figure/data/data_xgate_phi_fidelity_leakage.txt
figure/data/data_xgate_theta_fidelity_leakage.txt
figure/data/data_cz_fidelity_leakage.txt
```

Columns are:

```text
Phi:
  tg, drive_amp_1, drive_amp_2, detune_1, detune_2,
  f_leakage_160, fidelity_leakage_160

Theta (corrected max_step=3e-4):
  tg, drive_amp_1, drive_amp_2, detune_1, detune_2,
  f_leakage_157, fidelity_leakage_157

CZ:
  tg, drive_amp, detune, f_leakage_60, fidelity_leakage_60
```

The convention is:

```python
f_leakage = log10(1 - fidelity)
fidelity_leakage = 1 - 10**f_leakage
```

After the missing-only noisy jobs finish, append T1=170, 30, and 3 us noisy
columns to these files.

### Important rerun warning

Do not start the older full-sweep or five-point commands later in this document
while the current `worker_limited_228` schedule is alive. Check first with:

```bash
cat xgate/data/fidelity_qutip/worker_limited_228_20260803_021625.status
ps -fu "$USER" | grep -E 'run_xgate_fidelity|run_2q_fidelity' | grep -v grep
```

The remaining sections are retained as workflow history and reproducibility
instructions.

## 1. Environment

```bash
cd /home/zlqed/two_qubit_gate/zeropi/two_qubit_gate_clean
source /home/zlqed/anaconda3/etc/profile.d/conda.sh
conda activate zp2q
export MPLCONFIGDIR=/tmp/mpl
export NUMEXPR_MAX_THREADS=128
```

The base Conda environment is currently incompatible with the installed
NumPy/QuTiP/Matplotlib binaries. Use the `zp2q` environment.

## 2. Persistent execution with tmux

Create a session:

```bash
tmux new-session -s fidelity
```

Useful commands:

```text
Ctrl-b c       create a new window
Ctrl-b n       next window
Ctrl-b p       previous window
Ctrl-b d       detach while keeping jobs running
```

Reconnect later with:

```bash
tmux attach-session -t fidelity
```

Check sessions with:

```bash
tmux ls
```

`nohup` is optional inside tmux, but the commands below use it so each command
also writes a standalone log. Do not launch duplicate commands while the old
Codex-managed copy is still running.

## 3. Fidelity modes added to the code

The relevant scripts now accept:

```bash
--use-qt-fidelity true
--use-qt-fidelity false
```

`false` selects:

```python
average_gate_fidelity_trace_decreasing()
```

This is the leakage-inclusive formula. It contains both the average survival
term and target-overlap term. For `cz/scripts/run_2q_fidelity.py`, leakage-
inclusive fidelity is now the documented default:

```python
# Trace-decreasing fidelity is the project default because it includes
# leakage/survival loss. Pass --use-qt-fidelity true for QuTiP's formula.
cfg["use_qt_fidelity"] = False
```

For clarity, the commands below still pass `--use-qt-fidelity false`
explicitly.

## 4. Important files and truncations

### Original pulse tables

```text
figure/data/data_xgate_phi_mstep_1e3_npz.txt
figure/data/data_xgate_theta_mstep_3e4_npz.txt
figure/data/data_cz_fidelity_npz_select.txt
```

Initial reference fidelity columns:

```text
Phi:   f_charge_160_npz, n_full=300 -> charge space n_truc=160
Theta: f_157_charge,     n_full=300 -> charge space n_truc=157
CZ:    f_charge_60,      charge-pick n_truc=60
```

### Optimized pulse data

```text
cz/data/fidelity_optimize/cz_reopt.txt
cz/data/fidelity_optimize/cz_optimized_all30.csv
xgate/data/fidelity_optimize/phi_optimized_five_points_20260801_093726.csv
xgate/data/fidelity_optimize/theta_optimized_five_points_20260801_093726.csv
```

The CZ text file contains 25 printed `[tg, drive_amp, detune]` rows. The
`cz_optimized_all30.csv` file combines those rows with the final five resumed
optimization results.

## 5. Completed ideal calculations and optimization

- CZ ideal leakage-inclusive, `n_truc=60`: completed `30/30`.
- Original Phi leakage-inclusive optimization: completed `19/19`.
- Original Theta leakage-inclusive optimization: completed `18/18`, but it
  used an inconsistent solver step and must not be treated as final.
- CZ leakage-inclusive optimization: completed `30/30`.

Relevant logs:

```text
cz/data/fidelity_qutip/cz_ideal_trace_decreasing_n60_20260801_091651.log
xgate/data/fidelity_optimize/phi_trace_decreasing_n300_20260731_051907.log
xgate/data/fidelity_optimize/phi_trace_decreasing_n300_resume10_20260801_093208.log
cz/data/fidelity_optimize/cz_trace_decreasing_n60_20260731_051843.log
cz/data/fidelity_optimize/cz_trace_decreasing_n60_resume25_20260801_093208.log
```

## 6. Theta optimization correction

The first Theta optimization used `max_step_ideal=1e-3`, whereas the accurate
Theta fidelity runner uses `3e-4 ns`. This numerical inconsistency made several
apparently optimized points worse than the original points.

`xgate/Xgate_fidelity_optimize.py` has been corrected to use:

```python
max_step_ideal = 3e-4 if drive_theta else 1e-3
```

At the snapshot used to write this guide, the corrected run had completed
`4/18` points. Its partial log is:

```text
xgate/data/fidelity_optimize/theta_trace_decreasing_n300_mstep3e4_reopt_20260802_004241.log
```

Resume the corrected run from source index 4 in a tmux window:

```bash
stamp=$(date +%Y%m%d_%H%M%S)
log="xgate/data/fidelity_optimize/theta_trace_decreasing_n300_mstep3e4_resume4_${stamp}.log"
nohup python xgate/Xgate_fidelity_optimize.py \
  --drive theta \
  --use-qt-fidelity false \
  --start-index 4 \
  > "$log" 2>&1 < /dev/null &
echo "Theta optimizer PID=$! log=$log"
```

Monitor it with:

```bash
tail -f "$log"
grep -c 'Optimal result for tg=' "$log"
```

The new resume log should contain 14 results. Combine the first four corrected
results and the 14 resumed results, then select five equally spaced points:

```bash
theta_resume_log="$log"
theta_five="xgate/data/fidelity_optimize/theta_optimized_five_points_corrected_${stamp}.csv"
python xgate/scripts/prepare_optimized_five_points.py \
  --old-log xgate/data/fidelity_optimize/theta_trace_decreasing_n300_mstep3e4_reopt_20260802_004241.log \
  --resume-log "$theta_resume_log" \
  --expected 18 \
  --output "$theta_five"
```

## 7. Unfinished optimized X-gate noisy calculations

The optimized five-point X-gate noisy jobs print their result array only after
all five Joblib jobs finish. At the snapshot time, none had printed a final
array, so restarting means rerunning all five points.

### Phi, T1 = 170 us

```bash
stamp=$(date +%Y%m%d_%H%M%S)
nohup python xgate/scripts/run_xgate_fidelity.py \
  --drive-phi true --drive-theta false \
  --n-full 300 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 170 \
  --pulse-file data/fidelity_optimize/phi_optimized_five_points_20260801_093726.csv \
  > "xgate/data/fidelity_qutip/phi_noise_trace_decreasing_n300_T1=170us_five_points_tmux_${stamp}.log" 2>&1 < /dev/null &
```

### Phi, T1 = 30 us

```bash
stamp=$(date +%Y%m%d_%H%M%S)
nohup python xgate/scripts/run_xgate_fidelity.py \
  --drive-phi true --drive-theta false \
  --n-full 300 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 30 \
  --pulse-file data/fidelity_optimize/phi_optimized_five_points_20260801_093726.csv \
  > "xgate/data/fidelity_qutip/phi_noise_trace_decreasing_n300_T1=30us_five_points_tmux_${stamp}.log" 2>&1 < /dev/null &
```

### Theta, T1 = 170 and 30 us

Run these only after the corrected Theta optimization is complete and replace
`$theta_five` with the corrected five-point CSV produced in section 6.

```bash
stamp=$(date +%Y%m%d_%H%M%S)

nohup python xgate/scripts/run_xgate_fidelity.py \
  --drive-phi false --drive-theta true \
  --n-full 300 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 170 \
  --pulse-file "${theta_five#xgate/}" \
  > "xgate/data/fidelity_qutip/theta_noise_trace_decreasing_n300_T1=170us_five_points_corrected_${stamp}.log" 2>&1 < /dev/null &

nohup python xgate/scripts/run_xgate_fidelity.py \
  --drive-phi false --drive-theta true \
  --n-full 300 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 30 \
  --pulse-file "${theta_five#xgate/}" \
  > "xgate/data/fidelity_qutip/theta_noise_trace_decreasing_n300_T1=30us_five_points_corrected_${stamp}.log" 2>&1 < /dev/null &
```

The `${theta_five#xgate/}` expression is intentional: the runner changes its
working directory to `xgate`, so `--pulse-file` is interpreted relative to
that directory.

## 8. Unfinished optimized CZ noisy calculations

The original optimized-pulse noisy logs had completed:

```text
T1=170 us: indices 0..21
T1=30 us:  indices 0..22
T1=3 us:   indices 0..19
```

The first continuation attempt subsequently completed one more point in each
case. At this snapshot the remaining source indices were:

```text
T1=170 us: [23, 24]
T1=30 us:  [24]
T1=3 us:   [21, 22, 23, 24]
```

Before rerunning, re-check counts because an existing Codex-managed task may
have progressed after this snapshot:

```bash
for f in cz/data/fidelity_qutip/*reopt_resume*.log; do
  printf '%s: ' "$f"
  grep -c '^f_.* = np.array' "$f"
done
```

If the snapshot is still current, run:

### CZ T1 = 170 us

```bash
stamp=$(date +%Y%m%d_%H%M%S)
nohup python cz/scripts/run_2q_fidelity.py \
  --n-truc-list 60 --tg-list 23,24 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 170 \
  --pulse-file data/fidelity_optimize/cz_reopt.txt \
  > "cz/data/fidelity_qutip/cz_noise_trace_decreasing_n60_T1=170us_reopt_tmux_${stamp}.log" 2>&1 < /dev/null &
```

### CZ T1 = 30 us

```bash
stamp=$(date +%Y%m%d_%H%M%S)
nohup python cz/scripts/run_2q_fidelity.py \
  --n-truc-list 60 --tg-list 24 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 30 \
  --pulse-file data/fidelity_optimize/cz_reopt.txt \
  > "cz/data/fidelity_qutip/cz_noise_trace_decreasing_n60_T1=30us_reopt_tmux_${stamp}.log" 2>&1 < /dev/null &
```

### CZ T1 = 3 us

```bash
stamp=$(date +%Y%m%d_%H%M%S)
nohup python cz/scripts/run_2q_fidelity.py \
  --n-truc-list 60 --tg-list 21,22,23,24 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 3 \
  --pulse-file data/fidelity_optimize/cz_reopt.txt \
  > "cz/data/fidelity_qutip/cz_noise_trace_decreasing_n60_T1=3us_reopt_tmux_${stamp}.log" 2>&1 < /dev/null &
```

Because `run_2q_fidelity.py` changes into the `cz` directory, its pulse file is
passed as `data/fidelity_optimize/cz_reopt.txt`, not with an extra `cz/` prefix.

## 9. CZ old-vs-optimized two-point comparison

The old leakage-inclusive noisy reference exists at source indices 12 and 27,
T1=170 us:

```text
cz/data/fidelity_qutip/cz_noise_trace_decreasing_n60_tg12_27_20260731_034623.log
```

The optimized comparison task had completed index 12 but not index 27 at this
snapshot. Rerun only index 27 using the full 30-row optimized table:

```bash
stamp=$(date +%Y%m%d_%H%M%S)
nohup python cz/scripts/run_2q_fidelity.py \
  --n-truc-list 60 --tg-list 27 \
  --calculate-ideal false --calculate-noise true \
  --use-qt-fidelity false --t1 170 \
  --pulse-file data/fidelity_optimize/cz_optimized_all30.csv \
  > "cz/data/fidelity_qutip/cz_noise_trace_decreasing_n60_T1=170us_optimized_compare_tg27_tmux_${stamp}.log" 2>&1 < /dev/null &
```

## 10. Notebook and plots

Notebook:

```text
cz/fidelity_formula.ipynb
```

Cells were appended at the bottom to:

1. reconstruct and display all optimized Phi, Theta, and CZ pulse parameters;
2. plot ideal unoptimized versus optimized leakage-inclusive fidelity;
3. add a QuTiP baseline line to all three ideal subplots;
4. show the relevant truncation in each title (`160`, `157`, and `60`);
5. plot CZ noisy old-versus-optimized results at indices 12 and 27;
6. plot currently completed partial CZ noisy curves for T1=170, 30, and 3 us;
7. plot all six CZ noisy lines in one figure.

The partial CZ plot currently compares optimized leakage-inclusive results with
the old noisy columns stored in `data_cz_fidelity_npz_select.txt`. Those old
stored columns use the QuTiP convention, and the notebook legend says so.
The separate two-point comparison uses leakage-inclusive fidelity for both old
and optimized pulses.

After new tmux logs finish, update the corresponding log paths in the bottom
notebook cells or copy/merge the new results into the paths currently read by
those cells, then rerun the cells from the optimized-data section downward.

Launch Jupyter from the repository root with:

```bash
jupyter lab
```

## 11. Monitoring

Monitor a log:

```bash
tail -f path/to/task.log
```

Count completed CZ points:

```bash
grep -c '^f_.* = np.array' path/to/cz_task.log
```

Count completed optimization points:

```bash
grep -c 'Optimal result for tg=' path/to/optimizer.log
```

X-gate noisy sweeps generally print the final fidelity array only after the
whole Joblib batch completes, so a zero count does not necessarily mean that
no worker has finished.

List Python tasks inside the tmux window:

```bash
ps -fu "$USER" | grep -E 'run_xgate_fidelity|run_2q_fidelity|Xgate_fidelity_optimize' | grep -v grep
```
