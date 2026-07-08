# CZ sweep — optimize & re-optimize workflow

This directory (`cz/scripts/`) holds the scripts to **optimize a CZ gate over a
parameter sweep** of the two-qubit zero-pi qubit, and to **re-optimize** finished runs.

Run everything from **this directory** with the `zp2q` env active — several scripts use
relative paths that assume this cwd:

    cd /home/eweissler/src/two_qubit_gate_for_zero_pi_qubit/cz/scripts
    conda activate zp2q

**Sweep data** (input) lives outside the repo, one `<param>_sweep/` folder per sweep
(EC2, EJ2, ECJ2, EL2, Ecc, Ec0, ng2, Φ2), each a single `two_qubit_sweep_data.npz`
with all 21 points (coherent-only: `evecs_tot` dropped, loader returns `eket_tot=None`):

    DATA=/data/zp_sweeps/test2          # e.g. $DATA/EC2_sweep/two_qubit_sweep_data.npz

**Optimization results** (output) are written under the repo to
**`cz/data/results/cz_sweep_<param>_tg<gate>_<ts>/`** — one self-documenting dir per run
(npz + csv + `cz_sweep_summary.png`).

---

## Pipeline at a glance

    1. generate sweep data     sweep_data_generators/run_<param>_sweep.py   -> $DATA/<param>_sweep/two_qubit_sweep_data.npz
    2. optimize a CZ gate      optimize_cz_sweep.py --folder-load $DATA/<param>_sweep ...   -> cz_sweep_<param>_tg<gate>_<ts>/
    3. (optional) re-optimize  reoptimize_cz_sweep.py --result-dir <run>                  -> <run>_reopt/
    4. (re)generate plots      optimize_cz_sweep.py --plot <run>

Steps 2–3 have batch + monitor helpers (below).

---

## 1. Generate sweep data (upstream)

One generator per swept parameter, each writing `$DATA/<param>_sweep/two_qubit_sweep_data.npz`
(21 points, `truc_total=1000`). Heavy (sparse `eigsh` on ~54k-dim; **memory-bandwidth-
bound**), resumable via per-point checkpoints in `<param>_sweep/_points/`.

    python sweep_data_generators/run_ec2_sweep.py       # EC2, EJ2, ECJ2, EL2, Ecc, Ec0 — cheap-ish, can run several
    python sweep_data_generators/run_ng2_sweep.py       # ng2 and Φ2 are ~2.6x heavier -> run ALONE

> ng2/Φ2 have a complex Hamiltonian (nonzero offset charge/flux) and starve the memory
> bus if run alongside others — run them one at a time.

---

## 2. Optimize a CZ gate over a sweep

`optimize_cz_sweep.py` optimizes a CZ pulse `[tg, amp, detune]` at **every** sweep point
via differential evolution, warm-started outward from the center (each point seeded from
its neighbor toward the center; the central anchor from a pre-optimized pulse file).
Optimizes at `truc=300`, re-scores the optimum at `truc=1000`, then runs a fixed-pulse
drift eval calibrated at the sweep midpoint.

    python optimize_cz_sweep.py \
        --folder-load /data/zp_sweeps/test2/EC2_sweep \
        --workers 5 --n-cpu-inner 1 --tg-absolute --seed-tg-idx 71

### Flags

| flag | meaning |
|------|---------|
| `--folder-load PATH` | sweep folder holding `two_qubit_sweep_data.npz` |
| `--seed-tg-idx N` | seed gate time: **71 ≈ 91 ns**, **162 ≈ 182 ns** (labels the output dir `tg91`/`tg182`) |
| `--tg-absolute` | pin the `tg_bound` window on the fixed seed pulse for every point (gate time stays ~constant across the sweep) instead of drifting per point |
| `--workers N` | DE population workers |
| `--n-cpu-inner N` | inner sesolve parallelism per objective; total cores ≈ `workers × n_cpu_inner` |
| `--resume` | skip sweep indices already saved in the result file |
| `--dry-run` | optimize only the central anchor(s) + print the red-flag check |
| `--plot PATH` | regenerate infidelity + summary plots from an existing result and exit (no optimization) |

### Batch (all sweeps, both gate times) + monitor

    bash launch_cz_weekend.sh                  # 7 sweeps x (tg91@5 + tg182@10 workers); edit SWEEPS inside to change set
    nohup bash monitor_cz_weekend.sh &         # logs progress every 30 min -> $DATA/_cz_weekend_monitor.log

Output dirs never collide (labeled by param + gate + timestamp). Each run is resumable
(re-run its command with `--resume`).

---

## 3. Re-optimize (correctness + polish)

`reoptimize_cz_sweep.py` re-optimizes **every** point of a finished run, seeding each
from its **own** previous optimum. This does two things at once:

- **correctness** — uses the consistent `cz_fidelity_log_noise` evaluator (older runs'
  `truc=1000` re-score used a coarser legacy path, ~1% offset);
- **polish** — the per-point own-optimum warm start is a better seed than the original
  neighbor chain, so it shakes out quantized-step artifacts.

The original is preserved; results go to `<run>_reopt/`. `f_trunc` can't regress (the
prior optimum stays in the DE population).

    # one finished run:
    python reoptimize_cz_sweep.py --result-dir cz/data/results/cz_sweep_EC2_tg91_20260704_015433

Key flags: `--workers` / `--n-cpu-inner` (default: inherit the run's saved config),
`--in-place` (update the original instead of copying), `--out-dir PATH`,
`--no-fixed-eval` (skip re-doing the drift curve — not recommended).

### Batch: re-optimize everything that has finished

    DRYRUN=1 bash reoptimize_finished.sh       # preview: which finished runs would be refreshed
    bash reoptimize_finished.sh                # launch a re-opt for each finished, not-yet-refreshed run
    nohup bash reoptimize_finished_loop.sh &   # OR: auto-pick-up finishers every 45 min until all done

`reoptimize_finished.sh` is **idempotent** (skips any run that already has a `_reopt`
sibling or a re-opt in progress) and **core-neutral** (each re-opt inherits its
original's worker count, using ~the cores that finished run just released). Optional
args: `[WORKERS] [N_CPU_INNER]` to override.

> If a re-opt crashes partway, its `_reopt` dir persists and blocks retry —
> `rm -rf` that `_reopt` dir and the next scan re-launches it.

---

## 4. Plots

Every run writes `cz_sweep_summary.png` (2-column: optimized-per-point vs fixed-pulse
drift, plus CZ transitions / gate time / amplitude / detuning panels). Regenerate from
a saved npz (no recompute):

    python optimize_cz_sweep.py --plot cz/data/results/cz_sweep_<param>_tg<gate>_<ts>/   # summary + infidelity

**ng2 / Φ2** sweeps straddle zero (log-spaced symmetric about 0), so their plots use a
**symlog** x-axis automatically (double-sided log through 0). Refresh only existing
summary PNGs after those runs finish:

    RESULTS=/home/eweissler/src/two_qubit_gate_for_zero_pi_qubit/cz/data/results
    for d in "$RESULTS"/cz_sweep_ng2_*/ "$RESULTS"/cz_sweep_Phi2_*/; do
      [ -f "${d}cz_sweep_summary.png" ] || continue
      python -c "import sys; sys.path.insert(0,'.'); import optimize_cz_sweep as o; o.plot_sweep_summary('$d')"
    done

---

## Conventions / gotchas

- **Machine:** 128 physical cores / 256 SMT threads, ~2 TB RAM. CZ optimization is
  **compute-bound** (small-matrix sesolve), so budget against **128 physical** cores
  (SMT gives only ~15–25% extra for this FP-heavy work); keep total
  `Σ(workers × n_cpu_inner) ≲ 125`. Sweep *generation* is memory-bandwidth-bound and
  competes differently — the two can overlap, sweep-on-sweep cannot.
- **Gate times:** `--seed-tg-idx 71` ≈ 91 ns, `162` ≈ 182 ns. The 182 ns gate is ~2×
  slower per point (longer evolution) → give it ~2× the workers to finish alongside.
- **Fidelity evaluator:** optimization and all re-scoring use `ut.cz_fidelity_log_noise`
  (super-operator propagator, 3 pts/ns). Runs launched before this was standardized
  carry a ~1% offset in their `truc=1000` curve — that's what step 3 fixes.
- **Log files** (in `$DATA`): `opt_<param>_<gate>.out` (per optimize run),
  `opt_reopt_<run>.out` (per re-opt), `run_<param>.out` (per generation),
  `_cz_weekend_monitor.log`, `_reopt_finished_loop.log`.

_See also_ `/home/eweissler/src/two_qubit_gate_for_zero_pi_qubit/CZ_SWEEP_HANDOFF.md`
for the broader project context.
