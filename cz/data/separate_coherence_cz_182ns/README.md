# CZ: separate logical and auxiliary reference coherence scales

Script: `cz/scripts/run_2q_fidelity_separate_coherence.py` (independent copy;
`run_2q_fidelity.py` and shared utilities are unchanged).

## Configuration and execution

Edit `CONFIG` near the top of the script; no CLI options.

```bash
/home/zlqed/anaconda3/envs/zp2q/bin/python -u \
  cz/scripts/run_2q_fidelity_separate_coherence.py \
  > cz/data/separate_coherence_cz_182ns/run.log 2>&1
```

Selected pulse from `figure/data/data_cz_fidelity_npz_select.txt`:
`t_g=182.010007 ns`, amplitude `0.007880`, detuning `0.013127 GHz`.
Charge-picked truncation: 60 dressed states. Gaussian waveform and Hamiltonian
are inherited from the original script. See `config.json` for retained states.
80 flat workers, one logical density-matrix input operator per worker;
no nested process pools and one BLAS thread per worker. The repeated
(300, 30) us point is solved once and reused in both scans.

## Noise model

Manifolds are classified for each **bare single-qubit** source level before
dressing: logical = {0, 2}; auxiliary = all other levels.
Downward jumps, including auxiliary-to-logical jumps, use the source manifold.
Both qubits use the same pair of reference times.

For source j, choose T_j = T_logical or T_nonlogical in microseconds:

- Decay rate i <- j: `abs(n[i,j]/n[4,8])**2 / (1000*T_j)` in inverse ns.
- Dephasing gamma_j: `abs(saved_gamma[physical_level_j]) / T_j`.
- Dephasing operator: `sqrt(2*gamma_j) |j><j|`; ground projector omitted.

These T values are reference scales, not uniform lifetimes imposed on all
levels. In particular logical relaxation retains the original small matrix
elements; it is NOT forcibly set to 1/T_logical. Rates scale as 1/T, collapse
amplitudes as 1/sqrt(T). Both decay and dephasing are enabled.

Auxiliary decay retains the original `filter_ratio=0.3` selection. Every
nonzero logical-source downward matrix element within the selected bare basis
is explicitly restored, including weak 2->0 and 2->1 channels. Dephasing uses
**physical level indices**, not compressed charge-basis positions. This fixes
an indexing mismatch in the original script for nonconsecutive selected levels.
Operators are projected exactly as in the original simulation; no projection
renormalization or additional sink-state model is introduced.

## Fidelity and output

Fidelity uses the existing project's leakage-inclusive, trace-decreasing
average gate fidelity, with one optimized local-Z correction shared by the
entire channel (same implementation as the current original script).
Noiseless baseline: F = 0.9999340454097336 (approximately).

After all master-equation solves complete:

- `all_results.csv`: six scan rows, including the shared point twice.
- `gate_error_scans.png` and `.pdf`: the two requested 1-F scans, with noiseless baseline.
- `noise_audit_L*_A*.csv`: pre-projection bare-level rates for each pair.
- `channel_L*_A*_col*.npz`: final full density-matrix outputs, 16 per unique pair.
- `ideal.json`, `config.json`, `run.log`: baseline, full configuration, progress.

CSV diagnostics include mean full trace, mean logical survival and minimum
logical Choi eigenvalue. Solver tolerances are 1e-10 and maximum step is
0.001 ns; only the initial and final output times are stored. This changes
output storage, not the waveform or integration-step limit.
