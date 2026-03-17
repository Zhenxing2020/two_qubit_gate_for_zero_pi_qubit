# CNOT Folder Reorganized

This folder is reorganized by purpose.

## Layout

- `scripts/`: runnable CNOT/XI Python scripts.
- `notebooks/analysis/`: analysis notebooks.
- `notebooks/plot/`: plotting notebooks.
- `notebooks/exploration/`: exploratory notebooks.
- `archive/`: legacy or copied files.
- `history_code/`: historical versions kept as-is.
- `data/`: data files (unchanged).

## Main script entries

- `scripts/run_cnot_fidelity.py`
- `scripts/optimize_cnot_fidelity.py`
- `scripts/optimize_xi_fidelity.py`
- `scripts/optimize_xi_population.py`

## Notes

- Active scripts in `scripts/` now resolve imports/paths from `__file__`,
  so they can find `utils_2Q_gate_zp.py`, `ham_data.py`, and local `data/`.
