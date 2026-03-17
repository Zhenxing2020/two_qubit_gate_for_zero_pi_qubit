# CZ Folder Reorganized

This folder is reorganized by purpose.

## Layout

- `scripts/`: runnable CZ-related Python scripts.
- `notebooks/analysis/`: analysis notebooks.
- `notebooks/plot/`: plotting notebooks.
- `notebooks/exploration/`: exploratory notebooks.
- `archive/`: legacy or copied files.
- `history/`: historical versions kept as-is.
- `data/`: data files (unchanged).

## Main script entries

- `scripts/run_cz_fidelity.py`
- `scripts/run_2q_fidelity.py`
- `scripts/optimize_cz_fidelity.py`
- `scripts/prepare_cz_data.py`
- `scripts/analyze_cz_truncation_estimate.py`

## Notes

- Active scripts in `scripts/` now resolve imports/paths from `__file__`,
  so they can find `utils_2Q_gate_zp.py`, `ham_data.py`, and local `data/`.
