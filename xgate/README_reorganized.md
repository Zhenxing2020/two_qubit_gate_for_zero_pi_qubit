# xgate Reorganization Notes

This folder has been reorganized by function to make maintenance easier.

## Directory layout

- `scripts/`: main runnable Python scripts.
- `notebooks/analysis/`: analysis notebooks.
- `notebooks/plot/`: plotting notebooks.
- `notebooks/exploration/`: exploratory or method-development notebooks.
- `archive/`: legacy or superseded versions kept for reference.
- `data/`: existing data files (unchanged).

## Main script entry points

- `scripts/run_xgate_fidelity.py`
- `scripts/optimize_xgate_fidelity.py`
- `scripts/prepare_xgate_parameters.py`
- `scripts/analyze_xgate_fidelity_vs_truncation.py`

## Legacy moved to archive

- `archive/scripts/archive_run_xgate_fidelity_v2.py`
- `archive/scripts/archive_run_xgate_fidelity_legacy.py`
- `archive/scripts/archive_optimize_xgate_fidelity_legacy.py`
- `archive/scripts/optimize_xgate_fidelity_variant.py`
- `archive/scripts/optimize_xgate_fidelity_peter_variant.py`
- `archive/notebooks/archive_xgate_noise_model_decay_to_0.ipynb`

## Notes

- Active scripts in `scripts/` now resolve paths from `__file__`, so they can locate
  `utils_2Q_gate_zp.py` and `data/` after the move.
