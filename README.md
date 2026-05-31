# Two-Qubit Gate Implementation for 0-π Qubits

This repository contains code for optimizing and analyzing quantum gates implemented on 0-π qubits, including single-qubit X gates and two-qubit gates (CZ, CNOT, and simultaneous X gates).

## Environment Requirements

### Required Packages
- QuTip (4.X)
- SCqubits
- NumPy
- SciPy
- Matplotlib
- Jupyter

### Installation
Clone the repository and make sure you have a suitable environment for running the code. An anaconda environment yaml is included if you would like to create a fresh environment. Use the following command to create and activate the zp2q environment.

```bash
conda env create -f env_zp2q.yaml
conda activate zp2q
```

## How to Generate Hamiltonians

To generate all the Hamiltonian data, you simply run 
```bash
python ham_data.py
```
from the root of the repository. A new folder will be created in the data directory titled by the current time and date. The directory contains data files, a copy of the circuit parameters yaml used to generate the data, and a summary text file. Alternatively to generate only single or two-qubit hamiltonian data, you can run
```bash
python ham_data.py --data-type single
```
or
```bash
python ham_data.py --data-type two
```
or a more detailed example 
```bash
nohup time python ham_data.py --yaml 'data/params.yaml'  --data-type single > 'data/gen_q1_flux.txt' &
```
where 'nohup' ensures to run code that you want to continue after you disconnect from server. Ignore it if you are using your local computer.

Hamiltonian data is saved to .npz files, which can be read directly or using the helper functions
```python
load_single_qubit_data
```
or
```python
load_two_qubit_data
```
or
```python
load_single_qubit_data_for_two_qubits
```
in the same file. Note that values in the .npz files follow the scQubits units convention (i.e. $h=1$), as opposed to the qutip convention $\hbar = 1$. So you must multiply by $2 \pi$ for qutip dynamics.

## NPZ file format — saved datasets

General notes
- Frequencies/energies are saved in the same units used throughout the code (e.g., GHz). They are NOT multiplied by 2π unless loader functions explicitly do so.

Two-qubit file (example: two_qubit_data.npz)
- eval1 (np.ndarray): eigenvalues of single-qubit 1 (length = params["truc1"]). Units: GHz (no 2π).
- eval2 (np.ndarray): eigenvalues of single-qubit 2 (length = params["truc2"]). Units: GHz (no 2π).
- evecs1 (np.ndarray): eigenvectors for qubit 1 (each row is an eigenvector).
- evecs2 (np.ndarray): eigenvectors for qubit 2 (each row is an eigenvector).
- n_theta1, n_theta2 (np.ndarray): theta-mode charge operator matrices for each qubit in the bare single-qubit basis (no 2π).
- hspace_1_charge, hspace_2_charge (array/list): selected charge-basis indices after truncation-by-threshold.
- g_theta1theta2 (float): extracted coupling coefficient between theta modes. Units consistent with evals.
- evals_tot (np.ndarray): full-system eigenvalues for the truncated two-qubit Hamiltonian (no 2π).
- evecs_tot (np.ndarray): corresponding full-system eigenvectors.
- hspace_full (list): labels for full-system bare basis states (order matches tensor-product basis used).
- n_theta1_dressed, n_theta2_dressed (np.ndarray): n_theta operators represented in the dressed (full-system eigenbasis).
- top_idx (list): for each dressed eigenstate, top contributing bare-basis index pairs (i,j).
- top_overlap (list/array): complex overlaps corresponding to top_idx entries.
- hspace_n_theta1, hspace_n_theta2 (list/array): reduced indices used for any truncated analysis of n_theta in the dressed subspace.
- params (dict): parameters dictionary (from params.yaml) used to generate the dataset.

Single-qubit file (example: single_qubit_data.npz)
- evals (np.ndarray): single-qubit eigenvalues (length = params["truc"]). Units: GHz (no 2π).
- evecs (np.ndarray): single-qubit eigenvectors (each row is an eigenvector).
- n_theta (np.ndarray): theta-mode charge operator in the single-qubit eigenbasis (no 2π).
- n_phi (np.ndarray): phi-mode charge operator in the single-qubit eigenbasis (no 2π).
- phi_drive (dict), theta_drive (dict), mixed_drive (dict): drive metadata dicts containing:
  - w_trans_1, w_trans_2: transition frequencies used for graph estimates (same units as evals).
  - drive_term: operator matrix used as the drive.
  - hspace_charge: truncated charge indices selected by threshold.
  - hspace_graph: truncated indices selected by graph-based estimate.
- params (dict): parameters dictionary (from params.yaml) used to generate the dataset.

Quick usage examples
- Inspect keys:
  - import numpy as np
  - data = np.load("two_qubit_data.npz", allow_pickle=True)
  - list(data.keys())
- Load eigenvalues:
  - evals = data["evals"]

Loader note
- Helper functions in this repo (load_two_qubit_data, load_single_qubit_data) may multiply eigenvalues or operators by 2π on return. Verify the loader behavior before combining results with external code.


## Running Instructions

Activate the environment first, then run each gate script from its gate directory.

### a) Single Qubit X Gate

The X-gate fidelity and population simulations are run by:

```bash
cd xgate
python scripts/run_xgate_fidelity.py
```

The script configuration is edited in `xgate/scripts/run_xgate_fidelity.py`, inside `main()`.
Set `mode = "fidelity"` or `mode = "population"` and update the drive, truncation,
coherence, and timestep settings there before running.

Key settings in `common_args`:
- `mode`: choose `"fidelity"` or `"population"`.
- `drive_phi`, `drive_theta`: select the drive channel. Use only one for a pure phi/theta drive, or set both to `True` for the combined drive.
- `qubit_0`: choose which qubit is driven when loading the single-qubit X-gate data.
- `t1`: relaxation/coherence time in us; the script also uses this value for `tphi`.
- `charge_truc`: enable charge-based Hilbert-space truncation before time evolution.
- `calculate_ideal`, `calculate_noise`: turn ideal/noisy simulations on or off.
- `num_cpus`, `parallel_jobs`: control CPU usage for the QuTiP solver and joblib parallel runs.
- `apply_decay`, `apply_dephase`: include or exclude decay and dephasing channels in noisy simulations.

After `common_args`, the script selects gate-specific parameters for theta, phi, or combined theta+phi drives. Update `tg_list`, `n_full`, `max_step_ideal`, `max_step_noisy`, and the `population_args` values in the matching drive branch.

#### Output Files
- X-gate outputs are written under `xgate/data/`.


### b) Two-Qubit CZ and CNOT Gates

CZ and CNOT fidelity simulations are now combined in one script:

```bash
cd cz
python scripts/run_2q_fidelity.py
```

The script configuration is edited in `cz/scripts/run_2q_fidelity.py`, inside `main()`.
Set `cfg["cz_run"] = True` for CZ or `cfg["cz_run"] = False` for CNOT, then update
`cfg["tg_list"]`, `cfg["n_truc_list"]`, coherence settings, and model options as needed.

Key settings in `cfg`:
- `cz_run`: set `True` for CZ and `False` for CNOT.
- `t1_tphi_other`: relaxation/dephasing time in us for noisy two-qubit simulations.
- `n_truc_list`: list of truncation sizes to test.
- `reduced_model`: choose the reduced Hilbert-space selection rule, such as `"charge_pick"`, `"graph_pick"`, or `"lowest_state"`.
- `calculate_ideal`, `calculate_noise`: turn ideal/noisy fidelity calculations on or off.
- `apply_decay`, `apply_dephase`: include or exclude decay and dephasing collapse operators.
- `decay_enlarge`, `filter_ratio`: tune the noisy collapse-operator construction.
- `tg_para`: set `True` to parallelize over the full `tg_list`; set `False` to run each gate time separately with memory reporting.
- `tg_list`: choose rows from the pulse-parameter table.
- `params`: pulse parameters loaded from the CZ or CNOT data file according to `tg_list`.

For CZ, the default pulse table is `cz/data/npz/cz_pulse_neighbor.txt`. For CNOT, the script currently loads `../figure/data/data_cnot_fidelity_npz.txt`. Keep `tg_list` and `params` aligned: the number of selected gate times must match the number of selected parameter rows.

#### Output Files
- Two-qubit outputs and input pulse tables are stored under `cz/data/`.

## File Structure

```
├── cz/                         # Combined CZ and CNOT gate simulations
│   ├── data/                    # Two-qubit pulse tables and simulation data
│   └── scripts/
│       └── run_2q_fidelity.py   # Main CZ/CNOT fidelity entry point
├── data/                       # Hamiltonian datasets and parameter files
├── figure/                     # Figure generation notebook and figure data
│   ├── data/                    # Data used by plotting notebooks
│   └── pdf_before_inkscape/     # Local-only generated PDF figures
├── xgate/                      # Single-qubit X gate simulations
│   ├── data/                    # X-gate fidelity and population outputs
│   └── scripts/
│       └── run_xgate_fidelity.py # Main X-gate fidelity/population entry point
├── env_zp2q.yaml               # Conda environment file
├── ham_data.py                 # Hamiltonian data generation script
├── paths.py                    # Shared path helpers
├── truncation_estimate.py      # Truncation analysis utilities
├── utils_2Q_gate_zp.py         # Shared utilities for zero-π gate simulations
└── README.md                   # This file
```

## Citation

<!-- Add citation information after ArXiv submission -->
