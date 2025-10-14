# Two-Qubit Gate Implementation for Zero-π Qubits

This repository contains code for optimizing and analyzing quantum gates implemented on zero-π qubits, including single-qubit X gates and two-qubit gates (CZ, CNOT, and simultaneous X gates).

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

### a) Single Qubit X Gate

<!-- Instructions for optimizing and analyzing single-qubit X gate implementations -->

#### Setup

#### Running Optimization

#### Analysis

#### Output Files


### b) Two Qubit CZ Gate

<!-- Instructions for optimizing and analyzing two-qubit controlled-Z gate implementations -->

#### Setup

#### Running Optimization

#### Analysis

#### Output Files

### c) Two Qubit CNOT Gate

<!-- Instructions for optimizing and analyzing two-qubit CNOT gate implementations -->

#### Setup

#### Running Optimization

#### Analysis

#### Output Files

### d) Two Qubit X Gate on Each Qubit

<!-- Instructions for optimizing and analyzing simultaneous X gates on both qubits -->

#### Setup

#### Running Optimization

#### Analysis

#### Output Files

## File Structure

```
├── X_gate/                 # Single qubit X gate implementations
│   ├── data/               # X gate optimization results
│   └── get_fidelity_xgate.py
├── cnot/                   # CNOT gate implementations
│   ├── data/               # CNOT optimization results
│   └── cnot_fidelity_optimize.py
├── cz/                     # CZ gate implementations
│   ├── data/               # CZ optimization results
│   └── get_fidelity_cz.py
├── figure/                 # Figure generation and plotting code
├── data_fig/              # Data files for figures
├── utils_2Q_gate_zp.py    # Utility functions for two-qubit gates
├── .gitignore             # Git ignore file for data and cache files
└── README.md              # This file
```

## Citation

<!-- Add citation information after ArXiv submission -->
