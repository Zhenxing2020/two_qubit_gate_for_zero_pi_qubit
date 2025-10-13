# Two-Qubit Gate Implementation for Zero-π Qubits

This repository contains code for optimizing and analyzing quantum gates implemented on zero-π qubits, including single-qubit X gates and two-qubit gates (CZ, CNOT, and simultaneous X gates).

# To-Add

1. Instructions on how to run each of the 

2. Function in ham_data.py to add graph model to two qubit system after determining the 

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
in the same file.

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
