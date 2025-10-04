# Two-Qubit Gate Implementation for Zero-π Qubits

This repository contains code for optimizing and analyzing quantum gates implemented on zero-π qubits, including single-qubit X gates and two-qubit gates (CZ, CNOT, and simultaneous X gates).

## Environment Requirements

<!-- Add your Python environment setup, dependencies, and installation instructions here -->

### Required Packages
- QuTip version YY
- SCqubits version X
- NumPy
- SciPy
- Matplotlib
- Jupyter

### Installation
```bash
# Add installation commands here
conda env create -f env_qutip4.yml
conda activate qutip4
```

### Environment Setup
```bash
# Add conda/pip environment setup commands here
pip install -r requirements.txt
```

## How to Generate Hamiltonians

<!-- Describe the process for generating the Hamiltonian matrices for your zero-π qubit systems -->

### Single Qubit Hamiltonian
- Generate zero-π qubit Hamiltonian using SCqubits
- Set truncation parameters for charge and flux basis states
- Define coupling parameters and external flux

### Two Qubit Hamiltonian
- Construct composite system with two zero-π qubits
- Include coupling between qubits (capacitive or inductive)
- Set individual truncation parameters for each qubit

### Truncation and Basis Selection
- Determine optimal truncation numbers based on convergence studies
- Select computational basis states
- Handle leakage to non-computational states

## How to Run Optimization/Analysis

### a) Single Qubit X Gate

<!-- Instructions for optimizing and analyzing single-qubit X gate implementations -->

#### Setup
- Navigate to `X_gate/` directory
- Set up system parameters in configuration files
- Define optimization targets (fidelity, gate time, etc.)

#### Running Optimization
```bash
# Add command to run single qubit X gate optimization
python get_fidelity_xgate.py
jupyter notebook DRAG_model.ipynb
```

#### Analysis
- Pulse optimization for n_theta drive
- Noiseless simulations for n_theta transitions
- Noisy simulations including decoherence effects
- Parameter sensitivity analysis

#### Output Files
- Optimized pulse parameters saved to `X_gate/data/`
- Fidelity results and population dynamics
- Error analysis and robustness studies

### b) Two Qubit CZ Gate

<!-- Instructions for optimizing and analyzing two-qubit controlled-Z gate implementations -->

#### Setup
- Navigate to `cz/` directory
- Configure two-qubit system parameters
- Set up flux bias points and coupling strengths

#### Running Optimization
```bash
# Add command to run CZ gate optimization
python get_fidelity_cz.py
jupyter notebook cz_fidelity_optimize.ipynb
```

#### Analysis
- Pulse optimization for CZ gate implementation
- Noiseless simulations for CZ dynamics
- Noisy simulations with realistic decoherence
- Geometric phase analysis

#### Output Files
- Optimized pulse parameters saved to `cz/data/`
- Gate fidelity results and leakage analysis
- Population dynamics for all computational states

### c) Two Qubit CNOT Gate

<!-- Instructions for optimizing and analyzing two-qubit CNOT gate implementations -->

#### Setup
- Navigate to `cnot/` directory
- Set up CNOT gate decomposition or direct implementation
- Configure system parameters and drive schemes

#### Running Optimization
```bash
# Add command to run CNOT gate optimization
python cnot_fidelity_optimize.py
jupyter notebook cnot_fidelity.ipynb
```

#### Analysis
- Pulse optimization for CNOT gate
- Noiseless simulations for CNOT dynamics
- Noisy simulations with decoherence effects
- Comparison with CZ + single-qubit decomposition

#### Output Files
- Optimized pulse parameters saved to `cnot/data/`
- CNOT fidelity results and error analysis
- Performance comparison studies

### d) Two Qubit X Gate on Each Qubit

<!-- Instructions for optimizing and analyzing simultaneous X gates on both qubits -->

#### Setup
- Configure simultaneous single-qubit operations
- Set up independent drive lines for each qubit
- Define crosstalk and coupling effects

#### Running Optimization
```bash
# Add command to run simultaneous X gate optimization
python XI_fidelity_optimize.py
jupyter notebook XI_population.ipynb
```

#### Analysis
- Optimization of simultaneous X gates (XI operation)
- Crosstalk analysis between qubits
- Comparison with sequential single-qubit operations
- Parallel gate execution efficiency

#### Output Files
- Optimized pulse parameters for simultaneous operations
- Crosstalk characterization data
- Performance benchmarks

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

<!-- Add citation information if this work should be cited -->
