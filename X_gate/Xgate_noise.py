"""
X-Gate Fidelity Simulation for the 0-π Qubit

This script simulates X-gate fidelities (ideal and noisy) for the 0-π qubit under theta or phi drive.
It uses matrix elements and spectrum data from precomputed files, constructs Hamiltonians,
computes gate fidelities, and accounts for dissipation including decay (T1) and pure dephasing (Tφ).

Main Features:
---------------
- Load energy spectrum and matrix elements for either the first or second zero pi qubit (0 or 1).
- Support both θ and φ drive.
- Construct truncated Hamiltonians based on magnitudes of charge matrix elements.
- Simulate both ideal and noisy X-gate fidelities using QuTiP solvers and multiprocessing.
- Print simulation results in a structured `np.array` format for easy copy-paste.

Inputs:
-------
- `drive_phi` (bool): Whether φ-drive is active.
- `drive_theta` (bool): Whether θ-drive is active.
- `n_full` (int): Dimension of the full Hilbert space before truncation.
- `t1` (float): T₁ lifetime in μs, also used as Tφ unless otherwise specified.
- `qubit_0` (bool): Whether to load data for qubit 0 (`True`) or qubit 1 (`False`).
- `tg_list` (List[int]): Indices of gate durations to simulate (corresponding to rows in drive parameter CSV).

Returns:
--------
- Prints formatted numpy arrays of:
  - `hspace_charge`: Truncated Hilbert space basis indices.
  - `params`: Drive parameters for each gate time.
  - `f_ideal`: Ideal gate fidelities (no dissipation).
  - `f_170us`: Noisy fidelities with T1 = Tφ = 170us.
- No values are returned programmatically, but results are displayed and can be logged/redirected if desired.

File Dependencies:
-------------------
- `data/zeropi_0_specdata_truc=1000_3ncut.h5.h5`: Spectrum and matrix elements generated from `scqubits` simulations.
- `data/data_xgate_theta_3ncut.txt` / `data_xgate_phi_3ncut.txt`: CSVs with drive parameters.
- `data/data_gamma_theta_500.txt` / `data_gamma_phi_500.txt`: CSVs with dephasing rates (computed separately).
- `utils_2Q_gate_zp.py`: Contains utility functions, such as fidelity calculations and matrix truncation.

"""

import sys
sys.path.append('../')
from datetime import datetime
import pytz, os
import numpy as np
import scqubits as scq
import scqubits.settings as settings
import qutip as qt
from joblib import Parallel, delayed
import pandas as pd
import utils_2Q_gate_zp as ut
settings.OVERLAP_THRESHOLD = 0.3  # Update scqubits settings


def load_simulation_data(qubit_0, folder = '../../data/3ncut_one_zeropi/'):
    """
    Loads the energy spectrum and matrix elements (n_theta, n_phi) for the 0-π qubit.
    The function "generate_data()" in sigmaX_fidelity_import_paras.py can generate the data

    Parameters:
        qubit_0 (bool): If True, load data for qubit 0; otherwise, load for qubit 1.
        folder (str): Path to the directory containing the data files.
    Returns:
        evals (np.ndarray): Energy levels.
        n_theta (np.ndarray): Matrix elements for theta drive.
        n_phi (np.ndarray): Matrix elements for phi drive.
    """    
    suffix = '0' if qubit_0 else '1'
    evals = 2 * np.pi * scq.read(folder + f'zeropi_{suffix}_specdata_truc=1000_3ncut.h5').energy_table
    n_theta = 2 * np.pi * scq.read(folder + f'zeropi_{suffix}_n_theta_truc=1000_3ncut.h5').matrixelem_table
    n_phi = 2 * np.pi * scq.read(folder + f'zeropi_{suffix}_n_phi_truc=1000_3ncut.h5').matrixelem_table
    evals -= evals[0]
    return evals, n_theta, n_phi

def load_drive_params(drive_theta):
    """
    Load X-gate drive parameters from a CSV file.

    Parameters:
        drive_theta (bool): If True, load theta-drive data; else load phi-drive data.

    Returns:
        np.ndarray: Parameters array. different rows mean different gate time. 
        columns mean 'tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
    """
    folder = 'data_xgate_theta_3ncut.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/' + folder)
    return f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()

def build_hamiltonian(H0, drive_term, hspace_charge, logi_state):
    """
    Constructs the truncated Hamiltonian and drive terms.

    Parameters:
        H0 (Qobj): Diagonalized bare Hamiltonian.
        drive_term (np.ndarray): Drive matrix.
        hspace_charge (list): Truncated Hilbert space indices.
        logi_state (list): Logical states, e.g., [0, 2].

    Returns:
        H_qbt_drive (list): Full driven Hamiltonian.
        drive_truc (Qobj): Truncated drive matrix.
        logi_idx (list): Logical state indices in truncated space.
    """
    H0_truc = ut.truncate_2(H0, hspace_charge)
    drive_truc = ut.truncate_2(drive_term, hspace_charge)
    logi_idx = [hspace_charge.index(s) for s in logi_state] # 1 state may or may not be in the truncated model, 
    H_qbt_drive = [H0_truc, [drive_truc, ut.drive_gauss_A], [drive_truc, ut.drive_gauss_B]]
    return H_qbt_drive, drive_truc, logi_idx

def construct_c_ops(n_charge, drive_truc, Gamma_t1, gamma_dephase_new, tphi, hspace_charge, state_idx):
    """
    Constructs collapse operators for dissipation.

    Parameters:
        n_charge (int): Truncated Hilbert space dimension.
        drive_truc (Qobj): Drive operator.
        Gamma_t1 (float): Amplitude decay prefactor.
        gamma_dephase_new (np.ndarray): Dephasing rates (for 50μs).
        tphi (float): Desired Tphi in μs.

    Returns:
        list: All collapse operators (amplitude + dephasing).
    """
    gamma_decay_new = Gamma_t1 * np.abs(drive_truc.full()) ** 2 # 
    # the dephasing rate is calculated in some file for 50μs for 2 state, the line below change dephasing coeffs to the input tphi (170, 30, 3μs)
    gamma_dephase_new = gamma_dephase_new * 50 / tphi
    jump_t1, jump_tphi = [], []
    for i in range(1, n_charge):
        for j in range(i): # only consider downwards deacy
            jump_t1.append(np.sqrt(gamma_decay_new[j, i]) * qt.basis(n_charge, j) * qt.basis(n_charge, i).dag())
    for i, state in enumerate(hspace_charge):
        idx = list(state_idx).index(state) 
        jump_tphi.append(np.sqrt(2 * gamma_dephase_new[idx]) * qt.basis(n_charge, i).proj())
    
    # print('np.shape(jump_t1)=',  np.shape(jump_t1), '; np.shape(jump_tphi)=',  np.shape(jump_tphi))
    return jump_t1 + jump_tphi

def print_data_r2r(label, fidelities, num_each_row=4):
    """
    Pretty-print fidelity array in readable blocks.

    Parameters:
        label (str): Label for the data array.
        fidelities (list): Fidelity values.
        num_each_row (int): Entries per row in output.
    """
    print(f"\n{label} = np.array([")
    for i in range(0, len(fidelities), num_each_row):
        print(', '.join(map(str, fidelities[i:i + num_each_row])), ',')
    print('])')

def load_dephasing_data(drive_theta):
    """
    Load dephasing rates calculated for 50μs.

    Parameters:
        drive_theta (bool): If True, load theta dephasing; otherwise, phi.

    Returns:
        np.ndarray: Dephasing rates for each state.
    """
    gamma_file = 'data/data_gamma_theta_500.txt' if drive_theta else 'data/data_gamma_phi_500.txt'
    gamma_new = pd.read_csv(gamma_file)
    gamma_dephase = gamma_new['tphi_50us_02'].to_numpy()
    state_idx = gamma_new['hspace'].to_numpy()
    return state_idx, gamma_dephase

def xgate_fidelity_decay_all(drive_phi=True, drive_theta=False, n_full=100, t1=170, tg_list=[], qubit_0=True):
    """
    Run X-gate fidelity simulations (ideal + noisy) and print results.

    Parameters:
        drive_phi (bool): Whether using phi-drive.
        drive_theta (bool): Whether using theta-drive.
        n_full (int): Full Hilbert space dimension.
        t1 (float): T1 relaxation time in μs.
        qubit_0 (bool): Whether using qubit 0 or 1.
        tg_list (list): List of indices of tg values to simulate.

    Returns:
        important inputs and final gate fidelities are printed in a structured format.       
    """
    gamma_t1 = 1 / 1e3 / t1 # calculate decay rate given T1, unit in micro-second
    tphi = t1 # calculate decay rate given T1, unit in micro-second

    # Load data
    evals, n_theta, n_phi = load_simulation_data(qubit_0) # Load spectrum and matrix elements
    params = load_drive_params(drive_theta)[tg_list, ]  # [1::4,] # Load pulse parameters from CSV
    num_cpus, n_job = 4, len(params)    
    
    # Build Hamiltonian
    w_trans_1, w_trans_2, drive_term, Gamma_t1 = ut.compute_drive_terms(evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1)  
    hspace_charge = ut.get_truncated_subspace(drive_term, n_full) # truncate relevant subspace
    # hspace_charge = np.arange(n_full).tolist() # do not truncate

    values_to_remove = {} # remove elements that give nan
    hspace_charge = [item for item in hspace_charge if item not in values_to_remove]

    n_charge = len(hspace_charge)
    H0 = qt.Qobj(np.diag(evals))
    H_qbt_drive, drive_truc, logi_idx = build_hamiltonian(H0, drive_term, hspace_charge, [0, 2])

    # Print summary
    print("drive_phi=", drive_phi, "; drive_theta =", drive_theta, "; n_full =", n_full)
    print("n_charge =", n_charge)
    print("num_cpus = ", num_cpus, ";   n_job = ", n_job)  
    print_data_r2r(f'hspace_charge ({n_full}\{n_charge})', hspace_charge, num_each_row=10)
    print_data_r2r(f'params', params.tolist(), num_each_row=1)
    # print("gamma = ", gamma)
    print(f"T1 = Tphi = {1/gamma_t1} ns") if gamma_t1 != 0 else None

    # Ideal fidelity simulation
    args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, [], logi_idx]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args) for args_indep in params)
    print_data_r2r(f'f_ideal_{n_charge}', f_ideal)
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # Load data and prepare operators for noisy fidelity simulation
    state_idx, gamma_dephase_new = load_dephasing_data(drive_theta) # Load dephasing data
    c_op_list = construct_c_ops(n_charge, drive_truc, Gamma_t1, gamma_dephase_new, tphi, hspace_charge, state_idx) # Construct collapse operators

    # Noisy fidelity simulation
    args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args) for args_indep in params)
    print_data_r2r(f'f_{t1}us_{n_charge}', f_noise)
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



if __name__ == '__main__':
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # drive_phi, drive_theta, n_full = True, False, 263
    drive_phi, drive_theta, n_full = False, True, 400
    t1 = 170
    tg_list = [9] # [1, 5, 9, 13, 17 ] # [9]
    xgate_fidelity_decay_all(drive_phi, drive_theta, n_full, t1, tg_list)




