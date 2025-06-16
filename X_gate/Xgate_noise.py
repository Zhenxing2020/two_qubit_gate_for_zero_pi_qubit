import sys
sys.path.append('../')
from datetime import datetime
import pytz, os
import numpy as np
import scipy as sp
from tqdm import tqdm
import scqubits as scq
import scqubits.settings as settings
import qutip as qt
from multiprocessing import Pool
from joblib import Parallel, delayed
import pandas as pd
import utils_2Q_gate_zp as ut
import scipy.sparse as ssp
settings.OVERLAP_THRESHOLD = 0.3  # Update scqubits settings


def load_simulation_data(qubit_0):
    """
    Loads the energy spectrum and matrix elements (n_theta, n_phi) for the 0-π qubit.

    Returns:
        evals (np.ndarray): Energy levels.
        n_theta (np.ndarray): Matrix elements for theta drive.
        n_phi (np.ndarray): Matrix elements for phi drive.
    """
    folder = '../../data/3ncut_one_zeropi/'
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
        drive_theta (bool): Flag indicating which drive type to load.

    Returns:
        np.ndarray: Parameters array.
    """
    folder = 'data_xgate_theta_3ncut.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/' + folder)
    return f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()[1::4, :]

def compute_drive_terms(evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1):
    """
    Computes the transition frequencies and selects the appropriate drive term.

    Returns:
        w1 (float): Transition frequency 1.
        w2 (float): Transition frequency 2.
        drive_term (np.ndarray): Matrix elements for selected drive.
        Gamma_t1 (float): the decay rate coefficient fixed by certain transition matrix element
    """
    if drive_phi:
        w1 = evals[9] - evals[0]
        w2 = evals[9] - evals[2]
        drive_term = n_phi
        Gamma_t1 = gamma_t1 / (np.abs(n_phi[4,9])**2)
    elif drive_theta:
        w1 = evals[7] - evals[0]
        w2 = evals[7] - evals[2]
        drive_term = n_theta
        Gamma_t1 = gamma_t1 / (np.abs(n_theta[4,7])**2)
    else:
        raise ValueError("Either drive_phi or drive_theta must be True")
    return w1, w2, drive_term, Gamma_t1

def get_truncated_subspace(drive_term, n_full, hspace_charge=[0,2]):
    """
    Determines a reduced Hilbert space based on significant coupling elements.

    Returns:
        hspace_charge (list): List of basis indices to include.
    """
    for s in hspace_charge:
        for i in range(n_full):
            if np.abs(drive_term[s, i] / (2 * np.pi)) > 0.01 and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
    return hspace_charge

def build_hamiltonian(H0, drive_term, hspace_charge, logi_state):
    """
    Constructs the truncated Hamiltonian and drive terms.

    Returns:
        H_qbt_drive (list): Full driven Hamiltonian.
        drive_truc (Qobj): Truncated drive matrix.
        logi_idx (list): Logical state indices in truncated space.
    """
    H0_truc = ut.truncate_2(H0, hspace_charge)
    drive_truc = ut.truncate_2(drive_term, hspace_charge)
    logi_idx = [hspace_charge.index(s) for s in logi_state]
    H_qbt_drive = [H0_truc, [drive_truc, ut.drive_gauss_A], [drive_truc, ut.drive_gauss_B]]
    return H_qbt_drive, drive_truc, logi_idx

def construct_c_ops(n_truc, drive_truc, Gamma_t1, gamma_dephase_new, t1):
    """
    Constructs collapse operators for dissipation.

    Returns:
        list: All collapse operators (amplitude + dephasing).
    """
    gamma_decay_new = Gamma_t1 * np.abs(drive_truc.full()) ** 2
    gamma_dephase_new *= 50 / t1
    jump_t1, jump_tphi = [], []
    for i in range(1, n_truc):
        for j in range(i):
            jump_t1.append(np.sqrt(gamma_decay_new[i, j]) * qt.basis(n_truc, j) * qt.basis(n_truc, i).dag())
        jump_tphi.append(np.sqrt(2 * gamma_dephase_new[i]) * qt.basis(n_truc, i).proj())
    # print('np.shape(jump_t1)=',  np.shape(jump_t1), '; np.shape(jump_tphi)=',  np.shape(jump_tphi))
    return jump_t1 + jump_tphi

def print_data_r2r(label, fidelities, num_each_row=4):
    """
    Prints fidelities in a structured format.
    """
    print(f"\n{label} = np.array([")
    for i in range(0, len(fidelities), num_each_row):
        print(', '.join(map(str, fidelities[i:i + num_each_row])), ',')
    print('])')

def load_dephasing_data(drive_theta):
    """
    Loads the dephasing rates from a CSV file.

    Returns:
        np.ndarray: Dephasing rates.
    """
    gamma_file = 'data/data_gamma_theta_500.txt' if drive_theta else 'data/data_gamma_phi_500.txt'
    gamma_new = pd.read_csv(gamma_file)
    return gamma_new['tphi_50us_02'].to_numpy()

def xgate_fidelity_decay_all(drive_phi=True, drive_theta=False, n_full=100, t1=170, qubit_0=True):
    """
    Runs the X-gate fidelity simulation with and without noise.
    Prints the results for further analysis.
    """
    gamma = 1 / 1e3 / t1 # calculate decay rate given T1, unit in micro-second

    # Load data
    evals, n_theta, n_phi = load_simulation_data(qubit_0) # Load spectrum and matrix elements
    params = load_drive_params(drive_theta) # Load pulse parameters from CSV
    num_cpus, n_job = 4, len(params)    
    
    # Build Hamiltonian
    w_trans_1, w_trans_2, drive_term, Gamma_t1 = compute_drive_terms(evals, n_theta, n_phi, drive_phi, drive_theta, gamma)  
    hspace_charge = get_truncated_subspace(drive_term, n_full) # truncate relevant subspace
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
    print(f"T1 = Tphi = {1/gamma} ns") if gamma != 0 else None

    # Ideal fidelity simulation
    args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, [], logi_idx]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args) for args_indep in params)
    print_data_r2r(f'f_ideal_{n_charge}', f_ideal)
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # Load data and prepare operators for noisy fidelity simulation
    gamma_dephase_new = load_dephasing_data(drive_theta) # Load dephasing data
    c_op_list = construct_c_ops(n_charge, drive_truc, Gamma_t1, gamma_dephase_new, t1) # Construct collapse operators

    # Noisy fidelity simulation
    args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args) for args_indep in params)
    print_data_r2r(f'f_{t1}us_{n_charge}', f_noise)
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



if __name__ == '__main__':
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # drive_phi, drive_theta, truc = True, False, 100
    drive_phi, drive_theta, truc = False, True, 75
    t1 = 170

    xgate_fidelity_decay_all(drive_phi, drive_theta, truc, t1)




