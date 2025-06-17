import sys
import os
import pytz
from datetime import datetime
import numpy as np
import pandas as pd
import scipy as sp
import qutip as qt
import scqubits as scq
import scqubits.settings as settings
from tqdm import tqdm

import utils_2Q_gate_zp as ut

# Set threshold for matrix element overlap
settings.OVERLAP_THRESHOLD = 0.3

def construct_drive_matrix(evals, n_theta, n_phi, truc, drive_theta, drive_phi):
    """Constructs diagonalized Hamiltonians and drive operators."""
    H0 = qt.Qobj(np.diag(evals[:truc]))
    if drive_phi:
        w1 = evals[9] - evals[0]
        w2 = evals[9] - evals[2]
        drive = n_phi[:truc, :truc]
    elif drive_theta:
        w1 = evals[1] - evals[0]
        w2 = evals[2] - evals[1]
        drive = n_theta[:truc, :truc]
    return H0, drive, w1, w2

def extract_subspace(drive_term, truc, thresh=0.01, seed_indices=[0, 2]):
    """Finds relevant Hilbert subspace based on matrix elements."""
    hspace = seed_indices[:]
    for s in seed_indices:
        for i in range(truc):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace:
                hspace.append(i)
    return sorted(hspace)

def fidelity_de():
    """
    Optimize gate fidelity using differential evolution.

    Returns:
        None
    """
    fidelity = []
    drive_param = []
    fidelity_full = []
    n_cpu = 1
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu]

    for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)

        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_parallel,
            bounds=bounds,
            args=args,
            disp=True,
            callback=ut.print_soln,
            init="sobol",
            workers=workers,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=False
        )

        fidelity.append(res.fun)
        drive_param.append(res.x)

        # Evaluate full system fidelity
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = res.x
        argz = [H0_full, drive_full, w_trans_1, w_trans_2, hspace_full, 30,
                tg, drive_amp_A, drive_amp_B, detune_A, detune_B]
        fidelity_full.append(ut.xgate_fidelity_log(argz))

        # Print progress
        print(f"\nOptimal result for tg={tg}:")
        print(res)
        print(f"Gate errors (log10) (truc1={len(hspace_charge)}):")
        print(', '.join(map(str, np.round(fidelity[-4:], 8))))
        print(f"Drive parameters:")
        print(np.round(res.x, 6).tolist())
        print(f"Full system error (truc_full={truc_full}):")
        print(', '.join(map(str, np.round(fidelity_full[-4:], 8))))
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

if __name__ == '__main__':
    # Configuration
    drive_phi, drive_theta, drive_0  = False, True, True

    # Parameter bounds
    if drive_theta:
        amp1_bounds, amp2_bounds = (0.01, 0.05), (0.025, 0.2)
        detune1_bounds, detune2_bounds = (-0.17, -0.2), (0.1, 0.3)
        tg_vec = np.arange(40, 50, step=1).tolist()
    else:
        amp1_bounds, amp2_bounds = (0.15, 0.3), (0, 0.3)
        detune1_bounds, detune2_bounds = (0.3, 0.5), (0.3, 0.5)
        tg_vec = np.arange(10, 50, step=10).tolist()

    tg_bound = (-0.01, 0.01)
    folder = 'data_xgate_theta_3ncut.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/' + folder)

    # Differential evolution hyperparameters
    workers, popsize = 100, 10
    recombination, tol, mutation = 0.7, 0.01, (0.5, 1.0)
    truc1, truc_full = 150, 500

    # Load data from file
    truncation = 1000
    folder_path = '../../data/3ncut_one_zeropi/'

    if drive_0:
        evals = 2*np.pi*scq.read(folder_path + f'zeropi_0_specdata_truc={truncation}_3ncut.h5').energy_table
        n_Theta = 2*np.pi*scq.read(folder_path + f'zeropi_0_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_Phi = 2*np.pi*scq.read(folder_path + f'zeropi_0_n_phi_truc={truncation}_3ncut.h5').matrixelem_table
    else:
        evals = 2*np.pi*scq.read(folder_path + f'zeropi_1_specdata_truc={truncation}_3ncut.h5').energy_table
        n_Theta = 2*np.pi*scq.read(folder_path + f'zeropi_1_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_Phi = 2*np.pi*scq.read(folder_path + f'zeropi_1_n_phi_truc={truncation}_3ncut.h5').matrixelem_table

    evals -= evals[0]

    # Construct Hamiltonians
    H0, drive_term, w_trans_1, w_trans_2 = construct_drive_matrix(evals, n_Theta, n_Phi, truc1, drive_theta, drive_phi)
    H0_full, drive_full, *_ = construct_drive_matrix(evals, n_Theta, n_Phi, truc_full, drive_theta, drive_phi)

    # Define relevant Hilbert spaces
    hspace_charge = extract_subspace(drive_term, truc1)
    hspace_full = list(range(truc_full))

    # Print configuration
    print("drive_phi=", drive_phi, ", drive_theta=", drive_theta, ", drive_0=", drive_0)
    print("tg_vec:", tg_vec)
    print("amp1_bounds=", amp1_bounds, ", amp2_bounds=", amp2_bounds)
    print("detune1_bounds=", detune1_bounds, ", detune2_bounds=", detune2_bounds)
    print("truc1=", truc1, ", truc2 (optimized)=", len(hspace_charge))
    print("truc1_full=", truc_full, ", truc2_full=", len(hspace_full))

    # Run optimization
    fidelity_de()
