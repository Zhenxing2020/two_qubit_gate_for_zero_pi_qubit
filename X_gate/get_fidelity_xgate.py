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
import os
# os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "128"  # Or whatever upper limit you want  
import sys
sys.path.append('../')
from datetime import datetime
import pytz
import numpy as np
import scqubits as scq
import scqubits.settings as settings
import qutip as qt
from joblib import Parallel, delayed
import pandas as pd
import utils_2Q_gate_zp as ut
settings.OVERLAP_THRESHOLD = 0.3  # Update scqubits settings


def xgate_fidelity_decay_all():
    """
    Run X-gate fidelity simulations (ideal + noisy) and print results.
    """

    drive_phi, drive_theta, n_full = False, True, 150
    # drive_phi, drive_theta, n_full = True, False, 500 # 50 states →12 workers, (100 states/40 workers, 200/160). 
    t1 = 3 # μs
    tg_list = np.arange(18).tolist() # # [1, 5, 9, 13, 17 ] #   ## 18 for theta, 19 for phi     
    charge_truc = False  # whether to truncate the charge space
    calculate_ideal, calculate_noise = True, False # True, False # False, True #   whether to calculate noisy fidelity
    num_cpus = 4 # Lower num_cpus <4 can reduce num of workers while >4 won’t change the num.
    apply_decay, apply_dephase = True, True # False, True # True, False #


    if drive_theta:
        max_step_ideal, max_step_noisy = 3e-4, 3e-4 
    else:
        max_step_ideal, max_step_noisy = 1e-3, 1e-3 
    print("drive_phi=", drive_phi, "; drive_theta =", drive_theta, "; n_full =", n_full, "; charge_truc =", charge_truc)
    print(f"T1 = Tphi = {t1} μs, num_cpus = {num_cpus}")
    print(f"calculate_ideal = {calculate_ideal}, calculate_noise = {calculate_noise}")
    option_ideal, option_noisy = ut.get_qutip_options(max_step_ideal, max_step_noisy) 
    print(f'apply_decay = {apply_decay}, apply_dephase = {apply_dephase}')

    gamma_t1 = 1 / 1e3 / t1 # calculate decay rate given T1, unit in micro-second
    tphi = t1 # calculate decay rate given T1, unit in micro-second

    # Load data
    evals, n_theta, n_phi, logi_state = ut.load_qubit_data_xgate() # Load spectrum and matrix elements
    params = ut.load_drive_params_xgate(drive_theta)[tg_list, ]  # [1::4,] # Load pulse parameters from CSV
    n_job = len(params)    
    
    # Build Hamiltonian
    w_trans_1, w_trans_2, drive_term, Gamma_t1 = ut.compute_drive_xgate(evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1)  
    hspace = np.arange(n_full).tolist() # do not truncate
    if charge_truc:
        hspace = ut.get_truncated_subspace_xgate(drive_term, n_full)
    values_to_remove = {} # remove elements that give nan
    hspace = [item for item in hspace if item not in values_to_remove]

    n_hspace = len(hspace)
    H_qbt_drive, drive_truc, logi_idx = ut.build_hamiltonian_xgate(evals, drive_term, hspace, logi_state)

    # Print summary
    print("n_hspace =", n_hspace, ";   n_job = ", n_job)
    ut.print_data(f'hspace ({n_full}\{n_hspace})', hspace, num_each_row=10)
    ut.print_data(f'params', params.tolist(), num_each_row=1)

    if calculate_ideal:
        # Ideal fidelity simulation
        args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, [], logi_idx, option_ideal, option_noisy]
        f_ideal = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args) for args_indep in params)
        ut.print_data(f'f_ideal_{n_hspace}', f_ideal, num_digits=8)
        ut.print_time()

    if calculate_noise:
        # Load data and prepare operators for noisy fidelity simulation
        state_idx_tphi, gamma_dephase_new = ut.load_dephasing_data_xgate(drive_theta) # Load dephasing data
        c_op_list = ut.construct_c_ops_xgate(n_hspace, drive_truc, Gamma_t1, gamma_dephase_new, 
                                    tphi, hspace, state_idx_tphi, apply_decay, apply_dephase) # Construct collapse operators

        # Noisy fidelity simulation
        args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx, option_ideal, option_noisy]
        f_noise = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args) for args_indep in params)
        ut.print_data(f'f_{t1}us_{n_hspace}', f_noise, num_digits=8)

def xgate_population():
    # [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = [39.991616, 0.105045, 0.028058, 0.002171, 0.003988] # theta state 7
    # [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = [25.020473,0.180208,0.046132,-0.003789,0.001544 ]   # theta state 7
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = [828.759495, 0.013563, 0.034964, -0.003029, -0.003182 ]   # mix state 9
    # [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = [200.005395,0.208582,0.081626,0.335171,0.357777 ]   # phi
    # [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = [120.004766,0.209273,0.104767,0.341188,0.360503] # phi

    drive_phi, drive_theta = True, True
    # drive_phi, drive_theta = False, True
    # drive_phi, drive_theta = True, False

    charge_truc = True # truncate charge subspace
    calculate_ideal = True
    t1 = 999 # μs
    n_full = 300

    max_step_ideal = 1e-4 # Set max_step to 0 for parallel execution
    nsteps_ideal = 1/ max_step_ideal  # Set nsteps to a large number for parallel execution
    max_step_noisy = 3e-4 # Set max_step to 0 for parallel execution
    nsteps_noisy = 1/ max_step_noisy  # Set nsteps to a large number for parallel execution

    option_ideal =qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, store_states=True, num_cpus=1)  
    option_noisy =qt.Options(max_step=max_step_noisy, nsteps=nsteps_noisy, store_states=True, num_cpus=1)  
    print("drive_phi=", drive_phi, "; drive_theta =", drive_theta, "; n_full =", n_full, "; charge_truc =", charge_truc)
    print(f"T1 = Tphi = {t1} μs, tg = {tg:.0f}")
    print(f"calculate_ideal = {calculate_ideal}")
    print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')
    print(f'Noisy: max_step = {option_noisy.max_step}, nsteps = {option_noisy.nsteps}')

    evals, n_theta, n_phi = ut.load_qubit_data_xgate() # Load spectrum and matrix elements
    gamma_t1 = 1 / 1e3 / t1 # calculate decay rate given T1, unit in micro-second
    w_trans_1, w_trans_2, drive_term, Gamma_t1 = ut.compute_drive_xgate(evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1)  
    hspace = np.arange(n_full).tolist() # do not truncate
    if charge_truc:
        hspace = ut.get_truncated_subspace_xgate(drive_term, n_full)
    n_hspace = len(hspace)
    ut.print_data(f'hspace ({n_full}\{n_hspace})', hspace, num_each_row=10)

    logi_state = [0, 2]
    H_qbt_drive, drive_truc, logi_idx = ut.build_hamiltonian_xgate(evals, drive_term, hspace, logi_state)

    tlist = np.linspace(0, tg,  num=5* int(tg) )  # total time
    pulse_args = {'drive_amp_A': drive_amp_A ,
            'drive_freq_A': w_trans_1 + 2*np.pi*detune_A, #w_trans_1 + 2*np.pi*detune_A,
            'drive_amp_B': drive_amp_B ,
            'drive_freq_B': w_trans_2 + 2*np.pi*detune_B,
            'gate_time': tg,}
    states = [qt.basis(n_hspace, i) for i in range(n_hspace)]

    result = {}
    if not calculate_ideal:
        gamma_t1 = 1 / 1e3 / t1 # calculate decay rate given T1, unit in micro-second
        tphi = t1 # calculate decay rate given T1, unit in micro-second    
        w_trans_1, w_trans_2, drive_term, Gamma_t1 = ut.compute_drive_xgate(evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1)  

        # Load data and prepare operators for noisy fidelity simulation
        state_idx_tphi, gamma_dephase_new = ut.load_dephasing_data_xgate(drive_theta) # Load dephasing data
        tphi = t1 # calculate decay rate given T1, unit in micro-second 
        c_op_list = ut.construct_c_ops_xgate(n_hspace, drive_truc, Gamma_t1, gamma_dephase_new, 
                                    tphi, hspace, state_idx_tphi) # Construct collapse operators    
        
    for jdx, state_j in enumerate(logi_idx):
        result[jdx] = qt.mesolve(
            H_qbt_drive,
            states[state_j],
            tlist,
            c_ops=None if calculate_ideal else c_op_list,
            e_ops=[state * state.dag() for state in states],
            args=pulse_args,
            options=option_ideal if calculate_ideal else option_noisy,)

    arr = np.array(result[0].expect).sum(axis=1)
    # Get the indices of the top 10 largest values
    top_indices = np.argsort(arr)[-10:]  # Sort and take the last 10 indices
    # Get the top 10 largest values
    top_values = arr[top_indices]
    # Sorting in descending order (optional)
    sorted_order = np.argsort(top_values)[::-1]
    top_indices = top_indices[sorted_order]
    top_values = np.round(top_values[sorted_order] / np.sum(top_values), 4)

    state_phi = [0, 2, 9, 10, 19, 33, 37, 47] # phi
    state_phi_theta = [0, 2, 9] # phi
    state_theta = [0, 2, 7, 25] # theta
    col_phi = ['tg'] + [f'{i}' for i in state_phi] + ['other']
    col_theta = ['tg'] + [f'{i}' for i in state_theta] + ['other']
    col_phi_theta = ['tg'] + [f'{i}' for i in state_phi_theta] + ['other']

    if drive_phi and drive_theta:
        columns = col_phi_theta
        state_interest = state_phi_theta
    elif drive_phi and not drive_theta:
        columns = col_phi   
        state_interest = state_phi
    elif not drive_phi and drive_theta:
        columns = col_theta
        state_interest = state_theta

    ut.top_population(np.array(result[0].expect).sum(axis=1), hspace)
    ut.top_population(np.array(result[1].expect).sum(axis=1), hspace)
    state_interest_idx = [hspace.index(i) for i in state_interest]

    pop = np.array(result[0].expect)
    pop_interest = pop[state_interest_idx, :]
    pop_other = np.delete(pop, state_interest_idx, axis=0).sum(axis=0)
    pop_save = np.vstack((tlist,pop_interest, pop_other)).T

    print(f'data saved in data/population/population_phi={drive_phi}_theta={drive_theta}_tg={tg:.0f}_{t1}us_n={n_hspace}.txt')

    df = pd.DataFrame(pop_save, columns=columns)
    df.to_csv(f'data/population/population_phi={drive_phi}_theta={drive_theta}_tg={tg:.0f}_{t1}us_n={n_hspace}.txt', 
              sep=',', index=False, header=True)

if __name__ == '__main__':    
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    xgate_fidelity_decay_all()
    # xgate_population()

    ut.print_time()






