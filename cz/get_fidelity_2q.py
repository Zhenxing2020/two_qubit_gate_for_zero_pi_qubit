import os
# DIM = "128"
os.environ["OPENBLAS_NUM_THREADS"] = "128"
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"
os.environ["VECLIB_MAXIMUM_THREADS"] = "128"
os.environ["NUMEXPR_MAX_THREADS"] = "128"  # Or whatever upper limit you want  
import sys
sys.path.append('../')

import pandas as pd
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import utils_2Q_gate_zp as ut
import numpy as np
import qutip as qt
from multiprocessing import Pool
import scqubits as scq
from sympy import symbols
from joblib import Parallel, delayed
import scipy.sparse as ssp


def load_drive_params_2q(cz_run):
    """
    Load CZ-gate drive parameters from a CSV file.
    """
    if cz_run:
        folder = 'data/data_cz_3ncut_truc1=300_select.txt'
        params = pd.read_csv(folder)[['tg', 'drive_amp', 'detune']].to_numpy()
    else:
        folder = '../cnot/data/data_cnot_fidelity_3ncut.txt'
        params = pd.read_csv(folder)[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()
    return params

def construct_c_ops(dim_0, dim_1, n_theta0, n_theta1, gamma_dephase_02_q0, gamma_dephase_02_q1, eket_tot):
    """
    Constructs collapse operators for dissipation.
    """
    qubit_a = True
    arg_a = [dim_0, dim_1, n_theta0, eket_tot, qubit_a] 
    state_vec_a = [(i,j) for i in range(dim_0) for j in range(dim_0) if i < j]
    jump_t1_a = Parallel(n_jobs=10)(delayed(ut.get_jump_op_decay)(state_vec, *arg_a) for state_vec in state_vec_a)

    arg_a = [dim_0, dim_1, gamma_dephase_02_q0, eket_tot, qubit_a] 
    jump_tphi_a = Parallel(n_jobs=10)(delayed(ut.get_jump_op_dephase)(state_j, *arg_a) for state_j in range(1,dim_0))

    qubit_a = False
    arg_b = [dim_0, dim_1, n_theta1, eket_tot, qubit_a] 
    state_vec_b = [(i,j) for i in range(dim_1) for j in range(dim_1) if i < j]
    jump_t1_b = Parallel(n_jobs=10)(delayed(ut.get_jump_op_decay)(state_vec, *arg_b) for state_vec in state_vec_b)

    arg_b = [dim_0, dim_1, gamma_dephase_02_q1, eket_tot, qubit_a] 
    jump_tphi_b = Parallel(n_jobs=10)(delayed(ut.get_jump_op_dephase)(state_j, *arg_b) for state_j in range(1,dim_1))

    return jump_t1_a + jump_tphi_a + jump_t1_b + jump_tphi_b

def load_noise_data(t1_tphi_other, folder = '../../data/3ncut_two_zeropi/truc1=500/'):
    """
    Load dephasing rates calculated for 50μs.    
    """    
    gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    n_theta0 = np.load(folder+'n_theta0.npy')
    n_theta1 = np.load(folder+'n_theta1.npy')
    gamma_dephase_02_q0 = gamma_q0['tphi_02'].to_numpy() *50 /t1_tphi_other
    gamma_dephase_02_q1 = gamma_q1['tphi_02'].to_numpy() *50 /t1_tphi_other
    return n_theta0, n_theta1, gamma_dephase_02_q0, gamma_dephase_02_q1

def get_fidelity_2q(n_truc=50):
    cz_run = True # True  # whether to use CZ gate or CNOT gate
    import_2000_states = False # whether to import 2000 or 1000 states Hamiltonian
    truc_one_qubit = 300 # truncation for single zero pi
    n_full = 1000 # don't change this value, If import_2000 = True, n_full=2000, else 1000
    # n_truc = 100 # <=n_full, Number of states to select from the full Hamiltonian
    use_truc_model, truc_model_name = True, 'all_path' # 'short_path', 'all_path', 'hand_pick'

    calculate_ideal, calculate_noise = True, False # False, True #  whether to calculate noisy fidelity
    t1_tphi_other = 170 # μs
    tg_list = [2, 9, 16, 23, 30] # Select the first row for testing

    max_step_ideal = 1e-3 # Set max_step to 0 for parallel execution
    nsteps_ideal = 1 / max_step_ideal  # Set nsteps to a large number for parallel execution
    max_step_noisy = 1e-3 # Set max_step to 0 for parallel execution
    nsteps_noisy = 1/ max_step_noisy  # Set nsteps to a large number for parallel execution
    option_ideal =qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, num_cpus=1)  
    option_noisy =qt.Options(max_step=max_step_noisy, nsteps=nsteps_noisy, num_cpus=1)  

    num_cpus, n_job = 16, len(tg_list) # Number of CPUs and jobs for parallel processing
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    [hspace_full, eket_tot, eval_tot, n_theta0_dress, n_theta1_dress, dim_0, dim_1, logi_state
     ] = ut.load_qubit_data_2q(n_full, import_2000_states, truc_one_qubit)
    params = load_drive_params_2q(cz_run)[tg_list, ]  # [1::4,] # Load pulse parameters from CSV

    if cz_run: # CZ
        drive_term = n_theta1_dress
        W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
        print(f'Using CZ gate with W_20_50 = {W_20_50}')
    else: # CNOT     
        drive_term = n_theta0_dress
        mid_state = '8-2'
        idx_0 = hspace_full.index('0-2')
        idx_1 = hspace_full.index('2-2')
        idx_2 = hspace_full.index(mid_state)
        W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
        W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
        print(f'Using CNOT gate with W_0_2 = {W_0_2}, W_1_2 = {W_1_2}')
        print('\nmid_state = ', mid_state)

    if use_truc_model:
        if cz_run:
            hspace_select = ut.cz_truc_model[truc_model_name][:n_truc]
        else:
            hspace_select = ut.cnot_truc_model[truc_model_name][:n_truc]
    else:
        hspace_select = hspace_full[:n_truc]
    index_select = [hspace_full.index(i) for i in hspace_select]
    H_drive_select, eket_tot = ut.build_hamiltonian_2q(cz_run, index_select, eval_tot, eket_tot, drive_term)
    logi_idx_select = [hspace_select.index(i) for i in logi_state]

    ut.print_data(f'hspace_select (len={len(hspace_select)})', hspace_select, num_each_row=10)
    print(f'get_hamiltonian_given_state_list = {use_truc_model}')
    print(f"t1_tphi_other = {t1_tphi_other}, import_2000={import_2000_states}")
    print('truc_full=', n_full )
    print('num_cpus=', num_cpus, ', n_job=', n_job)
    print(f"calculate_ideal = {calculate_ideal}, calculate_noise = {calculate_noise}")
    print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')
    print(f'Noisy: max_step = {option_noisy.max_step}, nsteps = {option_noisy.nsteps}')
    ut.print_data(f'params', params.tolist(), num_each_row=1)
    ut.print_data(f'index_select (len={n_truc})', index_select, num_each_row=10)
    
    if calculate_ideal: # ideal fidelity
        c_op_list = []
        if cz_run:
            arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, 
                          logi_idx_select, option_ideal, option_noisy]
            f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_select)
                                                        for args_indep in params)
        else:
            arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_select, 
                          mid_state, option_ideal, option_noisy]
            f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_noise)(args_indep, *arg_select)
                                                        for args_indep in params)        
        ut.print_data(f'f_ideal_{n_truc}', f_ideal)
        ut.print_time()

    if calculate_noise: # Noisey fidelity        
        if cz_run:
            n_theta0, n_theta1, gamma_dephase_02_q0, gamma_dephase_02_q1 = load_noise_data(t1_tphi_other)
            c_op_list = construct_c_ops(dim_0, dim_1, n_theta0, n_theta1, gamma_dephase_02_q0, 
                                        gamma_dephase_02_q1, eket_tot) # Construct collapse operators
            
            np.savez(f'data/collapse_ops_truc1={truc_one_qubit}_n={n_truc}_{t1_tphi_other}us.npz', c_op_list=c_op_list)

            arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, logi_idx_select, 
                        option_ideal, option_noisy]
            f_noise = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_select)
                                                        for args_indep in params)
        ut.print_data(f'f_{t1_tphi_other}us_{n_truc}', f_noise)
        print(f'np.shape(c_op_list) = {np.shape(c_op_list)}')
    return f_ideal    


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    # get_fidelity_2q()
    f_list = []
    n_truc_list = [225, 250, 275, 300, 325, 350, 375, 400] # [50, 75, 100, 125, 150, 175, 200]
    for n_truc in n_truc_list:
        f_list.append(get_fidelity_2q(n_truc))
    print(f'n_truc_list={n_truc_list}')        
    ut.print_data(f'f_list', f_list, num_each_row=1, num_digits=8)


    ut.print_time()
