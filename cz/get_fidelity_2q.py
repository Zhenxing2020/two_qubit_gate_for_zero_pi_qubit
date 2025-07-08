import os
os.environ["MKL_NUM_THREADS"] = "128"
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

def get_fidelity_2q():
    cz_run = True # True  # whether to use CZ gate or CNOT gate
    import_2000 = False # whether to import 2000 or 1000 states Hamiltonian
    truc1 = 150 # truncation for single zero pi
    get_hamiltonian_given_state_list = False # False  # whether to build the Hamiltonian
    n_full = 1000 # don't change this value, If import_2000 = True, n_full=2000, else 1000
    n_hspace = 110 # <=n_full, Number of states to select from the full Hamiltonian

    calculate_ideal, calculate_noise = False, True # True, False #  whether to calculate noisy fidelity
    t1_tphi_other = 170 # μs
    tg_list = [30] # [2, 9, 16, 23, 30] # Select the first row for testing

    max_step_ideal = 1e-3 # Set max_step to 0 for parallel execution
    nsteps_ideal = 1/ max_step_ideal  # Set nsteps to a large number for parallel execution
    max_step_noisy = 1e-3 # Set max_step to 0 for parallel execution
    nsteps_noisy = 1/ max_step_noisy  # Set nsteps to a large number for parallel execution
    option_ideal =qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, num_cpus=1)  
    option_noisy =qt.Options(max_step=max_step_noisy, nsteps=nsteps_noisy, num_cpus=1)  

    [hspace_full, eket_tot, eval_tot, n_theta0_dress, 
     n_theta1_dress, dim_0, dim_1] = ut.load_qubit_data_2q(n_full, import_2000, truc1)
    num_cpus, n_job = 16, len(tg_list) # Number of CPUs and jobs for parallel processing
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    params = load_drive_params_2q(cz_run)[tg_list, ]  # [1::4,] # Load pulse parameters from CSV

    if cz_run:
        drive_term = n_theta1_dress
        W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    else:        
        drive_term = n_theta0_dress
        mid_state = '8-2'
        idx_0 = hspace_full.index('0-2')
        idx_1 = hspace_full.index('2-2')
        idx_2 = hspace_full.index(mid_state)
        W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
        W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]

    if get_hamiltonian_given_state_list:
        if cz_run: # CZ gate
            hspace_select = [ ### state_all_1000 ### charge_pick
    # '0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
    # '5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '0-9', '2-4', '9-0' ,
    # '1-5', '5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '0-13', '0-16' ,
    # '8-1', '4-4', '13-0', '15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5' ,
    # '0-18', '2-9', '1-12', '0-20', '0-21', '18-0', '9-2', '12-1', '5-5', '0-24' ,
    # '22-0', '2-12', '13-1', '24-0', '0-25' ,            

        '0-0', '5-0', '0-2', '2-0', '2-2', '5-2', '5-1', '0-1', '2-1', '1-0' ,
        '0-5', '2-5', '1-2', '9-0', '4-0', '2-4', '5-5', '1-1', '5-4', '1-5' ,
        '9-2', '0-4', '4-2', '2-8', '0-8', '9-1', '2-12', '2-9', '0-9', '0-12' ,
        '12-0', '2-16', '0-16', '5-8', '1-8', '4-5', '2-13', '2-21', '0-18', '2-20' ,
        '9-4', '4-9', '8-0', '2-18', '0-21', '2-24', '1-4', '18-0', '5-26', '0-13' ,

        '13-0', '0-26', '15-0', '2-26', '1-12', 
        '5-16', '8-1', '0-24', '15-1', '2-35' ,
        '15-4', '2-33', '2-45', '0-33', '2-39', '0-45', '5-12', '0-39', '5-9', '8-12' ,
        '5-33', '12-2', '4-4', '1-9', '4-1', '5-34', '2-30', '2-46', '1-16', '0-34' ,
        '2-34', '8-2', '5-21', '2-52', '0-52', '0-42', '2-42', '2-59', '0-59', '5-18' ,
        '0-65', '5-24', '2-55', '0-55', '0-20', '9-8', '8-9', '22-0', '1-25', '8-5' ,

        # '12-1', '4-8', '2-53', '2-36', '2-25', '5-20', '5-13', '9-24', '15-8', '1-30' ,
        # '0-25', '1-20', '5-25', '9-5', '1-13', '1-24', '13-2', '1-33', '18-1', '0-68' ,
        # '18-2', '20-0', '2-44', '9-12', '4-25', '9-9', '5-30', '0-83', '5-39', '9-16' ,
        # '0-36', '0-73', '12-4', '18-5', '22-2', '15-16', '1-21', '5-35', '9-13', '2-60' ,
        # '2-57', '15-2', '15-5', '2-28', '13-1', '4-12', '0-35', '9-20', '2-54', '1-18' ,

        # '0-81', '24-1', '4-39', '0-30', '1-26', '8-4', '4-16', '12-5', '1-35', '8-8' ,
        # '24-0', '12-9', '5-44', '0-77', '4-21', '0-57', '5-28', '1-39', '25-0', '8-18' ,
        # '1-28', '4-44', '33-0', '5-42', '12-12', '0-54', '13-12', '9-26', '4-24', '20-9' ,
        # '8-26', '24-4', '8-16', '0-46', '13-4', '1-34', '37-1', '0-44', '18-16', '22-4' ,
        # '1-45', '0-78', '9-25', '5-36', '24-9', '1-44', '4-35', '18-8', '0-53', '1-60' ,
            ]
        else: # CNOT gate
            hspace_select = [
'0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
'5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
'13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
'9-0', '20-2', '20-5', '12-5', '4-0', '15-5', '5-1', '18-2', '12-0', '24-2' ,
'26-5', '1-5', '15-0', '13-0', '25-2', '35-2', '8-1', '33-2', '9-5', '4-5' ,
'12-8', '9-8', '1-8', '37-2', '5-12', '25-0', '9-4', '1-25', '22-5', '56-2' ,
'50-2', '13-5', '9-9', '5-9', '4-1', '20-0', '25-5', '15-9', '18-5', '9-1' ,
'25-4', '25-1', '0-4', '9-12', '24-5', '37-5', '24-0', '1-18', '12-9', '44-2' ,
'24-9', '26-8', '39-2', '4-8', '8-4', '34-0', '22-8', '35-4', '12-4', '30-0' ,
'22-9', '33-0', '33-4', '22-4', '54-2', '8-18', '28-1', '0-45', '50-0', '24-1' ,
'13-1', '51-2', '33-5', '2-30', '1-21', '4-18', '59-0', '15-12', '0-25', '8-24' ,

    ## state_all
# '0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
# '5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
# '13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
# '9-0', '20-2', '20-5', '12-5', '4-0','15-5', '5-1', '18-2', '12-0', '24-2' ,
# '26-5', '1-5', '15-0', '13-0', '25-2', '35-2', '8-1', '33-2', '9-5', '4-5' ,

# '12-8', '9-8', '1-8', '37-2', '5-12',   ### this is needed
# '25-0', '9-4', '1-25', '22-5', '56-2' ,
# # '18-4', '13-9', '4-4', '2-4', '15-4',
# '50-2', '13-5', '9-9', '5-9', '4-1' ,
# '20-0', '25-5', '15-9', '18-5', '9-1', # is this need?
# # '30-5', '2-21', '26-9', '1-13', '20-9' ,
# # '45-2', '18-0', '5-16', '22-0', '34-5', '15-1', '5-4', '0-21', '20-4', '26-0' ,
# '25-4', '25-1', '0-4', '9-12', '24-5', ### this is needed
# '37-5', '24-0', '1-18', '12-9', '44-2' , # this is needed

# # '25-8', '2-12', '41-2', '4-16', '4-12', '4-21', '8-12', '18-9', '18-1', '45-0' ,
# '24-9', '26-8', '39-2', '4-8', '8-4', '34-0', '22-8', '35-4', '12-4', '30-0' , # this is needed
# # '12-1', '28-2', '18-8', '0-8', '35-5', '35-0', '9-16', '37-0', '2-8', '1-9' ,
# # '13-8', '46-2', '2-26', '46-0', '2-9', '41-0', '4-13', '39-0', '2-25', '0-12' ,
# '22-9', '33-0', '33-4', '22-4', '54-2', '8-18', '28-1', '0-45', '50-0', '24-1' , # this is needed

# # '28-5', '30-1', '15-8', '34-1', '22-1', '20-1', '13-16', '12-18', '0-9', '24-8' , #
# '13-1', '51-2', '33-5', '2-30', '1-21', '4-18', '59-0', '15-12', '0-25', '8-24' , # this is needed
# # '8-8', '9-18', '44-0', '26-1', '65-0', '56-0', '41-1', '41-4', '5-30', '39-4' , #
# # '30-4', '58-0', '5-13', '18-12', '0-30', '45-4', '0-39', '13-4', '46-1', '28-0' ,
# # '2-18', '0-24', '0-13', '9-13', '30-8', '69-0', '41-5', '33-1', '37-8', '4-24' ,

# # '5-18', '51-0', '39-5', '12-20', '26-4', '2-13', '60-0', '12-16', '1-20', '12-12' ,
# # '13-12', '44-1', '0-18', '5-25', '2-34', '1-42', '0-20', '24-4', '66-0', '8-13' ,
# # '34-8', '54-0', '24-16', '5-35', '35-1', '20-8', '12-13', '20-12', '28-8', '1-24' ,
# # '46-4', '54-1', '58-1', '34-4', '4-33', '1-30', '74-0', '4-30', '26-12', '15-13' ,
# # '1-16', '1-28', '64-0', '12-21', '8-16', '30-9', '28-4', '4-20', '45-1', '75-0' ,

# # '79-0', '81-0', '5-36', '51-1', '5-21', '5-24', '25-12', '24-12', '70-0', '15-21' ,
# # '8-30', '28-9', '8-20', '18-16', '2-35', '13-25', '4-25', '39-1', '22-12', '2-36' ,
# # '0-16', '18-13', '22-16', '60-1', '2-45', '8-21', '56-1', '2-16', '0-35', '1-12' ,
# # '1-39', '37-4', '5-20', '50-1', '77-0', '35-8', '24-13', '37-1', '44-4', '8-25' ,
# # '0-34', '59-1', '9-21', '0-36', '4-42', '13-18', '15-18', '25-9', '12-24', '15-16' ,
    ]
        index_select = [hspace_full.index(i) for i in hspace_select]
    else:
        index_select = np.arange(n_hspace)
        hspace_select = hspace_full[:n_hspace]

    ut.print_data_r2r(f'hspace_select (len={len(hspace_select)})', hspace_select, num_each_row=10)
    H_drive_select, eket_tot = ut.build_hamiltonian_2q(cz_run, index_select, eval_tot, eket_tot, drive_term)
    logi_idx_select = [hspace_select.index(i) for i in logi_state]
    len_select = len(index_select)

    if cz_run:
        print(f'Using CZ gate with W_20_50 = {W_20_50}')
    else:
        print(f'Using CNOT gate with W_0_2 = {W_0_2}, W_1_2 = {W_1_2}')
        print('\nmid_state = ', mid_state)

    print(f'get_hamiltonian_given_state_list = {get_hamiltonian_given_state_list}')
    print(f"t1_tphi_other = {t1_tphi_other}, import_2000={import_2000}")
    print('truc_full=', n_full )
    print('num_cpus=', num_cpus, ', n_job=', n_job)
    print(f"calculate_ideal = {calculate_ideal}, calculate_noise = {calculate_noise}")
    print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')
    print(f'Noisy: max_step = {option_noisy.max_step}, nsteps = {option_noisy.nsteps}')
    ut.print_data_r2r(f'params', params.tolist(), num_each_row=1)
    ut.print_data_r2r(f'index_select (len={len_select})', index_select, num_each_row=10)
    
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
        ut.print_data_r2r(f'f_ideal_{len_select}', f_ideal)
        ut.print_time()

    if calculate_noise: # Noisey fidelity        
        if cz_run:
            n_theta0, n_theta1, gamma_dephase_02_q0, gamma_dephase_02_q1 = load_noise_data(t1_tphi_other)
            c_op_list = construct_c_ops(dim_0, dim_1, n_theta0, n_theta1, gamma_dephase_02_q0, 
                                        gamma_dephase_02_q1, eket_tot) # Construct collapse operators
            
            np.savez(f'data/collapse_ops_truc1={truc1}_n={n_hspace}_{t1_tphi_other}us.npz', c_op_list=c_op_list)

            arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, logi_idx_select, 
                        option_ideal, option_noisy]
            f_noise = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_select)
                                                        for args_indep in params)
        ut.print_data_r2r(f'f_{t1_tphi_other}us_{len_select}', f_noise)
        print(f'np.shape(c_op_list) = {np.shape(c_op_list)}')


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    get_fidelity_2q()

    ut.print_time()
