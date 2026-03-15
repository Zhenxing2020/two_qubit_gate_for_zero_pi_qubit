import os
# DIM = "128"
# os.environ["OPENBLAS_NUM_THREADS"] = "128"
# os.environ["OMP_NUM_THREADS"] = "128"
# os.environ["MKL_NUM_THREADS"] = "128"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "128"
# os.environ["NUMEXPR_MAX_THREADS"] = "128"  # Or whatever upper limit you want  
import sys
sys.path.append('../')
import scqubits.settings as settings
import utils_2Q_gate_zp as ut
import numpy as np
import qutip as qt
from joblib import Parallel, delayed
settings.OVERLAP_THRESHOLD = 0.3
from pathlib import Path
import ham_data as hd

if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    n_truc_list = [200, 250, 1000] # np.arange(100, 1001, 50) #
    cz_run = True # True  # whether to use CZ gate or CNOT gate
    
    # 300_2000_True; 300_2000_False
    # truc_one_qubit, truc_full, charge_pick = 300, 1000, True 

    # 'cz_short_500_detune0', 'cz_short_500_detune1', 'hand_pick',
    use_truc_model, truc_model_name = False, 'cz_short_500_detune1'

    # below sets whether to calculate noisy fidelity, they should not be true at the same time to avoid error
    calculate_ideal, calculate_noise = True, True # True, False #     
    apply_decay, apply_dephase = True, True # True, False # Whether to apply decay and dephasing
    decay_enlarge = 1 # change this to test decay
    filter_ratio = 0.3

    t1_tphi_other = 3 # μs
    tg_list = np.arange(180)[0::43] # [0::6] # [  0,  45,  90, 135, 179] # np.arange(180) #[0::3]
    # np.arange(181)[0::6] # [2, 9, 16, 23, 30] # Select the first row for testing
    max_step_ideal, max_step_noisy = 1e-3, 1e-3 # Set max_step to 0 for parallel execution
    num_cpus, n_job = 16, len(tg_list) # Number of CPUs and jobs for parallel processing

    # [hspace_full, eket_tot, eval_tot, n_theta0_dress, 
    #     n_theta1_dress, hspace_0, hspace_1, logi_state
    #     ] = ut.load_qubit_data_2q(truc_one_qubit, truc_full, charge_pick)

    folder_load = '../../data/_truc_3000'
    [hspace_full, eket_tot, eval_tot, n_theta0_dress, 
        n_theta1_dress, hspace_0, hspace_1, logi_state] = hd.load_two_qubit_data(folder_load, return_full=False)

    dim_0 = len(hspace_0)
    dim_1 = len(hspace_1) 

    # Load pulse parameters from CSV
    folder = 'data/npz/cz_pulse_neighbor.txt'
    params = ut.load_drive_params_2q(cz_run, folder=folder)[tg_list, ]  # [1::4,] 
    
#     params = np.array([
# [20.036223, 0.047529, 0.031549] ,
# [21.056504, 0.045424, 0.032751] ,
# [22.039611, 0.042818, 0.035569] ,
# [23.091315, 0.039794, 0.039868] ,
# [24.071078, 0.038628, 0.039974] ,
# [25.07626, 0.038301, 0.039951] ,
# [26.092321, 0.037932, 0.039998] ,
# [27.082449, 0.038031, 0.039582] ,
# [28.069609, 0.013817, 0.005016] ,
# [29.101602, 0.013343, 0.004844] ,
# [30.057238, 0.041657, 0.010669] ,
# [31.100583, 0.03923, 0.009722] ,
# [32.060376, 0.037386, 0.010602] ,
# [33.112569, 0.035734, 0.012895] ,
# [34.099918, 0.034463, 0.015224] ,
# [35.065909, 0.034531, 0.019579] ,
# [36.044359, 0.036397, 0.021658] ,
# [36.932952, 0.036409, 0.02184] ,
# [38.061268, 0.036401, 0.022479] ,
# [39.028991, 0.036205, 0.023243] ,
# [40.041038, 0.03528, 0.024113] ,
# [41.10079, 0.034937, 0.024358] ,
# [42.035916, 0.034499, 0.024743] ,
# [43.095904, 0.033641, 0.02493] ,
# [44.052773, 0.032822, 0.02579] ,
# [45.070904, 0.031818, 0.026331] ,
# [46.046183, 0.031097, 0.026723] ,
# [47.034873, 0.030229, 0.026999] ,
# [48.000298, 0.029445, 0.027353] ,
# [49.0707, 0.028438, 0.027504] ,
#     ])
        
    option_ideal, option_noisy = ut.get_qutip_options(max_step_ideal, max_step_noisy) 

    if cz_run: # CZ
        drive_term = n_theta1_dress
        W_20_50 = ( eval_tot[hspace_full.index('5-0')] - 
                    eval_tot[hspace_full.index('2-0')] )
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
    print(f"calculate_ideal = {calculate_ideal}, calculate_noise = {calculate_noise}")
    print(f'apply_decay={apply_decay}, apply_dephase={apply_dephase}')
    print(f'filter_ratio = {filter_ratio}, decay_enlarge = {decay_enlarge}')     
    print(f'use_truc_model = {use_truc_model}, truc_model_name = {truc_model_name}')
    print(f"t1_tphi_other = {t1_tphi_other}")
    # print(f"truc_one_qubit = {truc_one_qubit}, truc_full={truc_full}, charge_pick={charge_pick}")
    print('num_cpus=', num_cpus, ', n_job=', n_job)
    ut.print_fidelity(f'params', params.tolist(), num_each_row=1)    

    f_list = []
    for n_truc in n_truc_list:
        if use_truc_model:
            if cz_run:
                hspace_select = ut.truc_model[truc_model_name][:n_truc]
            else:
                hspace_select = ut.truc_model[truc_model_name][:n_truc]
        else:
            hspace_select = hspace_full[:n_truc]

        index_select = [hspace_full.index(i) for i in hspace_select]
        H_drive_select, eket_truc = ut.build_hamiltonian_2q(cz_run, index_select, eval_tot, 
                                                            eket_tot, drive_term)
        logi_idx_select = [hspace_select.index(i) for i in logi_state]

        ut.print_fidelity(f'hspace_select (len={len(hspace_select)})', 
                        hspace_select, num_each_row=10)

        ut.print_fidelity(f'index_select (len={n_truc})', index_select, num_each_row=10)

        if calculate_ideal: # ideal fidelity
            c_op_list = []
            if cz_run:
                arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, 
                                logi_idx_select, option_ideal, option_noisy]
                f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)
                                                    (args_indep, *arg_select)
                                                for args_indep in params)
            else:
                arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, 
                                logi_idx_select, mid_state, option_ideal, option_noisy]
                f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_noise)
                                                    (args_indep, *arg_select)
                                                for args_indep in params)        
            ut.print_fidelity(f'f_ideal_{n_truc}', f_ideal, num_digits=8)
            ut.print_time()

        if calculate_noise: # Noisey fidelity       
            [n_theta0, n_theta1, gamma_dephase_02_q0, gamma_dephase_02_q1
            ] = ut.load_noise_data_2q(t1_tphi_other) 

            Gamma = 1 / 1e3 / t1_tphi_other
            Gamma_decay_q0 = Gamma / (n_theta0[4,8]**2)
            Gamma_decay_q1 = Gamma / (n_theta1[4,8]**2)
            transition_a, n_theta0_trunc = ut.get_transitions_for_collapse(hspace_0, n_theta0, 
                                                                        filter_ratio=filter_ratio)
            transition_b, n_theta1_trunc = ut.get_transitions_for_collapse(hspace_1, n_theta1, 
                                                                        filter_ratio=filter_ratio)
            # Construct collapse operators
            c_op_list = ut.construct_c_ops_2q(dim_0, dim_1, n_theta0_trunc, n_theta1_trunc, 
                                                gamma_dephase_02_q0, gamma_dephase_02_q1, eket_truc, 
                                                Gamma_decay_q0, Gamma_decay_q1, 
                                                transition_a, transition_b, 
                                                apply_decay, apply_dephase, decay_enlarge )     
            print(f'np.shape(c_op_list) = {np.shape(c_op_list)}')     

            arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, logi_idx_select, 
                        option_ideal, option_noisy]
            
            f_noise = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)
                                                (args_indep, *arg_select)
                                            for args_indep in params)
            ut.print_fidelity(f'f_{t1_tphi_other}us_{n_truc}', f_noise, num_digits=8)
            ut.print_time()
        f_list.append(f_ideal if calculate_ideal else f_noise)
    print(f'n_truc_list = {np.array(n_truc_list).tolist()}')     
    ut.print_fidelity(f'fidelity_list', f_list, num_each_row=1, num_digits=8)

