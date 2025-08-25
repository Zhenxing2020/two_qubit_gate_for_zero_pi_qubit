import os
import sys
sys.path.append('../')
import numpy as np
import utils_2Q_gate_zp as ut

if __name__ == "__main__":
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()
            
    gate = 'cnot' # 'x_gate_theta', 'x_gate_phi', 'cz', ''cnot
    n_full = 2000 # number of states in the full system
    n_truc = n_full # number of states in the graph model

    if gate == 'x_gate_phi': ## X-gate nphi
        A = [0.214044, 0.120290] # [0.02, 0.02]
        detune = [0.351830, 0.375322] # [0, 0]
        eval_tot, n_theta, n_phi, logi_state = ut.load_qubit_data_xgate() # Load spectrum and matrix elements
        drive_term = n_phi
        w_trans_1 = eval_tot[9] - eval_tot[0] + 2 * np.pi * detune[0]
        w_trans_2 = eval_tot[9] - eval_tot[2] + 2 * np.pi * detune[1]
        wd = [w_trans_1, w_trans_2]
        core_states = logi_state + [9]
        hspace_full = np.arange(n_full).tolist()
        print(f'Amplitude: A0={A[0]:.6f}, A1={A[1]:.6f}')
        print(f'detune_0={detune[0]:.6f}, detune_1={detune[1]:.6f}')

    elif gate == 'x_gate_theta': ## X-gate ntheta
        A = [0.12, 0.0287] # [0.02, 0.02]
        detune = [-0.00697 , -0.00204] # [0, 0]
        eval_tot, n_theta, n_phi, logi_state = ut.load_qubit_data_xgate() # Load spectrum and matrix elements
        drive_term = n_theta
        w_trans_1 = eval_tot[7] - eval_tot[0] + 2 * np.pi * detune[0]
        w_trans_2 = eval_tot[7] - eval_tot[2] + 2 * np.pi * detune[1]
        wd = [w_trans_1, w_trans_2]
        core_states = logi_state + [7]
        hspace_full = np.arange(n_full).tolist()    
        print(f'Amplitude: A0={A[0]:.6f}, A1={A[1]:.6f}')
        print(f'detune_0={detune[0]:.6f}, detune_1={detune[1]:.6f}')

    elif gate in ['cz', 'cnot']: # CZ & CNOT
        [hspace_full, _, eval_tot, n_theta0_dress, n_theta1_dress, 
         _, _, logi_state] = ut.load_qubit_data_2q(truc_full=n_full)

        if gate == 'cnot':
            #### mean=[0.04089, 0.023365]; median=[0.031868, 0.0203525]
            A = [0.031868, 0.0203525] 

            #### mean=[-0.00697,-0.0088359375]; median= [-0.006548, -0.006191]
            detune = [-0.006548, -0.006191] 
            drive_term = n_theta0_dress
            state_mid = '8-2'
            idx_0 = hspace_full.index('0-2')
            idx_1 = hspace_full.index('2-2')
            idx_2 = hspace_full.index(state_mid)
            core_states = logi_state + [state_mid]
            W_0_2 = eval_tot[idx_2] - eval_tot[idx_0] + 2 * np.pi * detune[0]
            W_1_2 = eval_tot[idx_2] - eval_tot[idx_1] + 2 * np.pi * detune[1]
            wd = [W_0_2, W_1_2]
        elif gate == 'cz':
            #### mean= 0.01717, median= 0.013895
            A = [0.013895] 

            #### mean=0.019343, median=0.018197
            detune = 0.018197 
            drive_term = n_theta1_dress
            state_mid = '5-0'
            core_states = logi_state + [state_mid]
            wd = (eval_tot[hspace_full.index(state_mid)] 
                  - eval_tot[hspace_full.index('2-0')] 
                  + 2 * np.pi * detune)
    print(f'gate={gate}, n_full={n_full}, n_truc={n_truc}')
    print(f'A={A}, detune={detune}')
    print(f'wd={wd},\n core_states={core_states}')

    # hspace_index_2 = trunc_by_thresh(hspace_index, drive_term, thresh=1e-2)
    # G = make_rate_graph(drive_term, evals, wd, A, labels = hspace_full)
    # df = make_leakage_df(core_states, drive_term, evals, wd, A, labels = hspace_full, n_cpu=100)
    
    ### states_short
    states_short = ut.trunc_by_graph_estimate(
        n_truc, core_states, drive_term, eval_tot, wd, A, 
        labels=hspace_full, path_func=ut.shortest_path_to_core)
    if gate in ['cz', 'cnot']:
        ut.print_data(f'{gate}_short_{n_full}_{n_truc}', states_short, 
                      num_each_row=10, n_make_blank_line=50)        

    states_short_index = [hspace_full.index(i) for i in states_short]
    ut.print_data(f'{gate}_short_index_{n_full}_{n_truc}', states_short_index,
                      num_each_row=10, n_make_blank_line=50)   
    
    
    ### states_all
    states_all = ut.trunc_by_graph_estimate(
        n_truc, core_states, drive_term, eval_tot, wd, A, 
        labels=hspace_full, path_func=ut.all_path_to_core)
    if gate in ['cz', 'cnot']:
        ut.print_data(f'{gate}_all_{n_full}_{n_truc}', states_all, 
                      num_each_row=10, n_make_blank_line=50)

    states_all_index = [hspace_full.index(i) for i in states_all]
    data = states_all_index
    ut.print_data(f'{gate}_all_index_{n_full}_{n_truc}', states_all_index,
                      num_each_row=10, n_make_blank_line=50)
     
    ut.compare_two_lists(states_short, states_all)




    