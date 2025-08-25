import sys
sys.path.append('../')

import pandas as pd
import datetime, pytz, os
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import utils_2Q_gate_zp as ut
from datetime import datetime
import numpy as np
import qutip as qt
from multiprocessing import Pool
import scqubits as scq
from sympy import symbols
from joblib import Parallel, delayed
import scipy.sparse as ssp


def import_select():
    max_step_ideal = 0 # Set max_step to 0 for parallel execution
    nsteps_ideal = 1e4# / max_step_ideal  # Set nsteps to a large number for parallel execution
    max_step_noisy = 1e-3 # Set max_step to 0 for parallel execution
    nsteps_noisy = 1/ max_step_noisy  # Set nsteps to a large number for parallel execution

    truc_full = 500
    num_cpus, n_job = 16, 10
    folder = f'../../data/3ncut_two_zeropi/truc1=300_truc2=1000_pick=True/'
    hspace_0 = pd.read_csv(folder+ 'hspace_0.txt').to_numpy().flatten()
    hspace_1 = pd.read_csv(folder+ 'hspace_1.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()[:truc_full]
    eket_tot = ssp.csr_matrix(np.load(folder+ 'eket_tot.npy'))[:truc_full]
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()[:truc_full]
    n_theta0_dress = 2*np.pi* np.load(folder+'n_theta0_dress.npy')
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')
    dim_0 = len(hspace_0)
    dim_1 = len(hspace_1)
    n_theta0_dress = ut.truncate_2(n_theta0_dress, np.arange(truc_full))
    n_theta1_dress = ut.truncate_2(n_theta1_dress, np.arange(truc_full))

    ### cnot part
    cnot = pd.read_csv('data/data_cnot_fidelity_3ncut.txt')
    x0_vec = cnot[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
                      ]].to_numpy()[ [2, 9, 16, 23, 30] ,:]
    # [[-1], :]     [ 2::3 ,:]   [ :13 ,:][1::2,:]
    mid_state = '8-2'
    idx_0 = hspace_full.index('0-2')
    idx_1 = hspace_full.index('2-2')
    idx_2 = hspace_full.index(mid_state)
    W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
    drive_term = n_theta0_dress
    hspace_select = [
    ## state_all
'0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
'5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
'13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
'9-0', '20-2', '20-5', '12-5', '4-0','15-5', '5-1', '18-2', '12-0', '24-2' ,
'26-5', '1-5', '15-0', '13-0', '25-2', '35-2', '8-1', '33-2', '9-5', '4-5' ,

'12-8', '9-8', '1-8', '37-2', '5-12',   ### this is needed
'25-0', '9-4', '1-25', '22-5', '56-2' ,
# '18-4', '13-9', '4-4', '2-4', '15-4',
'50-2', '13-5', '9-9', '5-9', '4-1' ,
'20-0', '25-5', '15-9', '18-5', '9-1', # is this need?
# '30-5', '2-21', '26-9', '1-13', '20-9' ,
# '45-2', '18-0', '5-16', '22-0', '34-5', '15-1', '5-4', '0-21', '20-4', '26-0' ,
'25-4', '25-1', '0-4', '9-12', '24-5', ### this is needed
'37-5', '24-0', '1-18', '12-9', '44-2' , # this is needed

# '25-8', '2-12', '41-2', '4-16', '4-12', '4-21', '8-12', '18-9', '18-1', '45-0' ,
'24-9', '26-8', '39-2', '4-8', '8-4', '34-0', '22-8', '35-4', '12-4', '30-0' , # this is needed
# '12-1', '28-2', '18-8', '0-8', '35-5', '35-0', '9-16', '37-0', '2-8', '1-9' ,
# '13-8', '46-2', '2-26', '46-0', '2-9', '41-0', '4-13', '39-0', '2-25', '0-12' ,
'22-9', '33-0', '33-4', '22-4', '54-2', '8-18', '28-1', '0-45', '50-0', '24-1' , # this is needed

# '28-5', '30-1', '15-8', '34-1', '22-1', '20-1', '13-16', '12-18', '0-9', '24-8' , #
'13-1', '51-2', '33-5', '2-30', '1-21', '4-18', '59-0', '15-12', '0-25', '8-24' , # this is needed
# '8-8', '9-18', '44-0', '26-1', '65-0', '56-0', '41-1', '41-4', '5-30', '39-4' , #
# '30-4', '58-0', '5-13', '18-12', '0-30', '45-4', '0-39', '13-4', '46-1', '28-0' ,
# '2-18', '0-24', '0-13', '9-13', '30-8', '69-0', '41-5', '33-1', '37-8', '4-24' ,

# '5-18', '51-0', '39-5', '12-20', '26-4', '2-13', '60-0', '12-16', '1-20', '12-12' ,
# '13-12', '44-1', '0-18', '5-25', '2-34', '1-42', '0-20', '24-4', '66-0', '8-13' ,
# '34-8', '54-0', '24-16', '5-35', '35-1', '20-8', '12-13', '20-12', '28-8', '1-24' ,
# '46-4', '54-1', '58-1', '34-4', '4-33', '1-30', '74-0', '4-30', '26-12', '15-13' ,
# '1-16', '1-28', '64-0', '12-21', '8-16', '30-9', '28-4', '4-20', '45-1', '75-0' ,

# '79-0', '81-0', '5-36', '51-1', '5-21', '5-24', '25-12', '24-12', '70-0', '15-21' ,
# '8-30', '28-9', '8-20', '18-16', '2-35', '13-25', '4-25', '39-1', '22-12', '2-36' ,
# '0-16', '18-13', '22-16', '60-1', '2-45', '8-21', '56-1', '2-16', '0-35', '1-12' ,
# '1-39', '37-4', '5-20', '50-1', '77-0', '35-8', '24-13', '37-1', '44-4', '8-25' ,
# '0-34', '59-1', '9-21', '0-36', '4-42', '13-18', '15-18', '25-9', '12-24', '15-16' ,
    ]

    ### common part
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    index_select = [hspace_full.index(i) for i in hspace_select]
    len_select = len(hspace_select)
    H0_full = qt.Qobj(np.diag(eval_tot))
    H0_select = ut.truncate_2( H0_full, index_select)
    drive_select = ut.truncate_2(drive_term, index_select)
    eket_tot = eket_tot[index_select]
    logi_idx_select = [hspace_select.index(i) for i in logi_state]
    H_drive_select = [ H0_select,   [drive_select, ut.drive_gauss_A],
                                    [drive_select, ut.drive_gauss_B]  ]    

    option_ideal =qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, num_cpus=1)  
    option_noisy =qt.Options(max_step=max_step_noisy, nsteps=nsteps_noisy, num_cpus=1)  
    print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')
    print(f'Noisy: max_step = {option_noisy.max_step}, nsteps = {option_noisy.nsteps}')
        
    print('\nmid_state = ', mid_state)
    print('truc_full=', truc_full )
    print('num_cpus=', num_cpus, ', n_job=', n_job)
    print('params =')
    for para in x0_vec:
        print(para.tolist(), ',')
    print(f'\nhspace_select (len={len_select}) = [')
    for i in range(0, len(hspace_select), 10):
        print(", ".join(f"'{x}'" for x in hspace_select[i:i + 10]), ',')
    print(']')

    states_all_index = [hspace_full.index(i) for i in hspace_select]
    data = states_all_index
    print(f'\nhspace_select_index = [')
    for i in range(0, len(data), 10):  # Step size of 10
        print(", ".join(f"{x}" for x in data[i:i + 10]), ',')
    print(']')

    # ### ideal fidelity
    # c_op_list = [qt.Qobj(np.zeros((len_select, len_select)))]
    c_op_list = []
    arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, 
                  logi_idx_select, mid_state, option_ideal, option_noisy]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_noise)(args_indep, *arg_select)
                                                for args_indep in x0_vec)
    print(f'\nf_ideal (dim={len(hspace_select)})  = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, np.round(f_ideal[i:i+4], 8).tolist())), ',')
    print(']')

    #################################################################
    ### Noisey fidelity
    # t1_tphi_other = 3 # μs

    # print(f"t1_tphi_other = {t1_tphi_other} μs", )
    # folder = f'../../data/3ncut_two_zeropi/truc1=500/'
    # gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    # gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    # gamma_decay_48_q0 = gamma_q0['t1_50us_48'].to_numpy() *50 /t1_tphi_other
    # gamma_decay_48_q1 = gamma_q1['t1_50us_48'].to_numpy() *50 /t1_tphi_other
    # gamma_dephase_02_q0 = gamma_q0['tphi_02'].to_numpy() *50 /t1_tphi_other
    # gamma_dephase_02_q1 = gamma_q1['tphi_02'].to_numpy() *50 /t1_tphi_other

    # qubit_a = True
    # arg_a = [dim_0, dim_1, gamma_decay_48_q0, gamma_dephase_02_q0, eket_tot, qubit_a] # old gamma
    # jump_op_a = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *arg_a) for state in range(1,dim_0))

    # qubit_a = False
    # arg_b = [dim_0, dim_1, gamma_decay_48_q1, gamma_dephase_02_q1, eket_tot, qubit_a] # old gamma
    # jump_op_b = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *arg_b) for state in range(1,dim_1))
    # jump_t1_list = np.array(jump_op_a)[:,0].tolist() + np.array(jump_op_b)[:,0].tolist()
    # jump_tphi_list = np.array(jump_op_a)[:,1].tolist() + np.array(jump_op_b)[:,1].tolist()
    # jump_t1_list = [qt.Qobj(matrix) for matrix in jump_t1_list]
    # jump_tphi_list = [qt.Qobj(matrix) for matrix in jump_tphi_list]

    # c_op_list = jump_t1_list + jump_tphi_list
    # arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_select]
    # f_noise = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_noise)(args_indep, *arg_select)
    #                                             for args_indep in x0_vec)
    # print(f'\nf_noise (dim={len(hspace_select)})  = [')
    # for i in range(0, len(f_noise), 4):
    #     print(', '.join(map(str, np.round(f_noise[i:i+4], 8).tolist())), ',')
    # print(']')


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_True_False() # compare np.abs(true) vs (False)
    # import_False()
    import_select()


    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




def import_True_False():

    cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    x0_vec = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()[[0, 5, 15, 25, 30],:]

    ### Get fidelity for input params (pick=True)
    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_full = 352

    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    n_theta0_dress = 2*np.pi* np.load(folder+'n_theta0_dress.npy')
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')

    truc_list = np.arange(truc_full)
    hspace_full = hspace_full[:truc_full]
    eval_tot = eval_tot[:truc_full]
    n_theta0_dress = ut.truncate_2(n_theta0_dress, truc_list)
    n_theta1_dress = ut.truncate_2(n_theta1_dress, truc_list)
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    num_cpus = 16
    n_job = 100
    c_op_list = []
    H0_full = qt.Qobj(np.diag(eval_tot))
    logi_idx_full = [hspace_full.index(i) for i in logi_state]
    H_drive_full = [H0_full, [n_theta1_dress, ut.drive_gauss_A] ]


    len_part = 190
    hspace_part = hspace_full[:len_part]
    index_part = np.arange(len_part)
    H0_part = ut.truncate_2( H0_full, index_part )
    n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
    logi_idx_part = [hspace_part.index(i) for i in logi_state]
    H_drive_part = [H0_part, [n_theta1_part, ut.drive_gauss_A] ]

    arg_part = [H_drive_part, W_20_50, num_cpus, c_op_list, logi_idx_part ]
    f_part = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_part)
                                                for args_indep in x0_vec)
    print(f' fidelity (dim={len_part},{charge_pick}) =', np.round(f_part, 8).tolist())


    arg_full = [H_drive_full, W_20_50, num_cpus, c_op_list, logi_idx_full ]
    f_full = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_full)
                                                for args_indep in x0_vec)
    print(f'fidelity (dim={truc_full},{charge_pick}) =', np.round(f_full, 8).tolist())



    ### Get fidelity for input params (pick=False)
    truc1, truc_tot, charge_pick = 300, 1000, False
    truc_full = 1000
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    n_theta1_dress =  2*np.pi* np.load(folder+'n_theta1_dress.npy')

    truc_list = np.arange(truc_full)
    hspace_full = hspace_full[:truc_full]
    eval_tot = eval_tot[:truc_full]
    n_theta1_dress = ut.truncate_2(n_theta1_dress, truc_list)
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    H0_full = qt.Qobj(np.diag(eval_tot))
    logi_idx_full = [hspace_full.index(i) for i in logi_state]
    H_drive_full = [H0_full, [n_theta1_dress, ut.drive_gauss_A] ]

    len_part = 500
    hspace_part = hspace_full[:len_part]
    index_part = np.arange(len_part)
    H0_part = ut.truncate_2( H0_full, index_part )
    n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
    logi_idx_part = [hspace_part.index(i) for i in logi_state]
    H_drive_part = [H0_part, [n_theta1_part, ut.drive_gauss_A] ]

    arg_part = [H_drive_part, W_20_50, num_cpus, c_op_list, logi_idx_part ]
    f_part = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_part)
                                                for args_indep in x0_vec)
    print(f' fidelity (dim={len_part},{charge_pick}) =', np.round(f_part, 8).tolist())


    arg_full = [H_drive_full, W_20_50, num_cpus, c_op_list, logi_idx_full ]
    f_full = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_full)
                                                for args_indep in x0_vec)
    print(f'fidelity (dim={truc_full},{charge_pick}) =', np.round(f_full, 8).tolist())



def import_False():

    # cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    # x0_vec = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()[[0, 5, 10, 15, 20, 25, 30],:]
    x0_vec = np.array([
[129.995414, 0.044102, 0.026819, -0.010756, -0.010313, -1.58074169, -1.52064501] ,

[160.000133, 0.036595, 0.022959, -0.008237, -0.007826, -2.00077611, -2.05780752] ,

[189.997344, 0.030915, 0.019755, -0.006194, -0.005848, -2.52671861, -2.53251749] ,

[219.997562, 0.026349, 0.016887, -0.00455, -0.004302, -2.84964821, -2.84286097] ,

[249.992868, 0.022922, 0.014688, -0.003423, -0.003252, -3.06948985, -3.06275222] ,

[280.000886, 0.020289, 0.013002, -0.002675, -0.002543, -3.23079202, -3.21686124] ,
    ])
    print('params =')
    for para in x0_vec:
        print(para.tolist(), ',')

    ### Get fidelity for input params (pick=False)
    truc1, truc_tot, charge_pick = 300, 2000, False
    truc_full = 2000
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    n_theta0_dress = 2*np.pi* np.load(folder+'n_theta0_dress.npy')
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')
    truc_list = np.arange(truc_full)
    hspace_full = hspace_full[:truc_full]
    eval_tot = eval_tot[:truc_full]
    n_theta0_dress = ut.truncate_2(n_theta0_dress, truc_list)
    n_theta1_dress = ut.truncate_2(n_theta1_dress, truc_list)

    mid_state = '8-2'
    idx_0 = hspace_full.index('0-2')
    idx_1 = hspace_full.index('2-2')
    idx_2 = hspace_full.index(mid_state)
    W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
    logi_state = ['0-0', '0-2', '2-0', '2-2']

    num_cpus = 16
    n_job = 100
    c_op_list = []

    H0_full = qt.Qobj(np.diag(eval_tot))
    logi_idx_full = [hspace_full.index(i) for i in logi_state]
    if mid_state in [ '8-2', '4-5', '1-4', '8-0' ]:
        H_drive_full = [ H0_full,     [n_theta0_dress, ut.drive_gauss_A],
                                        [n_theta0_dress, ut.drive_gauss_B]  ]
    else:
        H_drive_full = [ H0_full,     [n_theta1_dress, ut.drive_gauss_A],
                                        [n_theta1_dress, ut.drive_gauss_B]  ]

    # len_part = 500
    # hspace_part = hspace_full[:len_part]
    # index_part = np.arange(len_part)
    # H0_part = ut.truncate_2( H0_full, index_part )
    # n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
    # logi_idx_part = [hspace_part.index(i) for i in logi_state]
    # H_drive_part = [H0_part, [n_theta1_part, ut.drive_gauss_A] ]

    # arg_part = [H_drive_part, W_20_50, num_cpus, c_op_list, logi_idx_part ]
    # f_part = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_tg)(args_indep, *arg_part)
    #                                             for args_indep in x0_vec)
    # print(f' fidelity (dim={len_part},{charge_pick}) =')
    # for i in range(0, len(f_part), 4):
    #     print(', '.join(map(str, f_part[i:i+4])), ',')
    # print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    arg_full = [H_drive_full, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_full]
    f_full = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_tg)(args_indep, *arg_full)
                                                for args_indep in x0_vec[:,:5])
    print(f' fidelity (dim={truc_full},{charge_pick}) =')
    for i in range(0, len(f_full), 4):
        print(', '.join(map(str, f_full[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

















