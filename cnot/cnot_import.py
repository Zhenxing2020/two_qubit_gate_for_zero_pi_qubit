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


def import_select():

    cnot = pd.read_csv('data/data_cnot_fidelity_3ncut.txt')
    x0_vec = cnot[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
                      ]].to_numpy()[1::5, :]
    # [0::3,:]

    ### Get fidelity for input params (pick=True)
    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_full = 500
    num_cpus, n_job = 16, len(x0_vec)

    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
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

    mid_state = '8-2'
    idx_0 = hspace_full.index('0-2')
    idx_1 = hspace_full.index('2-2')
    idx_2 = hspace_full.index(mid_state)
    W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    hspace_select = [

'0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
'5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
'13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
'9-0', '20-2', '20-5', '12-5', '4-0',
'15-5', '5-1', '18-2', '12-0', '24-2' ,
'26-5', '1-5', '15-0', '13-0', '25-2',
 '35-2', '8-1', '33-2', '9-5', '4-5' ,

'12-8', '9-8', '1-8', '37-2', '5-12',
# '25-0', '9-4', '1-25', '22-5', '56-2' ,
# '18-4', '13-9', '4-4', '2-4', '15-4', '50-2', '13-5', '9-9', '5-9', '4-1' ,
# '20-0', '25-5', '15-9', '18-5', '9-1', '30-5', '2-21', '26-9', '1-13', '20-9' ,
# '45-2', '18-0', '5-16', '22-0', '34-5', '15-1', '5-4', '0-21', '20-4', '26-0' ,
'25-4', '25-1', '0-4', '9-12', '24-5',
# '37-5', '24-0', '1-18', '12-9', '44-2' ,

    ]
    index_select = [hspace_full.index(i) for i in hspace_select]
    len_select = len(hspace_select)
    H0_full = qt.Qobj(np.diag(eval_tot))
    H0_select = ut.truncate_2( H0_full, index_select)
    n_theta0_select = ut.truncate_2(n_theta0_dress, index_select)
    eket_tot = eket_tot[index_select]
    logi_idx_select = [hspace_select.index(i) for i in logi_state]
    H_drive_select = [ H0_select,   [n_theta0_select, ut.drive_gauss_A],
                                    [n_theta0_select, ut.drive_gauss_B]  ]

    print('\nmid_state = ', mid_state)
    print('truc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_full )
    print('num_cpus=', num_cpus, ', n_job=', n_job)
    print('params =')
    for para in x0_vec:
        print(para.tolist(), ',')
    print(f'\nhspace_select (len={len_select}) = [')
    for i in range(0, len(hspace_select), 10):
        print(", ".join(f"'{x}'" for x in hspace_select[i:i + 10]), ',')
    print(']')

    # #################################################################
    # ### ideal fidelity
    # c_op_list = [qt.Qobj(np.zeros((len_select, len_select)))]
    c_op_list = []
    arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_select]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_optimize)(args_indep, *arg_select)
                                                for args_indep in x0_vec)
    print(f'\nf_ideal (dim={len(hspace_select)},{charge_pick})  = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, np.round(f_ideal[i:i+4], 8).tolist())), ',')
    print(']')


    #################################################################
    ### Noisey fidelity
    tphi_logi = 100 # μs
    t1_other = 5 # μs
    gamma_decay_logi =  1 / 1600e3
    gamma_dephase_logi = 1 / 1e3 / tphi_logi
    gamma_decay_other =  1 / 1e3 / t1_other
    gamma_dephase_other = 1 / 1e3 / t1_other
    gamma_decay_old   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * 300
    gamma_dephase_old = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * 300

    folder = f'../../data/3ncut_two_zeropi/truc1=500/'
    gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    gamma_decay_28_q0 = gamma_q0['t1_50us_28'].to_numpy() *50 /t1_other
    gamma_decay_28_q1 = gamma_q1['t1_50us_28'].to_numpy() *50 /t1_other
    gamma_decay_08_q0 = gamma_q0['t1_50us_08'].to_numpy() *50 /t1_other
    gamma_decay_08_q1 = gamma_q1['t1_50us_08'].to_numpy() *50 /t1_other
    gamma_decay_01_q0 = gamma_q0['t1_50us_01'].to_numpy() *50 /t1_other
    gamma_decay_01_q1 = gamma_q1['t1_50us_01'].to_numpy() *50 /t1_other
    gamma_dephase_50us_q0 = gamma_q0['tphi_50us'].to_numpy() *50 /t1_other
    gamma_dephase_50us_q1 = gamma_q1['tphi_50us'].to_numpy() *50 /t1_other
    gamma_dephase_1e6_q0 = gamma_q0['tphi_1e6'].to_numpy()
    gamma_dephase_1e6_q1 = gamma_q1['tphi_1e6'].to_numpy()

    gamma_decay_28_q0[2] = gamma_decay_logi
    gamma_decay_28_q1[2] = gamma_decay_logi
    gamma_decay_08_q0[2] = gamma_decay_logi
    gamma_decay_08_q1[2] = gamma_decay_logi
    gamma_decay_01_q0[2] = gamma_decay_logi
    gamma_decay_01_q1[2] = gamma_decay_logi
    gamma_dephase_50us_q0[2] = gamma_dephase_logi
    gamma_dephase_50us_q1[2] = gamma_dephase_logi
    gamma_dephase_1e6_q0[2] = gamma_dephase_logi
    gamma_dephase_1e6_q1[2] = gamma_dephase_logi

    jump_t1   = []
    jump_tphi = []
    if charge_pick:
        qubit_a = True
        arg_a = [dim_0, dim_1, gamma_decay_old, gamma_dephase_old, eket_tot, qubit_a] # old gamma
        # arg_a = [dim_0, dim_1, gamma_decay_new_q0, gamma_dephase_new_q0, eket_tot, qubit_a] # new gamma
        jump_op_a = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *arg_a) for state in range(1,dim_0))

        qubit_a = False
        arg_b = [dim_0, dim_1, gamma_decay_old, gamma_dephase_old, eket_tot, qubit_a] # old gamma
        # arg_b = [dim_0, dim_1, gamma_decay_new_q1, gamma_dephase_new_q1, eket_tot, qubit_a] # new gamma
        jump_op_b = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *arg_b) for state in range(1,dim_1))
        jump_t1_list = np.array(jump_op_a)[:,0].tolist() + np.array(jump_op_b)[:,0].tolist()
        jump_tphi_list = np.array(jump_op_a)[:,1].tolist() + np.array(jump_op_b)[:,1].tolist()
        jump_t1_list = [qt.Qobj(matrix) for matrix in jump_t1_list]
        jump_tphi_list = [qt.Qobj(matrix) for matrix in jump_tphi_list]
    else:
        args = [truc1, gamma_decay_old, gamma_dephase_old, eket_tot]
        jump_op = Parallel(n_jobs=100)(delayed(ut.get_jump_op)(state, *args) for state in range(1,truc1))
        jump_t1 = np.array(jump_op)[:,:2]
        jump_tphi = np.array(jump_op)[:,2:]
        jump_t1_list = [qt.Qobj(matrix) for row in jump_t1 for matrix in row]
        jump_tphi_list = [qt.Qobj(matrix) for row in jump_tphi for matrix in row]
    print("gamma_decay_logi = ", gamma_decay_logi, "gamma_dephase_logi = ", gamma_dephase_logi)
    print("gamma_decay_other = ", gamma_decay_other, "gamma_dephase_other = ", gamma_dephase_other)
    print(f"T1_logi = {1/gamma_decay_logi} ns") if gamma_decay_logi != 0 else None
    print(f"Tphi_logi = {1/gamma_dephase_logi} ns") if gamma_dephase_logi != 0 else None
    print(f"T1_other = {1/gamma_decay_other} ns") if gamma_decay_other != 0 else None
    print(f"Tphi_other = {1/gamma_dephase_other} ns") if gamma_dephase_other != 0 else None

    c_op_list = jump_t1_list + jump_tphi_list
    arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_select]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_optimize)(args_indep, *arg_select)
                                                for args_indep in x0_vec)
    print(f'\nf_noise (dim={len(hspace_select)},{charge_pick})  = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, np.round(f_noise[i:i+4], 8).tolist())), ',')
    print(']')

if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_True_False() # compare np.abs(true) vs (False)
    # import_False()
    import_select()


    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















