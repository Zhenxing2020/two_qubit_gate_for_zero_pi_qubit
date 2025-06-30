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

    ### cz part
    cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    x0_vec = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()[[0],:]
    # [[-4],:]   [0::2,:] [[0,3,4, 16],:]

    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    drive_term = n_theta1_dress
    hspace_select = [
        ### state_all_1000
        ### charge_pick
'0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
'5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '0-9', '2-4', '9-0' ,
# '1-5', '5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '0-13', '0-16' ,
# '8-1', '4-4', '13-0', '15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5' ,
# '0-18', '2-9', '1-12', '0-20', '0-21',
# '18-0', '9-2', '12-1', '5-5', '0-24' ,

# '4-8', '1-13', '20-0', '8-4', '1-16',
# '22-0', '2-12', '13-1', '24-0', '0-25' ,
# '15-1', '0-26', '5-8', '4-9', '12-2', '2-13', '9-4', '2-16', '8-5', '25-0' ,
# '13-2', '26-0', '1-18', '15-2', '0-28', '5-9', '4-12', '28-0', '0-30', '18-1' ,
# '12-4', '9-5', '1-20', '1-21', '0-33',
# '8-8', '0-34', '0-35', '1-24', '2-18' ,
# '0-36', '20-1', '4-13', '22-1', '4-16',
# '30-0', '13-4', '5-12', '15-4', '24-1' ,
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
    H_drive_select = [ H0_select,   [drive_select, ut.drive_gauss_A] ]
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

    ### cz part
    # ### ideal fidelity
    # c_op_list = [qt.Qobj(np.zeros((len_select, len_select)))]
    c_op_list = []
    arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, logi_idx_select]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_optimize)(args_indep, *arg_select)
                                                for args_indep in x0_vec)
    print(f'\nf_ideal (dim={len(hspace_select)})  = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, np.round(f_ideal[i:i+4], 8).tolist())), ',')
    print(']')


    #################################################################
    ### Noisey fidelity
    t1_tphi_other = 170 # μs

    print("t1_tphi_other = ", t1_tphi_other)
    folder = f'../../data/3ncut_two_zeropi/truc1=500/'
    gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    gamma_decay_48_q0 = gamma_q0['t1_50us_48'].to_numpy() *50 /t1_tphi_other
    gamma_decay_48_q1 = gamma_q1['t1_50us_48'].to_numpy() *50 /t1_tphi_other
    gamma_dephase_02_q0 = gamma_q0['tphi_02'].to_numpy() *50 /t1_tphi_other
    gamma_dephase_02_q1 = gamma_q1['tphi_02'].to_numpy() *50 /t1_tphi_other

    qubit_a = True
    arg_a = [dim_0, dim_1, gamma_decay_48_q0, gamma_dephase_02_q0, eket_tot, qubit_a] # old gamma
    jump_op_a = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *arg_a) for state in range(1,dim_0))

    qubit_a = False
    arg_b = [dim_0, dim_1, gamma_decay_48_q1, gamma_dephase_02_q1, eket_tot, qubit_a] # old gamma
    jump_op_b = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *arg_b) for state in range(1,dim_1))
    jump_t1_list = np.array(jump_op_a)[:,0].tolist() + np.array(jump_op_b)[:,0].tolist()
    jump_tphi_list = np.array(jump_op_a)[:,1].tolist() + np.array(jump_op_b)[:,1].tolist()
    jump_t1_list = [qt.Qobj(matrix) for matrix in jump_t1_list]
    jump_tphi_list = [qt.Qobj(matrix) for matrix in jump_tphi_list]

    c_op_list = jump_t1_list + jump_tphi_list
    arg_select = [H_drive_select, W_20_50, num_cpus, c_op_list, logi_idx_select]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_optimize)(args_indep, *arg_select)
                                                for args_indep in x0_vec)
    print(f'\nf_noise (dim={len(hspace_select)})  = [')
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



def import_True_False():

    cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    x0_vec = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()[[0, 5, 10, 15, 20, 25, 30],:]

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

    num_cpus = None
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
[20.032421, 0.045974, 0.029778, -0.76245945, -0.75476894],

 [50.066456, 0.027173, 0.026765, -1.19196886, -1.15971719],

 [80.069609, 0.021699, 0.026059, -1.62299106, -1.61141391],

 [110.011766, 0.014013, 0.021038, -2.37358653, -2.38701254],

 [140.02276, 0.010556, 0.017, -3.25952568, -3.25427587],

 [170.005964, 0.008453, 0.014067, -3.9362217, -3.87857261],

 [199.99317, 0.007039, 0.011975, -3.95463104, -3.94867544]
    ])
    print('params =')
    for para in x0_vec:
        print(para.tolist(), ',')
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    num_cpus = 16
    n_job = 65
    c_op_list = []

    ### Get fidelity for input params (pick=False)
    truc1, truc_tot, charge_pick = 300, 2000, False
    truc_full = 2000
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')

    truc_list = np.arange(truc_full)
    hspace_full = hspace_full[:truc_full]
    eval_tot = eval_tot[:truc_full]
    n_theta1_dress = ut.truncate_2(n_theta1_dress, truc_list)
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    H0_full = qt.Qobj(np.diag(eval_tot))
    logi_idx_full = [hspace_full.index(i) for i in logi_state]
    H_drive_full = [H0_full, [n_theta1_dress, ut.drive_gauss_A] ]

    # len_part = 500
    # hspace_part = hspace_full[:len_part]
    # index_part = np.arange(len_part)
    # H0_part = ut.truncate_2( H0_full, index_part )
    # n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
    # logi_idx_part = [hspace_part.index(i) for i in logi_state]
    # H_drive_part = [H0_part, [n_theta1_part, ut.drive_gauss_A] ]

    # arg_part = [H_drive_part, W_20_50, num_cpus, c_op_list, logi_idx_part ]
    # f_part = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_part)
    #                                             for args_indep in x0_vec)
    # print(f' fidelity (dim={len_part},{charge_pick}) =')
    # for i in range(0, len(f_part), 4):
    #     print(', '.join(map(str, f_part[i:i+4])), ',')
    # print(']')
    # print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    arg_full = [H_drive_full, W_20_50, num_cpus, c_op_list, logi_idx_full ]
    f_full = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_full)
                                                for args_indep in x0_vec[:, :3])
    print(f' fidelity (dim={truc_full},{charge_pick}) =')
    for i in range(0, len(f_full), 4):
        print(', '.join(map(str, f_full[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

















