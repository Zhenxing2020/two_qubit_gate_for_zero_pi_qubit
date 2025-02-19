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
[20.043476, 0.045978, 0.03098] ,

[50.062652, 0.02695, 0.026935] ,

[80.084293, 0.021844, 0.026133] ,

[110.017394, 0.014028, 0.020992] ,

[140.019398, 0.010571, 0.017004] ,

[170.021986, 0.009896, 0.02066] ,

[199.994605, 0.007046, 0.01197] , # optimize_200_False_full
    ])
    print('params =')
    for para in x0_vec:
        print(para.tolist(), ',')
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    num_cpus = None
    n_job = 100
    c_op_list = []






    ### Get fidelity for input params (pick=False)
    truc1, truc_tot, charge_pick = 300, 1000, False
    truc_full = 1000
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')


    idx_0 = hspace_full.index('0-2')
    idx_1 = hspace_full.index('2-2')
    idx_2 = hspace_full.index('8-2')
    W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]


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
    print(f' fidelity (dim={len_part},{charge_pick}) =')
    for i in range(0, len(f_part), 4):
        print(', '.join(map(str, f_part[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    arg_full = [H_drive_full, W_20_50, num_cpus, c_op_list, logi_idx_full ]
    f_full = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *arg_full)
                                                for args_indep in x0_vec)
    print(f' fidelity (dim={truc_full},{charge_pick}) =')
    for i in range(0, len(f_full), 4):
        print(', '.join(map(str, f_full[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_True_False() # compare np.abs(true) vs (False)
    import_False()


    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















