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


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_True_False() # compare np.abs(true) vs (False)
    import_False()


    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















