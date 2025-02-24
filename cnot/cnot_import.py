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

    # cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    # x0_vec = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()[[0, 5, 15, 25, 30],:]
    x0_vec = np.array([
[30.016865, 0.119046, 0.056799, -0.032063, -0.030732] ,
[40.016766, 0.104833, 0.049914, -0.030496, -0.028693] ,
[50.008813, 0.090273, 0.046546, -0.027498, -0.025767] ,
[60.024831, 0.074075, 0.039782, -0.020488, -0.019859] ,
[69.994995, 0.065867, 0.034625, -0.016253, -0.015833] ,
[80.005168, 0.061271, 0.032441, -0.015181, -0.014477] ,
[90.004882, 0.057133, 0.030909, -0.013363, -0.01325] ,
[100.006167, 0.053076, 0.029316, -0.01188, -0.01157] ,
[110.008706, 0.049616, 0.028206, -0.01119, -0.010777] ,
[119.999923, 0.046969, 0.027146, -0.010649, -0.010409] ,
[129.995414, 0.044102, 0.026819, -0.010756, -0.010313] ,
[140.005331, 0.041394, 0.025794, -0.010206, -0.009612] ,

[149.996866, 0.039078, 0.024359, -0.009107, -0.008681] ,
[160.000133, 0.036595, 0.022959, -0.008237, -0.007826] ,
[169.993992, 0.034622, 0.02211, -0.007681, -0.007266] ,
[179.996641, 0.032821, 0.02095, -0.006902, -0.006534] ,
[189.997344, 0.030915, 0.019755, -0.006194, -0.005848] ,
[200.001513, 0.029121, 0.018721, -0.005572, -0.005265] ,
[209.996351, 0.02774, 0.017771, -0.005024, -0.004777] ,
[219.997562, 0.026349, 0.016887, -0.00455, -0.004302] ,
[230.001715, 0.025067, 0.016064, -0.00409, -0.003877] ,
[239.989977, 0.023957, 0.015387, -0.003744, -0.003546] ,
[249.992868, 0.022922, 0.014688, -0.003423, -0.003252] ,
[260.000623, 0.021992, 0.014066, -0.003138, -0.002978] ,
[269.994868, 0.021106, 0.013544, -0.002882, -0.002747] ,
[280.000886, 0.020289, 0.013002, -0.002675, -0.002543] ,
[290.001675, 0.019575, 0.01252, -0.00247, -0.002353] ,
[300.003294, 0.018943, 0.012103, -0.002317, -0.002203] ,
[310.008159, 0.018298, 0.011686, -0.002157, -0.002061] ,
[319.999489, 0.01768, 0.011265, -0.002005, -0.001905] ,
[330.004173, 0.017153, 0.010944, -0.001891, -0.001804] ,
[340.0063, 0.016618, 0.010608, -0.001774, -0.00169] ,
    ])[[0, 5, 15, 25, 30],:]
    ### Get fidelity for input params (pick=True)
    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_full = 500

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

    num_cpus, n_job = 16, len(x0_vec)
    c_op_list = []
    H0_full = qt.Qobj(np.diag(eval_tot))

    hspace_select = [
    '0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
    '5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
    '13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
    '9-0', '20-2', '20-5', '12-5', '4-0', '15-5', '5-1', '18-2', '12-0', '24-2' ,
    '26-5', '1-5', '15-0', '13-0', '25-2', '35-2', '8-1', '33-2', '9-5', '4-5' ,

    '12-8', '9-8', '1-8', '37-2', '5-12', '25-0', '9-4', '1-25', '22-5', '56-2' ,
    '18-4', '13-9', '4-4', '2-4', '15-4', '50-2', '13-5', '9-9', '5-9', '4-1' ,
    # '20-0', '25-5', '15-9', '18-5', '9-1', '30-5', '2-21', '26-9', '1-13', '20-9' ,
    # '45-2', '18-0', '5-16', '22-0', '34-5', '15-1', '5-4', '0-21', '20-4', '26-0' ,
    # '25-4', '25-1', '0-4', '9-12', '24-5', '37-5', '24-0', '1-18', '12-9', '44-2' ,

    # '25-8', '2-12', '41-2', '4-16', '4-12', '4-21', '8-12', '18-9', '18-1', '45-0' ,
    # '24-9', '26-8', '39-2', '4-8', '8-4', '34-0', '22-8', '35-4', '12-4', '30-0' ,
    # '12-1', '28-2', '18-8', '0-8', '35-5', '35-0', '9-16', '37-0', '2-8', '1-9' ,
    # '13-8', '46-2', '2-26', '46-0', '2-9', '41-0', '4-13', '39-0', '2-25', '0-12' ,
    # '22-9', '33-0', '33-4', '22-4', '54-2', '8-18', '28-1', '0-45', '50-0', '24-1' ,

    # '28-5', '30-1', '15-8', '34-1', '22-1', '20-1', '13-16', '12-18', '0-9', '24-8' ,
    # '13-1', '51-2', '33-5', '2-30', '1-21', '4-18', '59-0', '15-12', '0-25', '8-24' ,
    # '8-8', '9-18', '44-0', '26-1', '65-0', '56-0', '41-1', '41-4', '5-30', '39-4' ,
    # '30-4', '58-0', '5-13', '18-12', '0-30', '45-4', '0-39', '13-4', '46-1', '28-0' ,
    # '2-18', '0-24', '0-13', '9-13', '30-8', '69-0', '41-5', '33-1', '37-8', '4-24' ,
    ]
    index_select = [hspace_full.index(i) for i in hspace_select]
    len_select = len(hspace_select)
    H0_select = ut.truncate_2( H0_full, index_select)
    n_theta0_select = ut.truncate_2(n_theta0_dress, index_select)
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
    print(f'\nhspace_truc (len={len_select}) = [')
    for i in range(0, len(hspace_select), 10):
        print(", ".join(f"'{x}'" for x in hspace_select[i:i + 10]), ',')
    print(']')
    arg_select = [H_drive_select, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_select]
    f_select = Parallel(n_jobs=n_job)(delayed(ut.cnot_fidelity_log_optimize)(args_indep, *arg_select)
                                                for args_indep in x0_vec)
    print(f' fidelity (dim={len(hspace_select)},{charge_pick}) =', np.round(f_select, 8).tolist())


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_True_False() # compare np.abs(true) vs (False)
    # import_False()
    import_select()


    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















