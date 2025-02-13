import sys
sys.path.append('../')

import scqubits as scq
import pandas as pd
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from qutip.qip.operations import rz, cz_gate
import cmath
from tqdm import tqdm
from matplotlib.colors import LogNorm
import datetime
import pytz
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
from joblib import Parallel, delayed
import itertools
import scipy.sparse as ssp
from sympy import symbols
import scipy as sp
import utils_2Q_gate_zp as ut
import os
from datetime import datetime
from multiprocessing import Pool


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    print('"new" is Tphi_logi  0 ---> 100μs')
    cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    params = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()[[11,12,13,14, 16,17,18,19, 21,22,23,24, 26,27,28,29,30],:]
    # [[1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19, 21,22,23,24, 26,27,28,29,30],:]
    #[[0, 5, 10, 15, 20, 25],:]  #[[0, 15, 30],:]

    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_tot_2 = 70
    num_cpus, n_job = 16, len(params)
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    hspace_0 = pd.read_csv(folder+ 'hspace_0.txt').to_numpy().flatten()
    hspace_1 = pd.read_csv(folder+ 'hspace_1.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    n_theta0_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
    n_theta1_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta1_dress.txt').to_numpy()
    eket_tot = pd.read_csv(folder+ 'eket_tot.txt').map(complex).to_numpy()

    dim_0 = len(hspace_0)
    dim_1 = len(hspace_1)
    eval_tot = eval_tot[:truc_tot_2]
    eket_tot = eket_tot[:truc_tot_2]
    hspace_full = hspace_full[:truc_tot_2]
    hspace_dress = np.arange(truc_tot_2)
    n_theta0_dress = qt.Qobj(n_theta0_dress[np.ix_(hspace_dress, hspace_dress)])
    n_theta1_dress = qt.Qobj(n_theta1_dress[np.ix_(hspace_dress, hspace_dress)])
    print('truc1=', truc1, '; truc_tot = ', truc_tot, '; charge_pick = ', charge_pick)
    print("truc_tot_2 = ", truc_tot_2)
    print("num_cpus = ", num_cpus, ";   n_job = ", n_job)
    for para in params:
        print(para.tolist(), ',')

    gamma_decay_logi =  1 / 1600e3
    gamma_dephase_logi = 1 / 100e3
    gamma_decay_other = 1 / 50e3
    gamma_dephase_other = 1 / 50e3
    jump_t1   = []
    jump_tphi = []
    if charge_pick:
        gamma_decay   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (dim_1-3)
        gamma_dephase = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (dim_1-3)
        qubit_a = True
        args = [dim_0, dim_1, gamma_decay, gamma_dephase, eket_tot, qubit_a]
        jump_op_a = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *args) for state in range(1,dim_0))
        qubit_a = False
        args = [dim_0, dim_1, gamma_decay, gamma_dephase, eket_tot, qubit_a]
        jump_op_b = Parallel(n_jobs=100)(delayed(ut.get_jump_op_charge_pick)(state, *args) for state in range(1,dim_1))
        jump_t1_list = np.array(jump_op_a)[:,0].tolist() + np.array(jump_op_b)[:,0].tolist()
        jump_tphi_list = np.array(jump_op_a)[:,1].tolist() + np.array(jump_op_b)[:,1].tolist()
        jump_t1_list = [qt.Qobj(matrix) for matrix in jump_t1_list]
        jump_tphi_list = [qt.Qobj(matrix) for matrix in jump_tphi_list]
    else:
        gamma_decay   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (truc1-3)
        gamma_dephase = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (truc1-3)
        args = [truc1, gamma_decay, gamma_dephase, eket_tot]
        jump_op = Parallel(n_jobs=100)(delayed(ut.get_jump_op)(state, *args) for state in range(1,truc1))
        jump_t1 = np.array(jump_op)[:,:2]
        jump_tphi = np.array(jump_op)[:,2:]
        jump_t1_list = [qt.Qobj(matrix) for row in jump_t1 for matrix in row]
        jump_tphi_list = [qt.Qobj(matrix) for row in jump_tphi for matrix in row]
    print("gamma_decay_logi = ", gamma_decay_logi, "gamma_dephase_logi = ", gamma_dephase_logi)
    print("gamma_decay_other = ", gamma_decay_other, "gamma_dephase_other = ", gamma_dephase_other)
    print(f"T1_logi = {1/gamma_decay_logi} ns") if gamma_decay_logi != 0 else None
    print(f"T1_other = {1/gamma_decay_other} ns") if gamma_decay_other != 0 else None
    print(f"Tphi_logi = {1/gamma_dephase_logi} ns") if gamma_dephase_logi != 0 else None
    print(f"Tphi_other = {1/gamma_dephase_other} ns") if gamma_dephase_other != 0 else None

    logi_state = ['0-0', '0-2', '2-0', '2-2']
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    H0 = qt.Qobj(np.diag(eval_tot))
    logi_idx = [hspace_full.index(state) for state in logi_state]
    H_qbt_drive = [H0, [n_theta1_dress, ut.drive_gauss_A] ]
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # c_op_list = [qt.Qobj(np.zeros((truc_tot_2, truc_tot_2)))]
    c_op_list = []
    args = [H_qbt_drive, W_20_50, num_cpus, c_op_list, logi_idx ]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_ideal = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, f_ideal[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    c_op_list = jump_t1_list + jump_tphi_list
    args = [H_qbt_drive, W_20_50, num_cpus, c_op_list, logi_idx ]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_noise = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, f_noise[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




