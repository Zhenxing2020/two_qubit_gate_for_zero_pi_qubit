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
import pytz


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
# def f_noise_sweep():


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    truc1, truc_tot, charge_pick = 300, 100, False
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}_eket/'
    eval_tot = pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_0 = pd.read_csv(folder+ 'hspace_0.txt').to_numpy().flatten()
    hspace_1 = pd.read_csv(folder+ 'hspace_1.txt').to_numpy().flatten()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten()
    n_theta0_dress = pd.read_csv(folder+ 'n_theta0_dress.txt').map(complex).to_numpy()
    n_theta1_dress = pd.read_csv(folder+ 'n_theta1_dress.txt').map(complex).to_numpy()
    eket_tot = pd.read_csv(folder+ 'eket_tot.txt').map(complex).to_numpy()

    cz300_se_3ncut= pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    n_cpu = 20
    n_job = 3
    max_steps = 1e-4

    gamma_decay_logi =  1 / 1600e3
    gamma_dephase_logi = 0
    gamma_decay_other = 1 / 2e3
    gamma_dephase_other = 1 / 400
















    args_all = ut.get_operator_two_zeropi()
    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot  ] = args_all
    idxs = [order_sort.index(i) for i in trunc_states]

    trunc_dim = eket1.shape[0]
    truc = len(trunc_states)

    eket_truc = [eket_tot[i] for i in idxs]
    eket_truc = np.reshape(eket_truc, (truc, trunc_dim**2))
    jump_t1   = []
    jump_tphi = []
    gamma_t1   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (trunc_dim-3)
    gamma_tphi = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (trunc_dim-3)
    for i in range(1,trunc_dim):
        ladder_0i = qt.basis(trunc_dim,0) * qt.basis(trunc_dim,i).dag()
        a_0i_I = qt.tensor(ladder_0i, qt.qeye(trunc_dim))
        a_I_0i = qt.tensor(qt.qeye(trunc_dim), ladder_0i)
        jump_t1.append(qt.Qobj( eket_truc @ ( np.sqrt(gamma_t1[i])* a_0i_I ).data @ eket_truc.conj().T ))
        jump_t1.append(qt.Qobj( eket_truc @ ( np.sqrt(gamma_t1[i])* a_I_0i ).data @ eket_truc.conj().T ))

        # t_phi
        proj_ii = qt.basis(trunc_dim,i).proj()
        a_ii_I = qt.tensor(proj_ii, qt.qeye(trunc_dim))
        a_I_ii = qt.tensor(qt.qeye(trunc_dim), proj_ii)
        jump_tphi.append(qt.Qobj( eket_truc @ ( np.sqrt(2*gamma_tphi[i])* a_ii_I  ).data @ eket_truc.conj().T) )
        jump_tphi.append(qt.Qobj( eket_truc @ ( np.sqrt(2*gamma_tphi[i])* a_I_ii  ).data @ eket_truc.conj().T) )

    state_tot = [qt.basis(truc, i) for i in range(truc)]
    W_20_50 = np.abs(ut.transition_frequency(order_sort.index('20') , order_sort.index('50'), eval_tot))

    n1_22 = n_theta1_truc[trunc_states.index('22'), trunc_states.index('52')]
    n2_22 = n_theta2_truc[trunc_states.index('22'), trunc_states.index('52')]
    n1_20 = n_theta1_truc[trunc_states.index('20'), trunc_states.index('50')]
    n2_20 = n_theta2_truc[trunc_states.index('20'), trunc_states.index('50')]
    eta = - n2_22 / n1_22

    drive_ab = True


    # params = pd.read_csv('data/data_cz_n2_20.txt').to_numpy()
    # param2 = []
    # for i in range(20):
    #     param2.append(params[8*i+2].tolist())
    # params = np.array(param2)
    params = pd.read_csv('data/data_cz_n1n2_20_dark.txt').to_numpy()
    param2 = []
    for i in range(20):
        param2.append(params[4*i+2].tolist())
    params = np.array(param2)


    print("n_cpu = ", n_cpu, ";   n_job = ", n_job)
    print("gamma_2 = ", gamma_decay_logi, "gamma_p2 = ", gamma_dephase_logi,
          ";  gamma_3 / gamma_2 =", ratio,";   max_steps = ", max_steps)
    if gamma_decay_logi != 0:
        print(f"T1_2 = {1/gamma_decay_logi} ns")
    if gamma_dephase_logi != 0:
        print(f"Tphi_2 = {1/gamma_dephase_logi} ns")
    for para in params:
        print(para.tolist(), ',')

    c_op_list = []
    args = [H0, n_theta1_truc, n_theta2_truc, eta, W_20_50, max_steps, n_cpu, c_op_list, drive_ab ]
    f_ideal = Parallel(n_jobs=n_job, verbose=10)(delayed(ut.get_fidelity_noise_cz)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_ideal = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, f_ideal[i:i+4])), ',')
    print(']')


    c_op_list = jump_t1 + jump_tphi
    args = [H0, n_theta1_truc, n_theta2_truc, eta, W_20_50, max_steps, n_cpu, c_op_list, drive_ab ]
    f_noise = Parallel(n_jobs=n_job, verbose=10)(delayed(ut.get_fidelity_noise_cz)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_noise = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, f_noise[i:i+4])), ',')
    print(']')


    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




    # eket_truc = np.reshape([eket_tot[i] for i in idxs], (truc, eket1.shape[0]**2))
    # jump_op_t1 = np.zeros((truc, truc), dtype=complex)
    # jump_op_tphi = np.zeros((truc, truc), dtype=complex)
    # for state in trunc_states[1:]:
    #     a0i = eket0[0].conj().T @ eket0[int(state[0])]
    #     a0i = eket0 @ a0i @ eket0.conj().T
    #     b0j = eket1[0].conj().T @ eket1[int(state[1])]
    #     b0j = eket1 @ b0j @ eket1.conj().T
    #     jump_op_t1 += (eket_truc @ np.kron(a0i, b0j) @ eket_truc.conj().T)
    #     aii = eket0[int(state[0])].conj().T @ eket0[int(state[0])]
    #     aii = eket0 @ aii @ eket0.conj().T
    #     bjj = eket1[int(state[1])].conj().T @ eket1[int(state[1])]
    #     bjj = eket1 @ bjj @ eket1.conj().T
    #     eket_truc = np.reshape([eket_tot[i] for i in idxs], (truc, eket1.shape[0]**2))
    #     jump_op_tphi += (eket_truc @ np.kron(aii, bjj) @ eket_truc.conj().T)
    # jump_op_t1 = qt.Qobj(jump_op_t1) * np.sqrt(gamma_2)
    # jump_op_tphi = qt.Qobj(jump_op_tphi) * np.sqrt(2*gamma_p2)


    # jump_t1 = np.zeros((truc, truc), dtype=complex)
    # jump_tphi = np.zeros((truc, truc), dtype=complex)
    # for i in range(1,10):
    #     ladder_0i = qt.basis(trunc_dim,0) * qt.basis(trunc_dim,i).dag()
    #     a_0i_I = qt.tensor(ladder_0i, qt.qeye(trunc_dim))
    #     a_I_0i = qt.tensor(qt.qeye(trunc_dim), ladder_0i)
    #     jump_t1_i = (eket_truc @ ( a_0i_I + a_I_0i ).data @ eket_truc.conj().T)
    #     proj_ii = qt.basis(trunc_dim,i).proj()
    #     a_ii_I = qt.tensor(proj_ii, qt.qeye(trunc_dim))
    #     a_I_ii = qt.tensor(qt.qeye(trunc_dim), proj_ii)
    #     jump_tphi_i = (eket_truc @ ( a_ii_I + a_I_ii ).data @ eket_truc.conj().T)
    #     if i == 1:
    #         jump_t1 += np.sqrt(gamma_2)* jump_t1_i
    #         jump_tphi += np.sqrt(2*gamma_p2) * jump_tphi_i
    #     else:
    #         jump_t1 += np.sqrt(gamma_3)* jump_t1_i
    #         jump_tphi += np.sqrt(2*gamma_p3) * jump_tphi_i
    # jump_t1 = qt.Qobj(jump_t1)
    # jump_tphi = qt.Qobj(jump_tphi)