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
from datetime import datetime
import pytz
import os

###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_sweep():
    n_cpu = 1
    args_truc = [n_cpu, hspace_truc, W_20_50, H_drive_truc, logic_idx_truc]
    fidelity = []
    drive_param = []
    fidelity_full = []
    # for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
    for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp_bound, detune_bound)
        res = sp.optimize.differential_evolution(
            func=ut.cz_fidelity_optimize,
            bounds=bounds,
            args=args_truc,
            disp=True,
            callback=ut.print_soln,
            init="sobol",
            workers=workers,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x.tolist())

        print(res, '\n')
        print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())

        print(f'\nlog of gate error (truc={truc_len}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

        print(f'\ndrive_param (truc={truc_len}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        n_cpu_full = 50
        tg, drive_amp, detune = drive_param[jdx]
        arg_all = [tg, drive_amp, detune, n_cpu_full, hspace_full, W_20_50, H_drive_full, logic_idx_full]
        fidelity_full.append(ut.cz_fidelity(arg_all))
        print(f'\nlog of gate error (truc={len(eval_tot)}) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')
        print('\namp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    truc1, truc_tot, charge_pick = 300, 2000, True
    truc_tot_2 = 700

    folder = f'../two_qubit_data_truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    n_theta0_dress = pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
    n_theta1_dress = pd.read_csv(folder+ 'n_theta1_dress.txt').to_numpy()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    truc_list = np.arange(truc_tot_2)
    hspace_full = hspace_full[:truc_tot_2]
    eval_tot = eval_tot[:truc_tot_2]
    n_theta0_dress = ut.truncate_2(n_theta0_dress, truc_list)
    n_theta1_dress = ut.truncate_2(n_theta1_dress, truc_list)

    # amp_bound, detune_bound, tg_bound = [(0.008, 0.012), (0.012, 0.018), (0, 0.01)]
    # amp_bound, detune_bound, tg_bound = [(0.007, 0.00875), (0.012, 0.015), (0, 0.01)]
    amp_bound, detune_bound, tg_bound = [(0.044, 0.05), (0, 0.025), (0, 0.01)]

    tg_vec = np.arange(27, 38, 1)
    workers, popsize = 100, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    drive_ab = False

    logic_states = ['0-0', '0-2', '2-0', '2-2']
    # args_all = ut.get_operator_two_zeropi_v2(truc1=truc1, truc_tot=truc_tot, charge_pick=charge_pick)
    # [hspace_0, hspace_1, top3_index, top3_overlap,
    # n_theta0_dress, n_theta1_dress, eval_tot, hspace_full] = args_all
    # W_20_50 = 2*np.pi* ( eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')] )
    W_20_50 = 2*np.pi* ( eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')] )

    drive_term = n_theta0_dress + n_theta1_dress  if drive_ab else n_theta1_dress
    hspace_truc = ['0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
'5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '0-9', '2-4', '9-0' ,
'1-5', '5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '0-13', '0-16' ,
'8-1', '4-4', '13-0', '15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5' ,
'0-18', '2-9', '1-12', '0-20', '0-21', '18-0', '9-2', '12-1', '5-5', '0-24' ,
'4-8', '1-13', '20-0', '8-4', '1-16', '22-0', '2-12', '13-1', '0-25', '15-1' ,
'0-26', '5-8', '4-9', '12-2', '2-13', '9-4', '2-16', '8-5', '25-0', '13-2' ,
'1-18', '15-2', '5-9', '4-12', '0-30', '18-1', '12-4', '9-5', '1-20', '1-21' ,
'0-33', '8-8', '0-34', '0-35', '1-24', '2-18', '0-36', '5-12', '15-4', '24-1' ,
'2-20', '18-2', '2-21', '8-9', '1-25', '9-8', '1-26', '0-39', '12-5', '2-24' ,
'5-13', '5-16', '22-2', '0-42', '24-2', '15-5', '2-25', '2-26', '8-12', '9-9' ,
'18-4', '0-44', '12-8', '1-30', '0-45', '0-46', '1-33', '8-16', '22-4', '4-24' ,
'5-18', '1-34', '15-8', '2-28', '9-12', '24-4', '12-9', '33-1', '18-5', '2-30' ,
'0-52', '0-53', '5-20', '4-25', '5-21', '2-33', '13-9', '9-13', '1-39', '37-1' ,
'5-24', '9-16', '2-34', '2-35', '0-54', '2-36', '0-55', '0-57', '5-25', '5-26' ,
'0-59', '2-39', '1-45', '2-42', '0-65', '2-44', '15-16', '5-30', '2-45', '9-24' ,
'2-46', '5-33', '0-68', '5-34', '5-35', '0-73', '2-52', '2-53', '9-26', '0-78' ,
'5-39', '18-16', '4-44', '2-54', '2-55', '2-57', '5-42', '2-59', '0-83', '2-60' ,
'5-44', '8-39', '5-45', '9-34', '9-35', '5-46', '2-64', '2-65', '5-52', '5-53' ,
'2-66', '2-68', '5-55', '2-73', '2-76', '5-59', '2-78', '5-65', '2-83', '5-68'
    ]
    truc_index = [hspace_full.index(i) for i in hspace_truc]
    truc_len = len(hspace_truc)

    H0_full = 2*np.pi* qt.Qobj(np.diag(eval_tot))
    H_drive_full = [H0_full, [2*np.pi* qt.Qobj(drive_term), ut.drive_gauss_A] ]
    logic_idx_full = [hspace_full.index(i) for i in logic_states]

    H0_truc = ut.truncate_2( H0_full, truc_index)
    drive_truc = ut.truncate_2(drive_term, truc_index)
    H_drive_truc = [H0_truc, [2*np.pi* drive_truc, ut.drive_gauss_A] ]
    logic_idx_truc = [hspace_truc.index(i) for i in logic_states]


    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_tot_2)
    print('amp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
    print('W_20_50 = ', np.round(W_20_50, 3), ', drive_ab=', drive_ab)
    print('gate_time_vector:', np.array(tg_vec).tolist())
    print('workers=', workers, ', popsize=', popsize)
    print('recombination=', recombination, ', tol=', tol, ', mutation=', mutation)

    fidelity_sweep()

    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)





###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
# def fidelity_sweep_x0():
#     cz = pd.read_csv('data/data_cz_fidelity_sesolve.txt')
#     # cz = pd.read_csv('data/data_cz_fidelity_sesolve_select.txt')
#     n_cpu = 1
#     args_truc = [n_cpu, hspace_truc, W_20_50, H_drive_truc, logic_idx_truc]
#     fidelity = []
#     drive_param = []
#     fidelity_full = []
#     # for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
#     for jdx, tg in tqdm(enumerate(tg_vec)):
#         tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
#         x0 = cz[['tg','drive_amp','detune']].iloc[jdx].to_numpy()
#         bounds = (tg_bounds, amp_bound, detune_bound)
#         res = sp.optimize.differential_evolution(
#             func=ut.cz_fidelity_optimize,
#             bounds=bounds,
#             args=args_truc,
#             disp=True,
#             callback=ut.print_soln,
#             init="sobol",
#             workers=workers,
#             popsize=popsize,
#             mutation=mutation,
#             recombination=recombination,
#             tol=tol,
#             x0=x0,
#             polish=False, # 'True' will make the for-loop break
#             )
#         fidelity.append(res.fun)
#         drive_param.append(res.x.tolist())

#         print(res, '\n')
#         print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())

#         print(f'\nlog of gate error (truc={truc_len}) = ')
#         for i in range(0, len(fidelity), 4):
#             print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

#         print(f'\ndrive_param (truc={truc_len}) = ')
#         for i in drive_param:
#             print(np.round(i,6).tolist(),',')

#         n_cpu_full = 50
#         tg, drive_amp, detune = drive_param[jdx]
#         arg_all = [tg, drive_amp, detune, n_cpu_full, hspace_full, W_20_50, H_drive_full, logic_idx_full]
#         fidelity_full.append(ut.cz_fidelity(arg_all))
#         print(f'\nlog of gate error (truc={len(eval_tot)}) = ')
#         for i in range(0, len(fidelity_full), 4):
#             print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')
#         print('\namp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
#         print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))














