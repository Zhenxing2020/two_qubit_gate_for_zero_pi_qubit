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
    n_cpu_optimize = 1
    c_op_list = []
    args_truc = [H_drive_part, W_20_50, n_cpu_optimize, c_op_list, logi_idx_part]
    args_truc2 = [H_drive_part, W_20_50, n_cpu_optimize, c_op_list, logi_idx_part]
    fidelity = []
    drive_param = []
    fidelity_full = []
    # for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
    for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp_bound, detune_bound)
        print("\noptimize Time:", datetime.now(pytz.timezone('America/Denver')))
        # print('args_truc==args_truc2', args_truc==args_truc2)
        res = sp.optimize.differential_evolution(
            func=ut.cz_fidelity_log_optimize,
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
            # x0=x0_vec[jdx,:3],
            polish=False, # 'True' will make the for-loop break
            )
        print("\nfinish optimize Time:", datetime.now(pytz.timezone('America/Denver')))
        fidelity.append(res.fun)
        drive_param.append(res.x.tolist())

        print(res, '\n')
        print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        # print('\ntg = ')
        # for i in range(0, len(fidelity), 4):
            # print(', '.join(map(str, np.round(np.array((x0_vec[:,0])[:jdx+1])[i:i+4], 8))), ',')
        print(f'\nlog of gate error (truc={len_part}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')
        print(f'\ndrive_param (truc={len_part}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        print("Full Time:", datetime.now(pytz.timezone('America/Denver')))
        n_cpu_parallel = 16
        tg, drive_amp, detune = drive_param[jdx]

        arg_all = [tg, drive_amp, detune,
                   H_drive_False, W_20_50, n_cpu_parallel, c_op_list, logi_idx_False]
        fidelity_full.append(ut.cz_fidelity_log(arg_all))

        # arg_all = [tg, drive_amp, detune,
        #            n_cpu_parallel, hspace_False, W_20_50, H_drive_False, logi_idx_False]
        # fidelity_full.append(ut.cz_fidelity_log_old(arg_all))

        print(f'\nlog of gate error (truc={len(eval_tot)}) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')
        print('\namp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_full = 1000

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

    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick=False/'
    eval_False = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    hspace_False = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    n_theta0_False = qt.Qobj(2*np.pi* np.load(folder+'n_theta0_dress.npy'))
    n_theta1_False = qt.Qobj(2*np.pi* np.load(folder+'n_theta1_dress.npy'))

    # amp_bound, detune_bound, tg_bound = [(0.0085, 0.0106), (0.013, 0.0175), (-0.01, 0.01)] # tg141-157
    # amp_bound, detune_bound, tg_bound = [(0.008, 0.0092), (0.013, 0.0155), (-0.01, 0.01)] # tg160-171
    # amp_bound, detune_bound, tg_bound = [(0.007, 0.0088), (0.0128, 0.0142), (-0.01, 0.01)] # tg173-185
    # amp_bound, detune_bound, tg_bound = [(0.007, 0.008), (0.012, 0.0132), (-0.01, 0.01)] # tg186-195
    amp_bound, detune_bound, tg_bound = [(0.0068, 0.0075), (0.01175, 0.0125), (-0.01, 0.01)] # tg195-201

    # tg_vec = [141, 145, 149, 151, 153, 154, 157]
    # tg_vec = [160, 161, 162, 163, 165, 168, 170, 171,]
    # tg_vec = [173, 174, 177, 179, 180, 182, 184, 185]
    # tg_vec = np.arange(186, 195, 1)
    tg_vec = [2, 3] #np.arange(195, 201, 1)
    # cz = pd.read_csv('data/data_cz_3ncut_truc1=300.txt')
    # x0_vec = cz[['tg', 'drive_amp', 'detune']].to_numpy()[2::3,:]
    # x0_vec = np.array([
# [20.040941, 0.045968, 0.030859, -0.74640732],
#  [26.095068, 0.045891, 0.015649, -0.37350652],
#  [32.050555, 0.037113, 0.011292, -0.42044296],
#  [38.050827, 0.035562, 0.022828, -0.56293183],
#  [44.037972, 0.031932, 0.023962, -0.95698284],
#  [50.062652, 0.02695, 0.026935, -1.16069812],
#  [56.089236, 0.022853, 0.026795, -1.03691684],
#  [62.061714, 0.020772, 0.017392, -1.05329768],
#  [68.04014, 0.018304, 0.015964, -1.20314488],
#  [74.048199, 0.016257, 0.014902, -1.32245501],
    # ])


    print('gate_time_vector:', np.array(tg_vec).tolist()) if 'tg_vec' in globals() else None
    # for i in x0_vec:
    #     print(i.tolist(), ',')

    workers, popsize = 2, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    drive_term = n_theta1_dress
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    hspace_part = [
'0-0', '5-0', '0-2', '2-0', '2-2', '5-2', '5-1', '0-1', '2-1', '1-0' ,
# '0-5', '2-5', '1-2', '9-0', '4-0', '2-4', '5-5', '1-1', '5-4', '1-5' ,
# '9-2', '0-4', '4-2', '2-8', '0-8', '9-1', '2-12', '2-9', '0-9', '0-12' ,
# '12-0', '2-16', '0-16', '5-8', '1-8', '4-5', '2-13', '2-21', '0-18', '2-20' ,
# '9-4', '4-9', '8-0', '2-18', '0-21', '2-24', '1-4', '18-0', '5-26', '0-13' ,

# '13-0', '0-26', '15-0', '2-26', '1-12', '5-16', '8-1', '0-24', '15-1', '2-35' ,
# '15-4', '2-33', '2-45', '0-33', '2-39', '0-45', '5-12', '0-39', '5-9', '8-12' ,
# '5-33', '12-2', '4-4', '1-9', '4-1', '5-34', '2-30', '2-46', '1-16', '0-34' ,
# '2-34', '8-2', '5-21', '2-52', '0-52', '0-42', '2-42', '2-59', '0-59', '5-18' ,
# '0-65', '5-24', '2-55', '0-55', '0-20', '9-8', '8-9', '22-0', '1-25', '8-5' ,

# '12-1', '4-8', '2-53', '2-36', '2-25', '5-20', '5-13', '9-24', '15-8', '1-30' ,
# '0-25', '1-20', '5-25', '9-5', '1-13', '1-24', '13-2', '1-33', '18-1', '0-68' ,
# '18-2', '20-0', '2-44', '9-12', '4-25', '9-9', '5-30', '0-83', '5-39', '9-16' ,
# '0-36', '0-73', '12-4', '18-5', '22-2', '15-16', '1-21', '5-35', '9-13', '2-60' ,
# '2-57', '15-2', '15-5', '2-28', '13-1', '4-12', '0-35', '9-20', '2-54', '1-18' ,

# '0-81', '24-1', '4-39', '0-30', '1-26', '8-4', '4-16', '12-5', '1-35', '8-8' ,
# '24-0', '12-9', '5-44', '0-77', '4-21', '0-57', '5-28', '1-39', '25-0', '8-18' ,
# '1-28', '4-44', '33-0', '5-42', '12-12', '0-54', '13-12', '9-26', '4-24', '20-9' ,
# '8-26', '24-4', '8-16', '0-46', '13-4', '1-34', '37-1', '0-44', '18-16', '22-4' ,
# '1-45', '0-78', '9-25', '5-36', '24-9', '1-44', '4-35', '18-8', '0-53', '1-60' ,
    ]
    H0_full = qt.Qobj(np.diag(eval_tot))
    H0_False = qt.Qobj(np.diag(eval_False))
    logi_idx_False = [hspace_False.index(i) for i in logi_state]
    H_drive_False = [H0_False, [n_theta1_False, ut.drive_gauss_A] ]

    index_part = [hspace_full.index(i) for i in hspace_part]
    len_part = len(hspace_part)
    H0_part = ut.truncate_2( H0_full, index_part)
    n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
    H_drive_part = [H0_part, [n_theta1_part, ut.drive_gauss_A] ]
    logi_idx_part = [hspace_part.index(i) for i in logi_state]

    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_full)
    print('amp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
    print('W_20_50 = ', np.round(W_20_50, 3))
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














