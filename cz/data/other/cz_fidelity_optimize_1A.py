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
## Calculate fidelity for one set of parameters
###################################################################
def calculate_fidelity():
    detune=0.094
    drive_amp=0.0089
    args_indep = (detune, drive_amp)
    fidelity = ut.get_fidelity_2(args_indep, *args)

    print('Ec0:', Ec0)
    print('gate time:', tg,'(ns)')
    print('detune: ', detune)
    print('drive amp: ', drive_amp)
    print('fidelity=',fidelity)

###################################################################
## Calculate fidelity by sweeping A0, detune for single Jc & tg
###################################################################
def calculate_fidelity_sweep():

    n_jobs = 1
    Ec0_vec = np.linspace(1, 2.0, num=1)
    # tg_vec = np.linspace(20, 200, num=10)
    detune_vec = np.linspace(0.094, 0.1, num=1)
    drive_amp_vec = np.linspace(0.0089, 0.2, num=1)

    fidelity_best = []
    drive_amp_best = []
    detune_best = []
    fidelity_all = []
    # for tg in tg_vec:
    for Ec0 in tqdm(Ec0_vec):
        values_grid = list(itertools.product(detune_vec, drive_amp_vec))
        data = Parallel(n_jobs=n_jobs, verbose=1)(delayed(ut.get_fidelity)(args_indep, *args)
                                                    for args_indep in values_grid)

        fidelity = np.reshape(data, (len(drive_amp_vec), len(detune_vec)))
        (drive_amp_idx, detune_idx) = np.unravel_index(np.argmax(fidelity, axis=None), np.shape(fidelity))

        fidelity_best.append(np.max(fidelity))
        drive_amp_best.append(drive_amp_vec[drive_amp_idx])
        detune_best.append(detune_vec[detune_idx])
        fidelity_all.append(fidelity)

    print('detune: ', detune_vec)
    print('A0: ', drive_amp_vec)
    print('Ec0: ', Ec0_vec)
    print('tg: ', tg)
    print('fidelity: ',fidelity_best)


###################################################################
## Optimize fidelity with differential evolution
###################################################################
def calculate_fidelity_de():
    detune_bounds = (-0.05, 0.)
    amp_bounds = (0, 0.03)
    args_indep = (detune_bounds, amp_bounds)

    popsize = 30
    mutation = (0, 1.99)
    recombination = 0.7
    tol = 0.01
    print('gate_time:', tg)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    print('detune_bounds=',detune_bounds)
    print('amp_bounds=',amp_bounds)

    res = sp.optimize.differential_evolution(ut.get_fidelity_2,
                                            args_indep,
                                            args=args,
                                            disp=True,
                                            callback=ut.print_soln,
                                            workers=100,
                                            init="sobol",
                                            popsize=popsize,
                                            mutation=mutation,
                                            recombination=recombination,
                                            tol=tol,
                                            polish=False, # 'True' will make the for-loop break
                                            )
    print(res)






###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_sweep_x0():

    x0_vec = np.array([
[0.0, 0.001, 120.0, -3.656008],
[0.0, 0.001, 128.0, -4.0020057],
[0.0, 0.001, 136.0, -3.75665887],
[0.0, 0.001, 144.0, -4.1266732],
[0.0, 0.001, 160.0, -4.45320218]
        ])
    print('x0_vec=')
    for i in x0_vec:
        print(i.tolist(),',')
    Fidelity = {}
    Drive_param = {}
    repets = 5
    Drive_param[0] = x0_vec

    for idx in range(repets):
        print('\n\n\n***&&&***repeats=%d\n\n\n'%idx)
        fidelity = []
        drive_param = []
        for jdx, tg in tqdm(enumerate(x0_vec[:,2])):
            x0 = Drive_param[idx][jdx][:2]
            args = [H_qbt_drive, W_20_50, tg, state_tot]
            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cz,
                bounds=args_indep,
                args=args,
                disp=True,
                callback=ut.print_soln,
                init="sobol",
                workers=workers,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                tol=tol,
                polish=False, # 'True' will make the for-loop break
                x0=x0
                )
            fidelity.append(res.fun)
            drive_param.append(res.x.tolist())
            print('\ngate time=', tg)
            print(res, '\n')
            print('\nfidelity = ')
            for i in range(0, len(fidelity), 4):
                print(', '.join(map(str, fidelity[i:i+4])), ',')
            print('\ndrive_param = ')
            for i in drive_param:
                print(i,',')

        Fidelity[idx+1] = fidelity
        Drive_param[idx+1] = drive_param
        for i in range(idx):
            print(f'Fidelity_{i} = ')
            for j in range(0, len(Fidelity[i+1]), 4):
                print(', '.join(map(str, Fidelity[i+1][j:j+4])), ',')
            print(f'Drive_param_{i} = ')
            for j in Drive_param[i+1]:
                    print(j,',')


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_sweep():
    fidelity = []
    drive_param = []
    for jdx, tg in tqdm(enumerate(tg_vec)):
        args = [H_qbt_drive, W_target, tg, state_tot]
        res = sp.optimize.differential_evolution(
            func=ut.get_fidelity_cz,
            bounds=args_indep,
            args=args,
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
        print('\ngate time=', tg)
        print(res, '\n')
        print('\nfidelity = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, fidelity[i:i+4])), ',')
        print('\ndrive_param = ')
        for i in drive_param:
            print(i,',')




if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    args_all = ut.get_operator_two_zeropi()
    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot  ] = args_all
    truc = len(trunc_states)
    state_tot = [qt.basis(truc, i) for i in range(truc)]
    W_20_50 = np.abs(ut.transition_frequency(order_sort.index('20') , order_sort.index('50'), eval_tot))
    W_22_52 = np.abs(ut.transition_frequency(order_sort.index('22') , order_sort.index('52'), eval_tot))
    W_00_01 = np.abs(ut.transition_frequency(order_sort.index('00') , order_sort.index('01'), eval_tot))
    W_target = W_00_01
    print('W_20_50 = ', W_20_50)
    print('W_22_52 = ', W_22_52)
    print('W_00_01 = ', W_00_01)
    print('W_target = ', W_target)

    drive_ab = False
    drive_term = n_theta1_truc + n_theta2_truc  if drive_ab else n_theta2_truc
    H_qbt_drive = [H0, [2*np.pi* drive_term, ut.drive_gauss_A] ]

    detune_bounds = (-0.2, 0.1)
    amp_bounds = (0, 0.1)
    args_indep = (detune_bounds, amp_bounds)
    tg_vec = np.linspace(40, 56, num=3)
    workers = 60
    popsize = 40
    mutation = (0.5, 1.99)
    recombination = 0.7
    tol = 0.01
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
    print('\ndetune_bounds=',detune_bounds)
    print('amp_bounds=',amp_bounds)
    print('gate_time_vector:\n', np.array(tg_vec).tolist())
    print('workers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)

    fidelity_sweep()

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('workers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)


    print("End Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    # drive_ab = True
    # fidelity_best_ab = []
    # drive_amp_best_ab = []
    # detune_best_ab = []
    # fidelity_all_ab = []
    # for idx,Ec0 in tqdm(enumerate(Ec0_vec)):
    # # for tg in tg_vec:
    #     values_grid = list(itertools.product(detune_vec, drive_amp_vec, [Ec0], [tg], [drive_ab]))
    #     data_grid = Parallel(n_jobs=n_jobs,verbose=1)(delayed(fidelity_sweep)(*value)
    #                                                 for value in values_grid)
    #     np.savez('para_run_fn_name1.npz', x1 = detune_vec,
    #             x2 = drive_amp_vec, vals = values_grid, fn_vals = data_grid)
    #     data_x_y = np.load('para_run_fn_name1.npz','r')
    #     delts = data_x_y['x1']
    #     gams = data_x_y['x2']
    #     vals = data_x_y['vals']
    #     fidelity = data_x_y['fn_vals']

    #     fidelity = np.reshape(fidelity, (len(drive_amp_vec), len(detune_vec)))
    #     (drive_amp_idx, detune_idx) = np.unravel_index(np.argmax(fidelity, axis=None), np.shape(fidelity))

    #     fidelity_best_ab.append(np.max(fidelity))
    #     drive_amp_best_ab.append(drive_amp_vec[drive_amp_idx])
    #     detune_best_ab.append(detune_vec[detune_idx])
    #     fidelity_all_ab.append(fidelity)


    # fig, ax = plt.subplots(figsize=(4,4))
    # plt.plot(Ec0_vec, 1-np.array(fidelity_best), linestyle='--', marker='.', color='magenta', label='$22\\to 52,B$')
    # plt.plot(Ec0_vec, 1-np.array(fidelity_best_ab), marker='.', color='sienna', label='$22\\to 52,A+B$')
    # plt.xlabel('$Ec0$(GHz)')
    # plt.ylabel('$1-F$')
    # plt.title(f'$0-\pi$ (tg=100)\n detune=[{detune_vec[0]}, {detune_vec[-1]}, {len(detune_vec)}], '+
    #           f'A0=[{drive_amp_vec[0]}, {drive_amp_vec[-1]}, {len(drive_amp_vec)}]')
    # plt.yscale('log')
    # plt.legend()
    # plt.grid()

    # ####################################################################
    # ### save to .csv and .png
    # Dir = 'data/zp_fidelity/'

    # current_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    # formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # plt.savefig(Dir+f'Ec0_sweep_enum_dim=20_tg={tg}_detune=[{detune_vec[0]},{detune_vec[-1]}]'+
    #            f'_A0=[{drive_amp_vec[0]}, {drive_amp_vec[-1]}]'+
    #            f'_{formatted_time}.png',
    #            dpi=150,bbox_inches='tight')


    # fidelity_all = np.array(fidelity_all)
    # array_2d = fidelity_all.reshape(-1, fidelity_all.shape[-1])
    # np.savetxt(Dir+f'Ec0_sweep_enum_dim=20_tg={tg}_detune=[{detune_vec[0]},{detune_vec[-1]}]'+
    #            f'_A0=[{drive_amp_vec[0]}, {drive_amp_vec[-1]}]'+
    #            f'_{formatted_time}_B.txt',
    #            array_2d, delimiter=',')

    # fidelity_all_ab = np.array(fidelity_all_ab)
    # array_2d = fidelity_all_ab.reshape(-1, fidelity_all_ab.shape[-1])
    # np.savetxt(Dir+f'Ec0_sweep_enum_dim=20_tg={tg}_detune=[{detune_vec[0]},{detune_vec[-1]}]'+
    #            f'_A0=[{drive_amp_vec[0]}, {drive_amp_vec[-1]}]'+
    #            f'_{formatted_time}_AB.txt',
    #            array_2d, delimiter=',')















