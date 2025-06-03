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
def fidelity_sweep():
    x0_vec = np.array([
# [-3.044191800e-02,  2.07666100e-02,  1.20000000e+02,        -3.65600800e+00],
# [-3.016019900e-02,  1.98881600e-02,  1.28000000e+02,        -4.00200570e+00],
# [-6.015065900e-02,  1.71630100e-02,  1.36000000e+02,        -3.75665887e+00],
# [-7.090207400e-02,  1.66943400e-02,  1.44000000e+02,        -4.12667320e+00],
# [-9.32866700e-02,  1.58500900e-02,  1.52000000e+02,        -4.93692540e+00],
# [-3.054622800e-02,  1.54891200e-02,  1.60000000e+02,        -4.45320218e+00],
[-6.049735300e-02,  1.49408800e-02,  1.68000000e+02,        -4.37817473e+00],
# [-6.16776500e-02,  1.44568900e-02,  1.76000000e+02,        -4.24686447e+00],
# [-5.88977900e-02,  1.40144800e-02,  1.84000000e+02,        -4.42871367e+00],
# [-5.61922900e-02,  1.36042800e-02,  1.92000000e+02,        -4.41433082e+00],
# [-6.01700300e-02,  1.33077800e-02,  2.00000000e+02,        -4.39312433e+00]
        ])
    print('x0_vec=',x0_vec.tolist())
    Fidelity = {}
    Drive_param = {}
    repets = 5
    Drive_param[0] = x0_vec

    for idx in range(repets):
        print('\n\n\n***&&&***repeats=%d\n\n\n'%idx)
        fidelity = []
        drive_param = []
        for jdx, tg in tqdm(enumerate(tg_vec)):
            # x0 = Drive_param[idx][jdx][:2]
            args = [H0, n_theta1_truc, n_theta2_truc, eta, W_20_50, tg, state_tot, drive_ab]
            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cz_dark,
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
                # x0=x0
                )
            fidelity.append(res.fun)
            drive_param.append(res.x.tolist())
            print('\ngate time=', tg)
            print(res, '\n')
            # print('fidelity = ', fidelity)
            # print('drive_param = ', np.array(drive_param).tolist(), '\n')
            print('\nfidelity = ')
            for i in range(0, len(fidelity), 4):
                print(', '.join(map(str, fidelity[i:i+4])), ',')
            print('\ndrive_param = ')
            for i in drive_param:
                print(i,',')

        Fidelity[idx+1] = fidelity
        Drive_param[idx+1] = drive_param
    for i in range(repets):
        print(f'Fidelity_{i} = ')
        for j in range(0, len(Fidelity[i+1]), 4):
            print(', '.join(map(str, Fidelity[i+1][j:j+4])), ',')
        print(f'Drive_param_{i} = ')
        for j in Drive_param[i+1]:
                print(j,',')
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('workers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_sweep_no_x0():
    Fidelity = {}
    Drive_param = {}
    Drive_param[0] = [None]* len(tg_vec)
    repets = 1
    for idx in range(repets):
        print('\n\n\n***&&&***repeats=%d\n\n\n'%idx)
        fidelity = []
        drive_param = []
        for jdx, tg in tqdm(enumerate(tg_vec)):
            x0 = Drive_param[idx][jdx]
            args = [H0, n_theta1_truc, n_theta2_truc, eta, W_target, tg, state_tot, drive_ab]
            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cz_dark,
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
            # print('fidelity = ', fidelity)
            # print('drive_param = ', np.array(drive_param).tolist(), '\n')
            print('\nfidelity = ')
            for i in range(0, len(fidelity), 4):
                print(', '.join(map(str, fidelity[i:i+4])), ',')
            print('\ndrive_param = ')
            for i in drive_param:
                print(i,',')

        Fidelity[idx+1] = fidelity
        Drive_param[idx+1] = drive_param
    for i in range(repets):
        print(f'Fidelity_{i} = ')
        for j in range(0, len(Fidelity[i+1]), 4):
            print(', '.join(map(str, Fidelity[i+1][j:j+4])), ',')
        print(f'Drive_param_{i} = ')
        for j in Drive_param[i+1]:
                print(j,',')


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    args_all = ut.get_operator_two_zeropi()
    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot  ] = args_all
    truc = len(trunc_states)
    state_tot = [qt.basis(truc, i) for i in range(truc)]
    W_20_50 = np.abs(ut.transition_frequency(order_sort.index('20') , order_sort.index('50'), eval_tot))
    W_22_52 = np.abs(ut.transition_frequency(order_sort.index('22') , order_sort.index('52'), eval_tot))
    W_00_01 = np.abs(ut.transition_frequency(order_sort.index('00') , order_sort.index('01'), eval_tot))
    W_target = W_20_50

    print('W_20_50 = ', W_20_50)
    print('W_22_52 = ', W_22_52)
    print('W_00_01 = ', W_00_01)
    print('W_target = ', W_target)

    n1_22 = n_theta1_truc[trunc_states.index('22'), trunc_states.index('52')]
    n2_22 = n_theta2_truc[trunc_states.index('22'), trunc_states.index('52')]
    n1_20 = n_theta1_truc[trunc_states.index('20'), trunc_states.index('50')]
    n2_20 = n_theta2_truc[trunc_states.index('20'), trunc_states.index('50')]
    eta = - n2_22 / n1_22


    detune_bounds = (-0.1, 0.15)
    amp_bounds = (0, 0.1)
    eta_bounds = (-0.5, 0.5)
    args_indep = (detune_bounds, amp_bounds, eta_bounds)
    tg_vec = np.array(np.linspace(108, 196, num=12).tolist()
                       + np.linspace(42, 198, num=40).tolist())

    workers = 130
    popsize = 50
    mutation = (0.5, 1.99)
    recombination = 0.7
    tol = 0.01
    print('\ndetune_bounds=',detune_bounds)
    print('amp_bounds=',amp_bounds)
    print('eta_bounds=',eta_bounds)
    print('tg_vec = ')
    for i in range(0, len(tg_vec), 4):
        print(', '.join(map(str, tg_vec[i:i+4])), ',')
    print('workers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    print('eta = ', eta)

    drive_ab = True

    fidelity_sweep_no_x0()

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('workers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)


    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))