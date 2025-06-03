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
    W_target = W_20_50
    drive_n1 = False
    drive_n2 = True
    drive_term = qt.Qobj(np.zeros((truc, truc)))
    if drive_n1:
        drive_term += n_theta1_truc
    if drive_n2:
        drive_term += n_theta2_truc
    print('W_20_50 = ', W_20_50)
    print('W_22_52 = ', W_22_52)
    print('W_00_01 = ', W_00_01)
    print('W_target = ', W_target)
    print('drive_n1=', drive_n1)
    print('drive_n2=', drive_n2)

    H_qbt_drive = [H0, [2*np.pi* drive_term, ut.drive_gauss_A] ]

    detune_bounds = (-0.2, 0.2)
    amp_bounds = (0, 0.1)
    args_indep = (detune_bounds, amp_bounds)
    tg_vec = np.linspace(40, 100, num=11)
    workers = 120
    popsize = 40
    mutation = (0.5, 1.99)
    recombination = 0.7
    tol = 0.01
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


    print('\ndetune_bounds=', detune_bounds)
    print('amp_bounds=', amp_bounds)
    print('tg_vec = ')
    for i in range(0, len(tg_vec), 4):
        print(', '.join(map(str, tg_vec[i:i+4])), ',')
    print('workers=', workers)
    print('popsize=', popsize)
    print('mutation=', mutation)
    print('recombination=', recombination)
    print('tol=', tol)

    fidelity_sweep()

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('workers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)


    print("End Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


