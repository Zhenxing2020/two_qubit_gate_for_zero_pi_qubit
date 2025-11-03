import sys
sys.path.append('../')

import scqubits as scq # type: ignore
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
import ham_data as hd

###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_sweep():
    n_cpu_optimize = 1
    c_op_list = []
    args_truc = [H_drive_select, W_20_50, n_cpu_optimize, c_op_list, 
                logi_idx_select, option_ideal, option_noisy]
    fidelity = []
    drive_param = []
    fidelity_full = []
    for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
    # for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp_bound, detune_bound)

        ut.print_time()
        # print('args_truc==args_truc2', args_truc==args_truc2)
        res = sp.optimize.differential_evolution(
            func=ut.cz_fidelity_log_noise,
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
            x0=x0_vec[jdx,:3],
            polish=False, # 'True' will make the for-loop break
            )
        ut.print_time()
        fidelity.append(res.fun)
        drive_param.append(res.x.tolist())

        print(res, '\n')
        # print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        print('\ntg = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(np.array((x0_vec[:,0])[:jdx+1])[i:i+4], 8))), ',')
        print(f'\nlog of gate error (truc={truc_optimize}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')
        print(f'\ndrive_param (truc={truc_optimize}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        # ut.print_time()
        # n_cpu_parallel = 16
        # tg, drive_amp, detune = drive_param[jdx]

        # arg_all = [tg, drive_amp, detune,
        #            n_cpu_parallel, hspace_large, W_20_50, H_drive_large, logi_idx_large]
        # fidelity_full.append(ut.cz_fidelity_log_old(arg_all))

        # print(f'\nlog of gate error (truc={truc_large}) = ')
        # for i in range(0, len(fidelity_full), 4):
        #     print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')
        # print('\namp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
        # ut.print_time()

if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    ut.print_time()

    truc_large = 1000
    truc_optimize = 200

    use_truc_model, truc_model_name = False, 'cz_short_500_detune1'
    max_step_ideal, max_step_noisy = 1e-3, 1e-3 # Set max_step to 0 for parallel execution

    # amp_bound, detune_bound, tg_bound = [(0., 0.1), (-0.1, 0.1), (-0.01, 0.01)] # 
    amp_bound, detune_bound, tg_bound = [(0.032457, 0.0438), (0.024913, 0.075696), (-0.01, 0.01)] # 
    # tg_list =  [0, 1, 2] # np.arange(31) # Select the first row for testing

    cz_run = True
    # x0_vec = ut.load_drive_params_2q(cz_run) [[0,1,2],:] #[0::3,]  # [tg_list, ]  # [1::4,]
    x0_vec =  np.array([
        # [32.042591, 0.032458, 0.024914],
        [38.048132, 0.032458, 0.024914]
        ]) #[0::3,]  # [tg_list, ]  # [1::4,]
    
    workers, popsize = 50, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.5)]

    folder_load = '../../data/_truc_3000'
    [hspace_full, eket_tot, eval_tot, n_theta0_dress, 
        n_theta1_dress, hspace_0, hspace_1, logi_state] = hd.load_two_qubit_data(folder_load, return_full=False)
    drive_term = n_theta1_dress
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    if use_truc_model:
        hspace_select = ut.truc_model[truc_model_name][:truc_optimize]
    else:
        hspace_select = hspace_full[:truc_optimize]
        
    option_ideal, option_noisy = ut.get_qutip_options(max_step_ideal, max_step_noisy) 

    logi_idx_select = [hspace_select.index(i) for i in logi_state]
    index_select = [hspace_full.index(i) for i in hspace_select]
    H_drive_select, eket_truc = ut.build_hamiltonian_2q(cz_run, index_select, eval_tot, 
                                                        eket_tot, drive_term)

    hspace_large = hspace_full[:truc_large]
    logi_idx_large = [hspace_large.index(i) for i in logi_state]
    idx_large = np.arange(truc_large).tolist()
    H_drive_large, eket_truc = ut.build_hamiltonian_2q(cz_run, idx_large, eval_tot, 
                                                        eket_tot, drive_term)                                                        


    print('gate_time_vector:', np.array(tg_vec).tolist()) if 'tg_vec' in globals() else None
    print('params:')
    for i in x0_vec:
        print(i.tolist(), ',')
    ut.print_data(f'hspace_select (len={len(hspace_select)})', 
                    hspace_select, num_each_row=10)
    ut.print_data(f'index_select (len={truc_optimize})', index_select, num_each_row=10)

    print('truc_tot_2=', truc_large)
    print('amp_bounds=', amp_bound, ', detune_bounds=', detune_bound, ', tg_bound=', tg_bound)
    print('W_20_50 = ', np.round(W_20_50, 3))
    print('workers=', workers, ', popsize=', popsize)
    print('recombination=', recombination, ', tol=', tol, ', mutation=', mutation)

    fidelity_sweep()

    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)


















