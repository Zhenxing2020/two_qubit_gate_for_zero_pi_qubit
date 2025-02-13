import sys
sys.path.append('../')
import scqubits as scq
import pandas as pd
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from qutip.qip.operations import rz, cz_gate
from tqdm import tqdm
from matplotlib.colors import LogNorm
import time, pytz, os, itertools, cmath
from datetime import datetime
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
from joblib import Parallel, delayed
import scipy.sparse as ssp
from sympy import symbols
import scipy as sp
import utils_2Q_gate_zp as ut

###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_sweep():
    fidelity = []
    drive_param = []
    arg_truc = [H_drive_truc, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_truc]
    for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, A1_bound)

        res = sp.optimize.differential_evolution(
            func=cnot_fidelity_log_tg_only,
            bounds=bounds,
            args=arg_truc,
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
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_tot_2 = 500

    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    n_theta0_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    truc_list = np.arange(truc_tot_2)
    hspace_full = hspace_full[:truc_tot_2]
    eval_tot = eval_tot[:truc_tot_2]
    n_theta0_dress = ut.truncate_2(n_theta0_dress, truc_list)

    A1_bound = (0, 0.4)
    tg_bound = (-1, 1) #(-1e-10, 1e-10) #
    tg_vec = [50, 300, 400]#np.arange(20, 101, 10)

    workers, popsize = 100, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]

    logi_state = ['0-0', '0-2', '2-0', '2-2']
    idx_0 = hspace_full.index('0-2')
    idx_1 = hspace_full.index('2-2')
    idx_2 = hspace_full.index('8-2')
    W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
    n_theta0_0_2 = n_theta0_dress[idx_2, idx_0]
    n_theta0_1_2 = n_theta0_dress[idx_2, idx_1]

    # hspace_truc = hspace_full[:100]
    hspace_truc = [
    '0-0', '0-2', '2-2', '8-2', '2-0', '12-2', '1-2', '1-0', '5-2', '5-0' ,
    '4-9', '8-5', '2-1', '22-2', '0-1', '9-2', '5-8', '2-5', '5-5', '34-2' ,
    '8-0', '0-5', '13-2', '1-4', '30-2', '20-5', '4-2', '15-5', '15-0', '15-2' ,
    '1-1', '8-9', '26-5', '24-2', '26-2', '9-0', '20-2', '25-2', '4-5', '4-0' ,
    '5-1', '12-5', '12-0', '18-2', '9-5', '1-5', '13-0', '37-2', '35-2', '9-4' ,
    '8-1', '20-0', '4-1', '9-1', '13-9', '33-2', '1-13', '18-0', '18-5', '9-9' ,
    '12-8', '5-4', '0-21', '9-8', '1-8', '26-0', '5-12', '5-16', '15-1', '22-5' ,
    '25-4', '1-25', '4-4', '5-9', '2-4', '18-4', '9-12', '15-4', '1-18', '64-2' ,
    '45-0', '12-9', '13-5', '4-8', '44-2', '8-4', '45-5', '25-5', '2-12', '12-1' ,
    '0-8', '50-2', '4-12', '45-2', '2-21', '18-1', '15-9', '25-0', '1-9', '2-8' ,
    '24-9', '26-8', '12-4', '35-0', '18-8', '28-2', '41-0', '50-5', '20-4', '39-0' ,
    '0-4', '33-0', '41-2', '46-2', '39-2', '25-1', '56-2', '59-2', '34-5', '50-0' ,
    '30-5', '13-1', '22-0', '22-9', '8-18', '22-4', '26-9', '58-2', '60-2', '2-25' ,
    '20-9', '4-21', '4-13', '54-2', '4-16', '34-9', '8-12', '15-8', '35-4', '24-5' ,
    '59-0', '56-0', '65-0', '35-5', '2-30', '33-5', '51-2', '1-21', '0-25', '46-5' ,
    '41-1', '24-0', '18-9', '28-1', '25-8', '46-0', '2-9', '51-5', '13-8', '20-1' ,
    '65-2', '30-4', '9-16', '0-12', '28-5', '2-26', '0-30', '22-8', '0-45', '9-13' ,
    '0-24', '37-5', '13-4', '0-18', '24-1', '34-1', '0-9', '22-1', '33-12', '34-0' ,
    '37-0', '5-18', '4-24', '69-0', '41-5', '39-5', '13-12', '44-5', '12-20', '30-1' ,
    '12-12', '30-0', '13-16', '4-18', '44-0', '18-20', '33-1', '41-8', '15-12', '1-42' ,
    '2-13', '26-1', '85-0', '8-24', '0-20', '8-8', '5-13', '1-20', '33-4', '4-25' ,
    '74-0', '12-18', '24-4', '5-21', '12-13', '20-8', '41-4', '35-1', '2-18', '24-8' ,
    '46-4', '58-0', '0-13', '0-39', '30-12', '34-4', '4-30', '39-8', '9-18', '1-16' ,
    '81-0', '8-21', '51-1', '37-9', '18-12', '30-9', '5-30', '4-20', '5-25', '35-9' ,
    '22-18', '39-4', '60-0', '0-16', '51-0', '8-16', '2-34', '26-4', '18-21', '39-9' ,
    '46-1', '24-12', '66-0', '15-21', '25-12', '1-12', '25-9', '2-35', '8-13', '28-9' ,
    '2-16', '18-16', '12-24', '56-4', '37-8', '86-0', '30-8', '2-36', '28-4', '12-16' ,
    '13-25', '1-30', '1-24', '28-0', '22-12', '0-35', '1-39', '77-0', '54-0', '37-1' ,
    '37-4', '79-0', '1-28', '35-8', '34-12', '24-13', '33-9', '33-8', '25-16', '44-1' ,
    '34-8', '15-18', '59-1', '24-16', '45-4', '26-13', '0-36', '5-35', '60-1', '45-1' ,
    ]
    truc_index = [hspace_full.index(i) for i in hspace_truc]
    truc_len = len(hspace_truc)

    H0_full = qt.Qobj(np.diag(eval_tot))
    logi_idx_full = [hspace_full.index(i) for i in logi_state]
    H_drive_full = [ H0_full,   [n_theta0_dress, ut.drive_gauss_A],
                                [n_theta0_dress, ut.drive_gauss_B]  ]

    H0_truc = ut.truncate_2( H0_full, truc_index)
    n_theta0_truc = ut.truncate_2(n_theta0_dress, truc_index)
    logi_idx_truc = [hspace_truc.index(i) for i in logi_state]
    H_drive_truc = [ H0_truc,   [n_theta0_truc, ut.drive_gauss_A],
                                [n_theta0_truc, ut.drive_gauss_B]  ]

    c_op_list = []
    num_cpus = None
    def cnot_fidelity_log_tg_only(arg_optimize, *args):
        tg, drive_amp_A = arg_optimize
        drive_amp_B = drive_amp_A * n_theta0_0_2 / n_theta0_1_2

        [H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx] = args
        detune_A, detune_B = 0, 0

        arg_all = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
        H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx]
        return ut.cnot_fidelity_log(arg_all)


    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_tot_2,  )
    print('W_0_2 = ', np.round(W_0_2, 3), ', W_1_2 = ', np.round(W_1_2, 3))
    print('n_theta0_0_2 = ', np.round(n_theta0_0_2, 5),
        ', n_theta0_1_2 = ', np.round(n_theta0_1_2, 5))
    print('tg_bound=', tg_bound, 'A1_bound=',A1_bound, )
    print('gate_time_vector:', np.array(tg_vec).tolist())
    print('workers=', workers, ', popsize=', popsize)
    print('recombination=', recombination, ', tol=', tol, ', mutation=', mutation)
    print(f'\nhspace_truc (len={truc_len}) = [')
    for i in range(0, len(hspace_truc), 10):
        print(", ".join(f"'{x}'" for x in hspace_truc[i:i + 10]), ',')
    print(']')

    fidelity_sweep()

    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)


