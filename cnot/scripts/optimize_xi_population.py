import sys
from pathlib import Path

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)
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
from scipy.ndimage import gaussian_filter


tg_vec = np.arange(30,51,2)
XI_flag = True
flag_00, flag_22 = True, False
tg_bound = (-0.01, 0.01) #(-1e-10, 1e-10) #
A1_bound, A2_bound, detune1_bound, detune2_bound = [
    (0.001, 0.5), (0.001, 0.5), (-0.3, 0.3), (-0.3, 0.3)     ]
workers, popsize = 100, 10
recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
num_cpus = 4


truc_one_qubit, truc_tot, charge_pick = 300, 1000, True

[hspace_full, eket_tot, eval_tot, n_theta0_dress, 
    n_theta1_dress, hspace_0, hspace_1, logi_state
    ] = ut.load_qubit_data_2q(truc_one_qubit, truc_tot, charge_pick)

logi_state = ['0-0', '0-2', '2-0', '2-2']
hspace_part = logi_state + ['1-4', '8-2', '8-0', '0-8', '2-8', ] # hspace_full[:200] # ut.truc_model['cnot_short_1000'][:200]
H0_full = qt.Qobj(np.diag(eval_tot))
index_part = [hspace_full.index(i) for i in hspace_part]
len_part = len(hspace_part)
H0_part = ut.truncate_2( H0_full, index_part)
n_theta0_part = ut.truncate_2(n_theta0_dress, index_part)
n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
logi_idx_part = [hspace_part.index(i) for i in logi_state]

if XI_flag:
    mid_state = '8-2'
    idx_0 = hspace_full.index('0-2')
    idx_1 = hspace_full.index('2-2')
    H_drive_part = [ H0_part,   [n_theta0_part, ut.drive_gauss_A],
                                [n_theta0_part, ut.drive_gauss_B]  ]    
else:
    mid_state = '0-8'
    idx_0 = hspace_full.index('0-0')
    idx_1 = hspace_full.index('0-2')
    H_drive_part = [ H0_part,   [n_theta1_part, ut.drive_gauss_A],
                                [n_theta1_part, ut.drive_gauss_B]  ]

idx_2 = hspace_full.index(mid_state)
W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
state_full = [qt.basis(len_part, i) for i in range(4)]
logi_state_one = [0,2]

def optimize_population(arg_optimize, *args):
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = arg_optimize
    [H_drive_part, W_0_2, W_1_2, num_cpus, XI_flag, flag_00, flag_22] = args   

    pulse_args = {'drive_amp_A': drive_amp_A ,
                'drive_freq_A': W_0_2 + 2*np.pi*detune_A,
                'drive_amp_B': drive_amp_B ,
                'drive_freq_B': W_1_2 + 2*np.pi*detune_B,
                'gate_time': tg }
    tlist = np.linspace(0, tg, num=int(tg))  # total time    

    result = {}
    for idx, state_i in enumerate(logi_state_one):
        for jdx, state_j in enumerate(logi_state_one):
            # This simulation is just for viewing the affect of the pulse
            result[idx, jdx] = qt.mesolve(
                H_drive_part,
                qt.basis(len_part, hspace_part.index( str(state_i) +'-'+str(state_j) )),
                tlist,
                e_ops=[state * state.dag() for state in state_full],
                args=pulse_args,
                options=qt.Options(store_states=True, num_cpus=num_cpus, nsteps=10000) )    
    pop = 0
    if XI_flag:            
        p00 = np.log10(1- result[0,0].expect[2][-1] ) # population in |0,2>
        p02 = np.log10(1- result[0,1].expect[3][-1] )
        p20 = np.log10(1- result[1,0].expect[0][-1] )
        p22 = np.log10(1- result[1,1].expect[1][-1] )
        pop += (p00 + p20) if flag_00 else 0
        pop += (p02 + p22) if flag_22 else 0        
    else:
        p00 = np.log10(1- result[0,0].expect[1][-1] ) # population in |0,2>
        p02 = np.log10(1- result[0,1].expect[0][-1] )
        p20 = np.log10(1- result[1,0].expect[3][-1] )
        p22 = np.log10(1- result[1,1].expect[2][-1] )     
        pop += (p00 + p02) if flag_00 else 0
        pop += (p20 + p22) if flag_22 else 0   
    return  pop    


print(f'flag_00 = {flag_00}, flag_22 = {flag_22}')
print('W_0_2 = ', np.round(W_0_2, 3), ', W_1_2 = ', np.round(W_1_2, 3))
print('tg_bound=', tg_bound, ', A1_bound=',A1_bound, ', A2_bound=',A2_bound)
print('detune1_bound=',detune1_bound, ', detune2_bound=', detune2_bound)
if 'x0_vec' in globals():
    print('params = ')
    for i in x0_vec:
        print(np.round(i,6).tolist(),',')
if 'tg_vec' in globals():
    print('tg_vec = ')
    for i in range(0, len(tg_vec), 4):
        print(', '.join(map(str, tg_vec[i:i+4])), ',')

print('workers=',workers, ', popsize=',popsize)
print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)

arg_truc = [H_drive_part, W_0_2, W_1_2, num_cpus, XI_flag, flag_00, flag_22]
for tg in tg_vec:

    print('\n---- tg = ', tg, '----\n')

    tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
    bounds = (tg_bounds, A1_bound, A2_bound, detune1_bound, detune2_bound)
    res = sp.optimize.differential_evolution(
        func=optimize_population,
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

    print(res, '\n')