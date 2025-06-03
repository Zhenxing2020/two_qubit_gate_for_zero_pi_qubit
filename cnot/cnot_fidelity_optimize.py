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
    fidelity_full = []
    arg_truc = [H_drive_part, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_part, mid_state]
    for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
    # for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, A1_bound, A2_bound, detune1_bound, detune2_bound)
        res = sp.optimize.differential_evolution(
            func=ut.cnot_fidelity_log_optimize,
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
            x0=x0_vec[jdx],
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x.tolist())
        print(res, '\n')
        # print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        print('\ntg = ', np.array((x0_vec[:,0])[:jdx+1]).tolist())
        print(f'\nlog of gate error (truc={len_part}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')
        print(f'\ndrive_param (truc={len_part}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = drive_param[jdx]
        arg_all = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
                    H_drive_False, W_0_2, W_1_2, num_cpus, c_op_list, logi_idx_False, mid_state]
        fidelity_full.append(ut.cnot_fidelity_log(arg_all))

        print(f'\nlog of gate error (truc={H_drive_False[0].shape[0]},False) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')
            
        print('amp1_bounds=',A1_bound, ', amp2_bounds=',A2_bound, ', tg_bound=', tg_bound)
        print('detune1_bounds=',detune1_bound, ', detune2_bounds=', detune2_bound)
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_full = 500

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

    A1_bound, A2_bound, detune1_bound, detune2_bound = [
        # (0, 0.2), (0, 0.2), (-0.05, 0), (-0.05, 0)]
        # (0, 0.2), (0, 0.2), (-0.01, 0.01), (-0.01, 0.01)]
         (0, 0.2), (0, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
    tg_bound = (-0.01, 0.01) #(-1e-10, 1e-10) #

    # tg_vec = [2, 3] #np.arange(350, 401, 10)
    # tg_vec = [50, 100, 150, 200] #np.arange(350, 401, 10)
    x0_vec = np.array([
# [50.000812, 0.178565, 0.07488, -0.067799, -0.060233] ,
# [100.000812, 0.178565, 0.07488, -0.067799, -0.060233] ,
# [150.000812, 0.178565, 0.07488, -0.067799, -0.060233] ,
# [199.991759, 0.18619, 0.063599, -0.078985, -0.081857] ,

[50.000812, 0.178565, 0.07488, -0.0067799, -0.0060233] ,
[100.007417, 0.144212, 0.066543, 0.073432, 0.074519] ,
[150.007417, 0.144212, 0.066543, 0.073432, 0.074519] ,
[200.007078, 0.159118, 0.057099, 0.071357, 0.066645] ,
[250.007417, 0.144212, 0.066543, 0.073432, 0.074519] ,
[300.007078, 0.159118, 0.057099, 0.071357, 0.066645] ,
    ])

    workers, popsize = 100, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    logi_state = ['0-0', '0-2', '2-0', '2-2']

    mid_state = '4-1'

    if mid_state in ['8-2', '4-5' ]:
        idx_0 = hspace_full.index('0-2')
        idx_1 = hspace_full.index('2-2')
    elif mid_state in ['1-4', '8-0', '4-1']:
        idx_0 = hspace_full.index('0-0')
        idx_1 = hspace_full.index('2-0')

    if mid_state in ['8-2']:        
        hspace_part = [  ### 8-2
'0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
'5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '2-4', '9-0', '1-5' ,
'5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '8-1', '4-4', '13-0' ,
'15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5', '0-18', '2-9', '0-21' ,
'18-0', '9-2', '12-1', '5-5', '0-24', '4-8', '1-13', '20-0', '8-4', '22-0' ,

'2-12', '13-1', '24-0', '0-25', '15-1', '5-8', '4-9', '12-2', '9-4', '8-5' ,
'25-0', '13-2', '26-0', '1-18', '15-2', '5-9', '4-12', '0-30', '18-1', '12-4' ,
'9-5', '1-21', '8-8', '20-1', '4-13', '22-1', '4-16', '30-0', '13-4', '5-12' ,
'15-4', '24-1', '33-0', '18-2', '34-0', '2-21', '8-9', '35-0', '1-25', '9-8' ,
'12-5', '37-0', '20-2', '5-16', '22-2', '13-5', '25-1', '26-1', '24-2', '15-5' ,

'2-25', '39-0', '8-12', '2-26', '9-9', '18-4', '28-1', '12-8', '41-0', '0-45' ,
'4-21', '13-8', '25-2', '20-4', '22-4', '4-24', '26-2', '5-18', '15-8', '30-1' ,
'9-12', '44-0', '12-9', '28-2', '45-0', '33-1', '18-5', '2-30', '46-0', '34-1' ,
'13-9', '9-13', '9-16', '20-5', '22-5', '50-0', '30-2', '25-4', '15-9', '8-18' ,
'24-5', '12-12', '33-2', '18-8', '34-2', '35-2', '13-12', '41-1', '37-2', '22-8' ,

'56-0', '25-5', '26-5', '30-4', '18-9', '39-2', '13-16', '59-0', '33-4', '28-5' ,
'41-2', '35-4', '20-9', '22-9', '25-8', '65-0', '26-8', '24-9', '30-5', '12-18' ,
'44-2', '33-5', '45-2', '34-5', '35-5', '46-2', '12-20', '69-0', '37-5', '26-9' ,
'50-2', '51-2', '39-5', '54-2', '41-5', '56-2', '58-2', '34-9', '18-20', '59-2' ,
'44-5', '45-5', '60-2', '41-8', '46-5', '64-2', '65-2', '33-12', '50-5', '51-5' ,
        ]
    elif mid_state in [ '4-5']:        
        hspace_part = [  ## 4-5
'0-0', '0-2', '2-0', '4-5', '2-2', '8-5', '1-2', '1-0', '5-2', '5-0' ,
'12-2', '22-2', '2-1', '34-2', '12-5', '0-1', '13-2', '9-2', '2-5', '4-2' ,
'0-5', '26-5', '15-5', '56-2', '9-4', '1-1', '9-0', '5-5', '20-2', '22-5' ,
'13-0', '20-5', '26-2', '4-0', '5-1', '8-2', '9-8', '1-4', '15-2', '9-5' ,
'12-0', '1-5', '18-5', '15-0', '8-0', '34-5', '24-2', '8-9', '12-12', '9-9' ,
'45-2', '35-2', '4-9', '13-5', '4-1', '13-9', '18-2', '25-2', '1-18', '18-4' ,
'25-5', '30-2', '1-8', '33-2', '25-0', '37-2', '20-0', '5-16', '4-4', '2-4' ,
'18-1', '8-1', '18-0', '50-2', '1-13', '9-16', '12-9', '5-9', '9-12', '15-4' ,
'33-5', '30-5', '8-18', '0-21', '26-9', '5-12', '24-5', '5-8', '26-0', '22-0' ,
'25-4', '46-0', '13-4', '5-4', '20-9', '46-2', '15-1', '9-1', '2-9', '2-13' ,
'26-8', '13-8', '0-4', '0-30', '24-0', '4-12', '37-5', '1-25', '15-9', '8-12' ,
'2-30', '24-9', '2-25', '18-9', '45-0', '4-8', '8-4', '12-1', '26-12', '39-2' ,
'15-8', '30-0', '8-24', '22-4', '35-0', '13-16', '39-4', '22-8', '41-2', '12-8' ,
'12-18', '34-0', '2-21', '37-0', '0-39', '39-5', '18-12', '0-8', '2-26', '1-16' ,
'9-18', '41-0', '15-12', '30-4', '4-18', '39-0', '35-5', '2-8', '30-1', '18-8' ,
'33-0', '1-9', '20-4', '5-30', '35-4', '2-34', '0-25', '4-13', '50-0', '0-12' ,
'8-16', '1-21', '44-2', '13-1', '25-1', '8-8', '28-1', '30-8', '33-4', '20-1' ,
'24-8', '20-12', '0-9', '28-5', '22-1', '28-2', '25-8', '0-20', '2-12', '59-0' ,
'41-5', '0-24', '39-1', '65-0', '56-0', '24-16', '20-8', '45-4', '22-9', '24-4' ,
'4-16', '66-0', '12-16', '1-20', '0-18', '4-24', '51-2', '58-0', '5-13', '34-1' ,
        ]
    elif mid_state in [ '4-1']:        
        hspace_part = [  ### 4-1
'0-0', '0-2', '2-0', '4-1', '2-2', '34-0', '4-4', '8-1', '1-2', '1-0' ,
'5-2', '5-0', '22-0', '13-0', '4-5', '15-1', '26-1', '2-1', '20-1', '0-1' ,
'1-4', '8-2', '2-5', '0-5', '8-0', '12-0', '9-0', '12-4', '15-0', '15-2' ,
'0-18', '24-0', '18-1', '8-8', '37-0', '56-0', '18-2', '1-8', '2-4', '5-8' ,
'20-2', '4-13', '18-0', '24-4', '20-0', '0-21', '5-4', '9-5', '1-21', '1-13' ,

'9-1', '13-4', '33-1', '0-20', '1-18', '8-9', '25-0', '26-0', '0-13', '12-5' ,
'4-8', '12-1', '4-9', '9-4', '25-2', '5-9', '34-4', '22-4', '8-4', '34-1' ,
'5-1', '8-16', '45-1', '15-13', '0-8', '50-1', '1-9', '2-8', '24-2', '26-2' ,
'5-5', '9-12', '4-12', '81-0', '45-0', '13-5', '2-12', '35-0', '39-1', '9-8' ,
'15-8', '64-0', '13-2', '13-1', '46-1', '35-2', '74-0', '45-2', '35-1', '2-21' ,

'59-0', '41-1', '4-2', '41-0', '60-0', '1-26', '39-0', '44-0', '1-1', '65-0' ,
'8-5', '34-2', '20-4', '41-2', '39-2', '37-4', '28-0', '70-0', '44-1', '4-24' ,
'33-2', '33-0', '22-1', '1-25', '2-18', '50-0', '58-0', '28-4', '60-1', '34-5' ,
'50-2', '46-4', '5-18', '12-2', '9-2', '5-12', '1-5', '2-13', '0-24', '35-4' ,
'30-4', '4-0', '41-5', '54-0', '44-4', '69-0', '75-0', '46-0', '56-2', '28-1' ,

'25-1', '1-16', '18-9', '26-4', '12-9', '77-0', '30-1', '33-4', '1-12', '28-5' ,
'0-36', '33-8', '4-16', '34-8', '2-9', '9-20', '22-2', '41-4', '30-0', '26-8' ,
'51-1', '1-20', '59-1', '0-9', '18-5', '24-1', '9-28', '18-8', '20-12', '15-4' ,
'25-5', '0-16', '30-2', '24-9', '37-1', '13-12', '2-44', '26-5', '2-16', '45-4' ,
'66-0', '15-16', '1-24', '0-39', '9-9', '24-5', '20-5', '22-5', '5-13', '15-5' ,
    ]
    elif mid_state in [ '1-4']:        
        hspace_part = [  ### 1-4 which is indeed 8-0
'0-0', '0-2', '2-0', '1-4', '2-2', '12-0', '8-1', '1-2', '1-0', '5-2' ,
'5-0', '4-4', '30-0', '22-0', '2-1', '9-0', '0-1', '34-0', '2-5', '26-0' ,
'4-2', '1-1', '0-5', '1-8', '8-2', '15-0', '5-1', '4-8', '13-0', '9-2' ,
'12-1', '5-5', '20-0', '8-4', '4-0', '8-0', '13-2', '35-0', '1-13', '33-0' ,
'12-2', '18-0', '1-5', '24-0', '8-8', '15-2', '25-0', '2-4', '0-13', '25-2' ,

'4-5', '0-21', '4-9', '45-0', '18-1', '13-4', '22-1', '50-0', '5-4', '24-4' ,
'56-0', '8-5', '9-1', '37-0', '24-1', '20-2', '9-4', '34-1', '74-0', '12-4' ,
'4-1', '13-1', '18-2', '9-5', '5-8', '81-0', '22-4', '41-0', '22-2', '30-1' ,
'59-0', '26-4', '1-21', '65-0', '8-16', '39-0', '25-1', '15-4', '26-2', '34-4' ,
'24-2', '69-0', '0-4', '4-12', '20-4', '26-1', '44-0', '64-0', '1-12', '8-9' ,

'51-1', '12-8', '77-0', '5-12', '5-9', '12-5', '34-2', '13-8', '70-0', '45-1' ,
'30-2', '9-12', '41-4', '45-2', '15-1', '15-8', '37-2', '15-5', '35-2', '2-8' ,
'0-8', '60-1', '18-4', '28-1', '30-4', '1-9', '4-13', '37-1', '2-9', '28-0' ,
'13-5', '75-0', '33-2', '0-12', '41-2', '58-0', '33-4', '39-2', '37-4', '18-5' ,
'4-24', '8-13', '34-5', '50-2', '33-5', '20-1', '46-0', '0-9', '9-8', '12-9' ,

'22-8', '46-4', '33-1', '44-4', '20-5', '1-16', '8-12', '4-21', '24-8', '58-1' ,
'2-12', '41-5', '35-4', '34-8', '56-1', '5-18', '56-2', '35-5', '9-9', '2-13' ,
'26-5', '51-0', '33-8', '0-20', '54-0', '51-2', '46-2', '4-16', '35-1', '18-9' ,
'28-4', '15-9', '25-4', '26-8', '2-21', '59-1', '46-1', '9-28', '12-16', '13-9' ,
'28-5', '2-26', '44-2', '44-1', '28-2', '0-18', '15-12', '79-0', '66-0', '45-4' ,
]
    elif mid_state in [ '8-0']:        
        hspace_part = [  ### 8-0 which is indeed 1-4
'0-0', '0-2', '2-0', '2-2', '8-0', '8-2', '12-0', '8-1', '1-2', '1-0' ,
'1-8', '5-2', '5-0', '1-4', '22-0', '20-1', '8-8', '30-0', '34-0', '2-1' ,
'4-4', '13-0', '9-0', '0-1', '2-5', '9-5', '0-5', '5-1', '4-5', '15-1' ,
'15-0', '15-2', '9-1', '26-1', '18-1', '24-0', '25-0', '13-4', '4-1', '12-4' ,
'2-4', '5-4', '9-4', '1-21', '37-0', '5-8', '0-13', '20-2', '24-4', '18-2' ,

'20-0', '18-0', '5-9', '41-2', '0-21', '1-13', '4-13', '4-12', '1-9', '22-4' ,
'4-9', '81-0', '8-16', '0-8', '2-8', '15-9', '20-5', '8-9', '0-36', '26-0' ,
'35-4', '12-2', '28-0', '8-5', '12-5', '4-8', '1-20', '12-1', '5-21', '60-0' ,
'8-4', '45-1', '0-35', '26-2', '9-12', '15-8', '4-2', '64-0', '44-0', '37-1' ,
'13-2', '50-1', '2-13', '13-5', '0-20', '1-1', '70-0', '45-0', '35-0', '22-2' ,

'30-4', '35-2', '4-24', '45-2', '9-2', '13-1', '1-5', '30-2', '0-18', '9-8' ,
'5-5', '34-2', '26-4', '75-0', '41-0', '25-2', '39-0', '4-0', '25-4', '0-9' ,
'58-0', '13-9', '33-0', '5-18', '24-2', '24-5', '2-21', '18-5', '39-2', '20-4' ,
'35-1', '35-5', '33-2', '51-0', '50-0', '46-0', '37-4', '0-30', '34-5', '50-2' ,
'56-0', '4-21', '28-4', '46-4', '34-4', '54-0', '44-4', '12-9', '1-25', '59-0' ,

'26-8', '41-5', '41-1', '33-8', '39-1', '56-2', '28-1', '26-5', '22-1', '65-0' ,
'33-1', '18-9', '25-1', '34-1', '12-8', '66-0', '5-12', '15-5', '44-1', '24-9' ,
'28-5', '15-18', '30-1', '59-1', '13-8', '9-28', '60-1', '56-1', '2-12', '58-1' ,
'24-1', '41-4', '0-24', '15-16', '74-0', '1-12', '1-26', '69-0', '0-16', '2-44' ,
'1-33', '34-8', '46-1', '1-16', '18-4', '9-9', '25-5', '2-16', '15-4', '18-8' ,
]

    idx_2 = hspace_full.index(mid_state)
    W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
    H0_full = qt.Qobj(np.diag(eval_tot))
    
    H0_False = qt.Qobj(np.diag(eval_False))
    logi_idx_False = [hspace_False.index(i) for i in logi_state]

    index_part = [hspace_full.index(i) for i in hspace_part]
    len_part = len(hspace_part)
    H0_part = ut.truncate_2( H0_full, index_part)
    n_theta0_part = ut.truncate_2(n_theta0_dress, index_part)
    n_theta1_part = ut.truncate_2(n_theta1_dress, index_part)
    logi_idx_part = [hspace_part.index(i) for i in logi_state]

    if mid_state in [ '8-2', '4-5', '1-4', '8-0', '4-1' ]:
        H_drive_False = [ H0_False,     [n_theta0_False, ut.drive_gauss_A],
                                        [n_theta0_False, ut.drive_gauss_B]  ]
        H_drive_part = [ H0_part,   [n_theta0_part, ut.drive_gauss_A],
                                    [n_theta0_part, ut.drive_gauss_B]  ]
    else:
        H_drive_False = [ H0_False,     [n_theta1_False, ut.drive_gauss_A],
                                        [n_theta1_False, ut.drive_gauss_B]  ]
        H_drive_part = [ H0_part,   [n_theta1_part, ut.drive_gauss_A],
                                    [n_theta1_part, ut.drive_gauss_B]  ]
    c_op_list = []
    num_cpus = 1
    print('\nmid_state = ', mid_state)
    print('truc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_full )
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
    print('workers=', workers, ', popsize=', popsize)
    print('recombination=', recombination, ', tol=', tol, ', mutation=', mutation)
    print(f'\nhspace_truc (len={len_part}) = [')
    for i in range(0, len(hspace_part), 10):
        print(", ".join(f"'{x}'" for x in hspace_part[i:i + 10]), ',')
    print(']')

    fidelity_sweep()

    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)




###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
# def calculate_fidelity_tgbound_de_sweep_copy(smal_tgbound=False):
#     amp_bounds = (0, 0.1)
#     detune_bounds = (-0.3, 0.)
#     tg_vec = np.linspace(100, 300, num=6)
#     tg_bounds = []
#     # for i in range(len(tg_vec)-1):
#     #     if smal_tgbound == True:
#     #         tg_bounds.append((tg_vec[i+1]-0.01, tg_vec[i+1]))
#     #     else:
#     #         tg_bounds.append((tg_vec[i], tg_vec[i+1]))

#     workers = 100
#     popsize = 50
#     mutation = (0.5, 1.)
#     recombination = 0.7
#     tol = 0.01

#     x0_vec =  [[0.08214998234257803, 0.022990968193458108, 296.66484505688254, -0.023540986415900633, -0.037436621530911374],
# [0.0754751709841197, 0.024475013338660225, 317.4485123699629, -0.02987912201182486, -0.03701554537424631],
# [0.08382476280437687, 0.022136314047758274, 346.44487004688864, -0.027351130436972983, -0.03970968382979541],
# [0.09229964280612983, 0.026601756554862972, 376.59966093024786, -0.010816171264945452, -0.033824786376229904],
# [0.08305134439614023, 0.0263249315943987, 403.9988097743255, -0.01803758091532333, -0.02972465374945854],
# [0.06489069860635269, 0.016290503464261363, 420.1249512139678, -0.015430544119623952, -0.02485317699266697]]
#     x0_vec = np.array(x0_vec)


#     # x0_vec = np.array(x0_vec)
#     print('\namp_bounds=',amp_bounds)
#     print('detune_bounds=',detune_bounds)
#     print('tg_bounds=',tg_bounds)

#     print('\nworkers=',workers)
#     print('popsize=',popsize)
#     print('mutation=',mutation)
#     print('recombination=',recombination)
#     print('tol=',tol)
#     print('x0_vec=',x0_vec)

#     result_opt = []
#     fidelity = []
#     drive_param = []
#     for tg_bound in tqdm(tg_bounds):
#         x0 = None
#         for i in range(x0_vec.shape[0]):
#             if x0_vec[i,1] > tg_bound[0] and x0_vec[i,1] < tg_bound[1]:
#                 x0 = x0_vec[i]

#         bounds = (amp_bounds, amp_bounds, tg_bound, detune_bounds, detune_bounds)
#         res = sp.optimize.differential_evolution(
#             func=ut.get_fidelity_cnot_2A0,
#             bounds=bounds,
#             args=args,
#             disp=True,
#             callback=ut.print_soln,
#             init="sobol",
#             workers=workers,
#             popsize=popsize,
#             mutation=mutation,
#             recombination=recombination,
#             tol=tol,
#             polish=False, # 'True' will make the for-loop break
#             # x0=x0
#             )

#         print(res, '\n')
#         fidelity.append(res.fun)
#         drive_param.append(res.x)

#         print('fidelity = ', fidelity)
#         print('drive_param = ', np.array(drive_param).tolist())


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de_sweep():
    amp_bounds = (0, 0.04)
    detune_bounds = (-0.02, 0.)

    workers = 100
    popsize = 50
    mutation = (0.5, 1.)
    recombination = 0.7
    tol = 0.01

    x0_vec =   np.array([[0.02998763398547402, 129.060364854945, -0.009524334449526474],
 [0.029493350591334434, 157.71779952810067, -0.009941382763999338],
 [0.025950915455974895, 187.30292652442327, -0.00821681040657746],
 [0.022534777844595476, 216.42619098890876, -0.006268319044336534],
 [0.019572332065418703, 243.7496949100433, -0.004686625611314125],
 [0.01700764083302345, 274.1637763512114, -0.003590807111667511],
 [0.015643541999459592, 300.81632686689034, -0.003026750944939577],
 [0.014080775914738629, 329.4778392022009, -0.0024729159770241482],
 [0.012941227938039604, 354.7592332298872, -0.0020370898504066593],
 [0.011829096115033699, 385.78884937994235, -0.0017382166860813444],
 [0.011257929230028885, 404.36043366270167, -0.0015458429030142506],
 [0.010279964335750055, 445.0323390656677, -0.0013057734346887317],
 [0.009513295140347789, 476.83700141609415, -0.0011199238032474656],
 [0.009046422511480925, 501.3241321052972, -0.0010093707724371384],
 [0.008466852336793954, 535.2523726323412, -0.0008810170202933889],
 [0.00822939850536708, 549.9230118770298, -0.0008156841033343572],
 [0.007675580940429315, 589.106942356207, -0.0007218788768798149],
 [0.007247654633780758, 623.8079249944897, -0.0006615194170941016],
 [0.007084520133174579, 635.6321429096266, -0.0005958385713969502],
 [0.0066357557646690764, 680.9737743178056, -0.0005502122424595225],
 [0.006523487692003517, 691.0266744208469, -0.0005193310668362792],
 [0.006091476575864481, 738.6159881855083, -0.00044466628394118114],
 [0.005938864303736805, 760.76318597172, -0.00043410563429241867],
 [0.005718650638369904, 787.4597105231765, -0.00038813713923993576]])
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)

    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    print('x0_vec=',x0_vec)

    fidelity = []
    drive_param = []
    x0 = None
    for i in tqdm(range(x0_vec.shape[0])):
        tg_bound = (x0_vec[i,1]-0.01, x0_vec[i,1]+0.01)
        x0 = (x0_vec[i,0], x0_vec[i,0], x0_vec[i,1], x0_vec[i,2], x0_vec[i,2])

        bounds = (amp_bounds, amp_bounds, tg_bound, detune_bounds, detune_bounds)
        res = sp.optimize.differential_evolution(
            func=ut.get_fidelity_cnot_two_A0,
            bounds=bounds,
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
        drive_param.append(res.x)
        print(res, '\n')
        print('fidelity = ', fidelity)
        print('drive_param = ', np.array(drive_param).tolist())



###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de_x0():
    amp_bounds = (0, 0.1)
    detune_bounds = (-0.25, 0.)
    tg_bounds = []
    workers = 70
    popsize = 50
    mutation = (0.5, 1.)
    recombination = 0.7
    tol = 0.01
    x0_vec = np.array([[0.07138275557618617, 0.06263709015005474, 179.5743123263788, -0.23823111049155965, -0.2351910652491388],
                       [0.09998562868919722, 0.057375890802752766, 296.37375343645454, -0.06433356140617433, -0.06618230878238646],
])
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)

    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    print('x0_vec=',x0_vec)

    # def constraint(vars):
    #     A1, A2, tg, detune_1, detune_2 = vars
    #     return A2 - A1  # Ensure that y - x >= 0 which implies y >= x
    # nonlinear_constraint = sp.optimize.NonlinearConstraint(constraint, 0, np.inf)
    fidelity = []
    drive_param = []
    for i in tqdm(range(x0_vec.shape[0])):
        x0 = x0_vec[i]
        tg_bound = (x0_vec[i][2] - 0.000001, x0_vec[i][2] + 0.000001)
        bounds = (amp_bounds, amp_bounds, tg_bound, detune_bounds, detune_bounds)

        res = sp.optimize.differential_evolution(
            func=ut.get_fidelity_cnot_2A0,
            bounds=bounds,
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
            # x0=x0,
            # constraints=(nonlinear_constraint,)
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)
        print(res, '\n')
        print('fidelity = ', fidelity)
        print('drive_param = ', np.array(drive_param).tolist())
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)




###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_shgo():
    amp_bounds = (0, 0.1)
    detune_bounds = (-0.05, 0.)
    x0_vec = np.array([[0.08519997507149087, 0.028762603136169497, 404.00207461095397, -0.025975962691102063, -0.034013945266784036],
[0.07368360422310369, 0.01681222219061629, 420.13378612121545, -0.0164934651853555, -0.029706520480420585],
[0.019847538056258038, 0.04643890457429329, 452.94356604958097, -0.03221107458131659, -0.02656755643591971]]
    )
    options = {'disp': True, 'f_min':-6.3}
    fidelity = []
    drive_param = []
    for i in tqdm(range(x0_vec.shape[0])):
        tg_bound = (x0_vec[i][2] - 0.000001, x0_vec[i][2] + 0.000001)
        bounds = (amp_bounds, amp_bounds, tg_bound, detune_bounds, detune_bounds)
        res = sp.optimize.shgo(
            func=ut.get_fidelity_cnot_2A0,
            bounds=bounds,
            args=args,
            workers=100,
            options=options
            )
        print(res, '\n')
        fidelity.append(res.fun)
        drive_param.append(res.x)
        print('fidelity = ', fidelity)
        print('drive_param = ', np.array(drive_param).tolist())







###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de_repeat2(smal_tgbound=True):
    amp_bounds = (0, 0.2)
    detune_bounds = (-0.3, 0.)

    workers = 100
    popsize = 50
    mutation = (1.5, 1.99)
    recombination = 0.7
    tol = 0.01

    tg_vec = np.linspace(142.14, 200+6.43, num=6)
    # tg_vec = np.linspace(20, 200, num=15)
    tg_bounds = []
    if smal_tgbound:
        for i in range(len(tg_vec)):
            tg_bounds.append((tg_vec[i]-0.00000001, tg_vec[i]))
    else:
        for i in range(len(tg_vec)-1):
            tg_bounds.append((tg_vec[i], tg_vec[i+1]))

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)

    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)

    Fidelity = {}
    Drive_param = {}
    repets = 10
    for idx in range(repets):
        print('\n\n\n***&&&***repeats=%d\n\n\n'%idx)
        fidelity = []
        drive_param = []
        for jdx, tg_bound in tqdm(enumerate(tg_bounds)):
            if idx==0:
                x0 = None
            else:
                x0 = Drive_param[idx][jdx]

            bounds = (amp_bounds, amp_bounds, tg_bound, detune_bounds, detune_bounds)

            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cnot_2A0,
                bounds=bounds,
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
                x0=x0,
                # constraints=(nonlinear_constraint,)
                )
            fidelity.append(res.fun)
            drive_param.append(res.x)
            print(res, '\n')
            print('fidelity = ', fidelity)
            print('drive_param = ', np.array(drive_param).tolist())
        Fidelity[idx+1] = fidelity
        Drive_param[idx+1] = drive_param
    print('Fidelity = ', Fidelity)
    print('Drive_param = ', Drive_param)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de_x0_repeat():
    amp_bounds = (0, 0.4)
    detune_bounds = (-0.5, 0.)
    workers = 75
    popsize = 50
    mutation = (1.5, 1.99)
    recombination = 0.7
    tol = 0.01
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)

    x0_vec = np.array([
# [ 1.29375677e-01,  4.12452337e-02,  1.29287143e+02,-2.60842903e-01, -2.79434355e-01],
# [ 1.39375677e-01,  4.12452337e-02,  1.29287143e+02,-2.60842903e-01, -2.79434355e-01, -2.26002033e+00],
# [ 1.26599461e-01,  3.43961750e-02,  1.80715714e+02, -2.65980086e-01, -2.79133255e-01,],
#    [ 1.52635845e-01,  9.30792596e-02,  1.80715714e+02, -1.45921117e-01, -1.53642899e-01, -1.46603733e+00],
# [ 1.22295234e-01,  3.22886233e-02,   1.87142856e+02, -2.65458734e-01, -2.76565235e-01],
# [ 1.86138183e-01,  3.74156793e-02,  1.87142856e+02, -2.97170859e-01, -3.29698991e-01, -3.23563],
[ 2.23997351e-01,  4.20319395e-02,  1.99999999e+02, -3.34368079e-01, -3.94603445e-01],
# [ 2.33997351e-01,  4.20319395e-02,  1.99999999e+02, -3.34368079e-01, -3.94603445e-01,  -3.44211]
])
    print('x0_vec=',x0_vec)
    Fidelity = {}
    Drive_param = {}
    repets = 2
    Drive_param[0] = x0_vec
    for idx in range(repets):
        print('\n\n\n***&&&***repeats=%d\n\n\n'%idx)
        fidelity = []
        drive_param = []
        for jdx in tqdm(range(x0_vec.shape[0])):
            x0 = Drive_param[idx][jdx]
            tg_bound = (x0_vec[jdx][2] - 0.000001, x0_vec[jdx][2] + 0.000001)
            bounds = (amp_bounds, amp_bounds, tg_bound, detune_bounds, detune_bounds)

            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cnot_2A0,
                bounds=bounds,
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
                x0=x0,
                # constraints=(nonlinear_constraint,)
                )
            fidelity.append(res.fun)
            drive_param.append(res.x)
            print(res, '\n')
            print('fidelity = ', fidelity)
            print('drive_param = ', np.array(drive_param).tolist())
        Fidelity[idx+1] = fidelity
        Drive_param[idx+1] = drive_param
    print('fidelity = ', Fidelity)
    print('drive_param = ', Drive_param)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)