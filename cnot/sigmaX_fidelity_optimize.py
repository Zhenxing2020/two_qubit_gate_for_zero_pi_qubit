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
import time
import utils_2Q_gate_zp as ut
import os
from datetime import datetime
import pytz


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de(tg_vec, argss):
    (w_trans_1, w_trans_2, states, H_qbt_drive) = argss


    recombination = 0.7
    tol = 0.01
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    bounds = (amp_bounds, amp_bounds, detune_bounds, detune_bounds)

    fidelity = []
    drive_param = []
    for jdx, tg in tqdm(enumerate(tg_vec)):
        args = (tg, w_trans_1, w_trans_2, states, H_qbt_drive)

        res = sp.optimize.differential_evolution(
            func=ut.get_fidelity_x,
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
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)
        print(res, '\n')
        print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        print('\nfidelity = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')
        print('\ndrive_param = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
    EL        = 0.377 # GHz
    EJ        = 6.013 # Soft Zero Pi (Gyenis)
    EC_phi    = 1.142
    EC_theta  = 0.092
    E_CJ = 2 * EC_phi
    E_C = 2./(1./EC_theta -1./EC_phi)

    truc = 40
    workers = 100
    popsize = 50
    amp_bounds = (0, 0.3)
    detune_bounds = (-0.1, 0.1)
    mutation = (0.5, 1.99)

    drive_phi = True
    drive_theta = True

    thresh = 0.01
    hilbert_space = [0, 2]
    print('truc = ',truc,
          ';   drive_phi = ', drive_phi,
          ';   drive_theta = ', drive_theta)

    phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, 100)
    zero_pi = scq.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=E_CJ, EC = E_C, dEJ=0.,
                            ng=0., flux=0., ncut=30, truncated_dim=truc)
    n_Theta = zero_pi.matrixelement_table(operator='n_theta_operator', evals_count=truc)
    n_Phi = zero_pi.matrixelement_table(operator='i_d_dphi_operator', evals_count=truc)
    n_phi = 2*np.pi * qt.Qobj(n_Phi)
    n_theta = 2*np.pi * qt.Qobj(n_Theta)

    evals = 2*np.pi * zero_pi.eigenvals(evals_count=truc)
    H0 = qt.Qobj(np.diag(evals))

    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta
    if drive_phi and drive_theta:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = 0.976*n_phi+ 0.024*n_theta

    for s in hilbert_space:
        for i in range(truc):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hilbert_space:
                hilbert_space.append(i)
    hilbert_space.sort()
    print('hilbert_space :', len(hilbert_space), hilbert_space)

    truc2 = len(hilbert_space)
    states = [qt.basis(truc2, i) for i in range(truc2)]

    H0 = ut.truncate_2(H0, hilbert_space)
    drive_term = ut.truncate_2(drive_term, hilbert_space)
    H_qbt_drive = [H0,  [drive_term, ut.drive_gauss_A],
                        [drive_term, ut.drive_gauss_B],]

    tg_vec = np.arange(22, 82, step=4).tolist()
    argss = (w_trans_1, w_trans_2, states, H_qbt_drive)
    print('tg_vec = ')
    for i in range(0, len(tg_vec), 4):
        print(', '.join(map(str, tg_vec[i:i+4])), ',')

    fidelity_de(tg_vec, argss)

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
