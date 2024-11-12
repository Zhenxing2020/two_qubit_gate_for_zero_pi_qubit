import sys
sys.path.append('../')

import scqubits as scq
import qutip as qt
import numpy as np
from tqdm import tqdm
import datetime
import pytz
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import scipy as sp
import utils_2Q_gate_zp as ut
import os
from datetime import datetime
import pytz


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de(tg_vec, argss):
    [H0, drive_term, w_trans_1, w_trans_2, hilbert_space] = argss

    x0 = [0.299939, 0.044382, -0.049202, -0.046809]
    recombination = 0.7
    tol = 0.01
    print('x0=', x0)
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
        args = (H0, drive_term, tg, w_trans_1, w_trans_2, hilbert_space)
        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_optimize,
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
            x0=x0,
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

        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


    truc1 = 80
    drive_phi = False
    drive_theta = True
    print('truc=', truc1, '; drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
    workers = 150
    popsize = 50
    amp_bounds = (0, 0.3)
    detune_bounds = (-0.1, 0.1)
    mutation = (0.5, 1.)

    [H0, drive_term, w_trans_1, w_trans_2] = ut.zero_pi_intialize(drive_phi, drive_theta, truncation=truc1,)

    ## find hilbert space
    thresh = 0.25
    hspace_charge = [0, 2]
    for s in hspace_charge:
        for i in range(truc1):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
    hspace_full = np.arange(truc1)


    ## set hilbert space
    hilbert_space = hspace_charge
    print('num of truncation =', len(hilbert_space))
    print('hilbert_space :')
    dim = 10
    data = hilbert_space
    for i in range(0, len(data), dim):
        print(', '.join(map(str, np.round(data[i:i+dim], 8) )), ',')


    ### optimize
    tg_vec = [40] # np.arange(22, 82, step=4).tolist()
    print('tg_vec = ')
    for i in range(0, len(tg_vec), 4):
        print(', '.join(map(str, tg_vec[i:i+4])), ',')
    argss = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space]
    fidelity_de(tg_vec, argss)

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
