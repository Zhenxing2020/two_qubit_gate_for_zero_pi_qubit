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
def fidelity_alpha(tg_vec, argss):
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = argss
    x0_vec = np.array([
    [0.248363, 0.060381, -0.016372, -0.013767, -0.995504] ,
    [0.283221, 0.051568, -0.038113, -0.035568, 0.177783] ,
    [0.209667, 0.045522, -0.025267, -0.022739, -0.666255] ,
    ])
    recombination = 0.7
    tol = 0.01
    print('x0=', x0_vec)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('detune_bounds=',alpha_bounds)
    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)

    d = 1e-18
    fidelity = []
    drive_param = []
    for i, tg in tqdm(enumerate(tg_vec)):
        x = x0_vec[i]
        bounds = ((x[0]-d,x[0]+d), (x[1]-d,x[1]+d), (x[2]-d,x[2]+d), (x[3]-d,x[3]+d), alpha_bounds)
        args = [tg, H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu]
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
            x0=x0_vec[i],
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        print(res, '\n')
        print('\ntg = ', np.array(tg_vec[:i+1]).tolist())

        print('\nlog of gate error = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

        fidelity_real = 1-10**np.array(fidelity)
        print('\nfidelity = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

        print('\ndrive_param = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de(tg_vec, argss):
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = argss
    x0 = [0.299939, 0.044382, -0.049202, -0.046809]

    # print('x0=', x0)
    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)

    fidelity = []
    drive_param = []
    for jdx, tg in tqdm(enumerate(tg_vec)):
        args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, tg, drag]
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
            # x0=x0,
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        print(res, '\n')
        print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())

        print('\nlog of gate error = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

        fidelity_real = 1-10**np.array(fidelity)
        print('\nfidelity = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

        print('\ndrive_param = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')

    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)





if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


    truc1 = 80
    drive_phi = True
    drive_theta = False
    print('truc=', truc1, '; drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)

    workers = 100
    popsize = 20
    mutation = (0.5, 1.9)
    recombination = 0.7
    tol = 0.01

    amp_bounds = (0, 0.3)
    detune_bounds = (-0.4, 0.4)
    alpha_bounds = (-20, 20)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('alpha_bounds=',alpha_bounds)

    drag = 1
    print('\nmode of DRAG (0 is no drag, 1 is one drag) =',drag)
    if drag == 0:
        bounds = (amp_bounds, amp_bounds, detune_bounds, detune_bounds)
    else:
        bounds = (amp_bounds, amp_bounds, detune_bounds, detune_bounds, alpha_bounds)

    ## set hilbert space
    print('num of truncation =', len(hspace_charge))
    # print('hspace_charge :')
    # dim = 10
    # data = hspace_charge
    # for i in range(0, len(data), dim):
    #     print(', '.join(map(str, np.round(data[i:i+dim], 8) )), ',')


    ### optimize
    # tg_vec = [45, 50, 55, 65, 70, 80] + np.arange(40, 62.5, step=2.5).tolist()
    tg_vec = [20.0, 40.0]
    print('tg_vec = ')
    for i in range(0, len(tg_vec), 4):
        print(', '.join(map(str, tg_vec[i:i+4])), ',')
    argss = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge]
    fidelity_de(tg_vec, argss)

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
