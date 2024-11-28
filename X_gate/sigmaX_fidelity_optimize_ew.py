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
def fidelity_de(argss, x0_vec):
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, drag, bounds] = argss
    fidelity = []
    drive_param = []
    f300 = []
    # hspace_4 = [0,2,7,25]
    for jdx, tg_base in tqdm(enumerate(x0_vec[:,0])):
        args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, tg_base, drag]
        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_optimize,
            bounds=bounds,
            args=args,
            disp=True,
            callback=ut.print_soln,
            init="halton",
            workers=workers,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            x0=x0_vec[jdx,3:],
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        ### print fidelity for truc=44
        print(res, '\n')
        print('\ntg = ', np.array((x0_vec[:,0])[:jdx+1]).tolist())

        print('\nlog of gate error (truc2=44) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

        fidelity_real = 1-10**np.array(fidelity)
        print('\nfidelity =  (truc2=44) ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

        print('\ndrive_param = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        ## get fidelity for truc=157
        if drag == 0:
            [alpha_A, alpha_B] = [0, 0]
            [tg_mod, drive_amp_A, drive_amp_B, detune_A, detune_B] = drive_param[jdx]
        elif drag == 1:
            alpha_B = 0
            [tg_mod, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = drive_param[jdx]
        else:
            [tg_mod, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = drive_param[jdx]
        n_cpu = workers
        argz = [H0_300, drive_300, w_trans_1, w_trans_2, hspace_300, n_cpu, tg_base+tg_mod,
                drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
        f300.append(ut.xgate_fidelity(argz))
        print('\nlog of gate error (truc2=157) = ')
        for i in range(0, len(f300), 4):
            print(', '.join(map(str, np.round(f300[i:i+4], 8))), ',')

        print("\n***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    drive_phi, drive_theta, drag = True, False, 1
    tg_bounds, amp_bounds, detune_bounds, alpha_bounds = [(-2.5, 2.5), (0, 2.0), (-0.5, 0.5), (-20, 20)]
    # tg_vec = [20] # + np.arange(40, 62.5, step=2.5).tolist()
    # tg_vec = [40, 45, 50, 55]
    # tg_vec = [40]
    x0_vec = np.array([
        [25.0,-2.17368,-2.36741531, 0, 0.179494,0.046241,-0.003228,-0.000476,0],
        [30.0,-2.876481,-3.07440094,0, 0.144862,0.038029,0.000865,0.003487,0],
        [35.0,-3.688729,-3.79677893,0, 0.121768,0.032286,0.001859,0.004392,0],
        [40.0,-3.301470,-3.302673,0, 0.105147,0.028068,0.001898,0.00438,0],
        [45.0,-2.453880,-1.532119,0, 0.154128,0.091293,-0.295173,-0.270381,0],
        [50.0,-3.77945,-2.803664,0, 0.10746657,0.10323223,-0.30225871,-0.25967603,0],
        [55.0,-3.052643,-1.884744,0, 0.157689,0.052548,0.034872,0.037391,0],
        [60.0,-3.409958,-3.348597,0,  0.081606,0.106285,-0.299703,-0.247238,0],
        [65.0,-3.36216,-2.18657946,0, 0.139866,0.041224,0.035288,0.0378,0],
        [70.0,-3.442062,-2.453763,0, 0.069316, 0.109747, -0.311983, -0.252449,0],
        [80.0,-3.907725,-3.11743643,0, 0.143017,0.046456,0.050847,0.053381,0],
    ])

    
    workers, popsize = 200, 40
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    truc1 = 80
    print('drive_phi=', drive_phi, ', drive_theta = ', drive_theta, ', Drag=', drag)
    print('tg_bounds', tg_bounds, 'amp_bounds=',amp_bounds,', detune_bounds=',detune_bounds, ', alpha_bounds=',alpha_bounds)
    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)
    if 'x0_vec' in globals():
        print('x0_vec = ')
        for i in x0_vec:
            print(np.round(i,6).tolist(),',')
    if 'tg_vec' in globals():
        print('tg_vec = ')
        for i in range(0, len(tg_vec), 4):
            print(', '.join(map(str, tg_vec[i:i+4])), ',')


    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)
    [H0_300, drive_300, _, _, hspace_300] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=300,)
    print('truncation_1 =', truc1, ', truncation_2 (in optimization) =', len(hspace_charge))

    if drag == 0:
        bounds = (tg_bounds, amp_bounds, amp_bounds, detune_bounds, detune_bounds)
    else:
        bounds = (tg_bounds, amp_bounds, amp_bounds, detune_bounds, detune_bounds, alpha_bounds)


    ### optimize
    argss = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, drag, bounds]
    fidelity_de(argss, x0_vec[4:5])


    #### PROFILING FIDELITY FUNCTION
    # import time

    # [tg_mod, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = x0_vec[3, 3:]
    # tg_base = x0_vec[3, 0]
    # runtimes = []
    # for n_cpu in [1, 2, 4, 8, 16, 32, 44]:
    #     # n_cpu = 44
    #     t0 = time.time()
    #     xgate_fidelity_args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, tg_base,
    #                             drive_amp_A, drive_amp_B, detune_A, detune_B, 0, 0]
    #     # print("fidelity", ut.xgate_fidelity(xgate_fidelity_args))
    #     ut.xgate_fidelity(xgate_fidelity_args)

    #     tf = time.time()

    #     print("n_cpu:", n_cpu, np.round(tf-t0, 1), "s to run")
    

    # print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
# def fidelity_alpha(tg_vec, argss):
#     [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = argss
#     x0_vec = np.array([
#     [0.248363, 0.060381, -0.016372, -0.013767, -0.995504] ,
#     [0.283221, 0.051568, -0.038113, -0.035568, 0.177783] ,
#     [0.209667, 0.045522, -0.025267, -0.022739, -0.666255] ,
#     ])
#     recombination = 0.7
#     tol = 0.01
#     print('x0=', x0_vec)
#     print('\namp_bounds=',amp_bounds)
#     print('detune_bounds=',detune_bounds)
#     print('detune_bounds=',alpha_bounds)
#     print('\nworkers=',workers)
#     print('popsize=',popsize)
#     print('mutation=',mutation)
#     print('recombination=',recombination)
#     print('tol=',tol)

#     d = 1e-18
#     fidelity = []
#     drive_param = []
#     for i, tg in tqdm(enumerate(tg_vec)):
#         x = x0_vec[i]
#         bounds = ((x[0]-d,x[0]+d), (x[1]-d,x[1]+d), (x[2]-d,x[2]+d), (x[3]-d,x[3]+d), alpha_bounds)
#         args = [tg, H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu]
#         res = sp.optimize.differential_evolution(
#             func=ut.xgate_fidelity_optimize,
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
#             x0=x0_vec[i],
#             polish=False, # 'True' will make the for-loop break
#             )
#         fidelity.append(res.fun)
#         drive_param.append(res.x)

#         print(res, '\n')
#         print('\ntg = ', np.array(tg_vec[:i+1]).tolist())

#         print('\nlog of gate error = ')
#         for i in range(0, len(fidelity), 4):
#             print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

#         fidelity_real = 1-10**np.array(fidelity)
#         print('\nfidelity = ')
#         for i in range(0, len(fidelity), 4):
#             print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

#         print('\ndrive_param = ')
#         for i in drive_param:
#             print(np.round(i,6).tolist(),',')

#         print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

#     print('\namp_bounds=',amp_bounds)
#     print('detune_bounds=',detune_bounds)