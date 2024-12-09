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
def fidelity_de(argss):
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = argss
    fidelity = []
    drive_param = []
    f300 = []
    # hspace_4 = [0,2,7,25]
    for jdx, tg in tqdm(enumerate(tg_vec)):
    # for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
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
            # x0=x0_vec[jdx,3:],
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        ### print fidelity for truc=44
        print(res, '\n')
        # print('\ntg = ', np.array((x0_vec[:,0])[:jdx+1]).tolist())

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
        # if drag == 0:
        #     [alpha_A, alpha_B] = [0, 0]
        #     [drive_amp_A, drive_amp_B, detune_A, detune_B] = drive_param[jdx]
        # elif drag == 1:
        #     alpha_B = 0
        #     [drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = drive_param[jdx]
        # else:
        #     [drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = drive_param[jdx]
        # n_cpu = 30
        # argz = [H0_300, drive_300, w_trans_1, w_trans_2, hspace_300, n_cpu, tg,
        #         drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
        # f300.append(ut.xgate_fidelity(argz))
        # print('\nlog of gate error (truc2=157) = ')
        # for i in range(0, len(f300), 4):
        #     print(', '.join(map(str, np.round(f300[i:i+4], 8))), ',')

        print("\n***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    drive_phi, drive_theta, drag = False, True, 0
    amp_bounds, detune_bounds, alpha_bounds = [(0, 0.2), (-0.4, 0.1), (-20, 20)]
    tg_vec = [10] # + np.arange(40, 62.5, step=2.5).tolist()
    # tg_vec = [37.5, 42.5, 47.5, 52.5 , 57.5, 62.5]
    # x0_vec = np.array([
    # [45.0, -2.45388, -1.532119, 0.154128, 0.091293, -0.295173, -0.270381] ,
# [ 20.      ,  -0.840893,  -0.297285,   0.236014,   0.210179,
#           0.325496,   0.377853],
    #    [ 25.      ,  -1.311848,  -0.345414,   0.221721,   0.211883,
    #       0.329296,   0.371963],
    #    [ 30.      ,  -1.786595,  -0.391148,   0.213381,   0.20848 ,
    #       0.33484 ,   0.370759],
    #    [ 35.      ,  -2.165116,  -0.440385,   0.23764 ,   0.174945,
    #       0.323456,   0.383149],
    #    [ 40.      ,  -2.449442,  -0.525286,   0.198725,   0.196598,
    #       0.322698,   0.35027 ],
    #    [ 45.      ,  -2.831378,  -0.703973,   0.191642,   0.193185,
    #       0.312935,   0.337679],
    #    [ 50.      ,  -2.79914 ,  -0.348502,   0.229852,   0.152328,
    #       0.301876,   0.364361],
    #    [ 55.      ,  -2.82329 ,  -0.432485,   0.225624,   0.147674,
    #       0.300016,   0.3569  ],
    #    [ 75.      ,  -3.63606 ,  -0.301651,   0.171833,   0.172007,
    #       0.284062,   0.299904],
    #    [100.      ,  -5.627182,  -0.506612,   0.202817,   0.123664,
    #       0.275593,   0.31813 ]
    # ])


    workers, popsize = 70, 20
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    truc1 = 80
    print('drive_phi=', drive_phi, ', drive_theta = ', drive_theta, ', Drag=', drag)
    print('amp_bounds=',amp_bounds,', detune_bounds=',detune_bounds, ', alpha_bounds=',alpha_bounds)
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
        bounds = (amp_bounds, amp_bounds, detune_bounds, detune_bounds)
    else:
        bounds = (amp_bounds, amp_bounds, detune_bounds, detune_bounds, alpha_bounds)


    ### optimize
    argss = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge]
    fidelity_de(argss)

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



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