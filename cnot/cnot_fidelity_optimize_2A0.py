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
# ut.set_fig_font() ### Set various sizes in plotting





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
def fidelity_de(smal_tgbound=True):
    amp_bounds = (0, 0.3)
    detune_bounds = (-0.3, 0.)
    tg_vec = np.linspace(20, 200, num=15)
    tg_bounds = []
    if smal_tgbound:
        for i in range(len(tg_vec)):
            tg_bounds.append((tg_vec[i]-0.00000001, tg_vec[i]))
    else:
        for i in range(len(tg_vec)-1):
            tg_bounds.append((tg_vec[i], tg_vec[i+1]))

    workers = 120
    popsize = 50
    mutation = (0.5, 1.)
    recombination = 0.7
    tol = 0.01
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)

    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)

    fidelity = []
    drive_param = []
    for tg_bound in tqdm(tg_bounds):
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




if __name__ == '__main__':
    args_all = ut.get_operator_two_zeropi()
    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states ] = args_all

    truc = len(trunc_states)
    state_tot = [qt.basis(truc, i) for i in range(truc)]
    # e_ops = [qt.basis(truc, i) * qt.basis(truc, i).dag()
    #         for i in range(truc)]

    [w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
    n_theta2_20_14] = ut.get_transition_freq(args_all)


    n1n1 = True
    if n1n1:
        H_qbt_drive = [ H0, [2*np.pi*n_theta1_truc, ut.drive_gauss_A],
                            [2*np.pi*n_theta1_truc, ut.drive_gauss_B]  ]
    else:
        H_qbt_drive = [ H0, [2*np.pi*n_theta2_truc, ut.drive_gauss_A],
                            [2*np.pi*n_theta1_truc, ut.drive_gauss_B]  ]
    print('n1n1=', n1n1)

    args = (w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
            n_theta2_20_14, state_tot, H_qbt_drive, n1n1)
    start_time = time.time()

    fidelity_noise_sweep()

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_hours = elapsed_time_seconds / 3600
    print(f"Elapsed time: {elapsed_time_hours:.2f} hours")





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

