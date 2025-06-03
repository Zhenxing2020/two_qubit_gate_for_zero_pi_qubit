import sys
sys.path.append('../')

import  os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


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
# ut.set_fig_font() ### Set various sizes in plotting

###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def calculate_fidelity_de_sweep(smal_tgbound=True):
    amp_bounds = (0, 0.04)
    detune_bounds = (-0.3, 0.)
    tg_vec = np.linspace(100, 800, num=26)
    tg_bounds = []
    for i in range(len(tg_vec)-1):
        if smal_tgbound == True:
            tg_bounds.append((tg_vec[i+1]-0.01, tg_vec[i+1]))
        else:
            tg_bounds.append((tg_vec[i], tg_vec[i+1]))

    workers = 70
    popsize = 50
    mutation = (0.5, 1.)
    recombination = 0.7
    tol = 0.01
    x0_vec =   [[0.02998763398547402, 129.060364854945, -0.009524334449526474],
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
 [0.005718650638369904, 787.4597105231765, -0.00038813713923993576]]

    x0_vec = np.array(x0_vec)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)

    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    # print('x0=',x0)

    result_opt = []
    fidelity = []
    drive_param = []
    for tg_bound in tqdm(tg_bounds):
        x0 = None
        for i in range(x0_vec.shape[0]):
            if x0_vec[i,2] > tg_bound[0] and x0_vec[i,2] < tg_bound[1]:
                x0 = x0_vec[i]

        bounds = (amp_bounds, tg_bound, detune_bounds)
        res = sp.optimize.differential_evolution(
            func=ut.get_fidelity_cnot_1A0,
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

        print(res, '\n')
        fidelity.append(res.fun)
        drive_param.append(res.x)
        print('fidelity = ', fidelity)
        print('drive_param = ', np.array(drive_param).tolist())




###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def calculate_fidelity_de_sweep_x0(smal_tgbound=True):
    amp_bounds = (0, 0.04)
    detune_bounds = (-0.3, 0.)
    tg_vec = np.linspace(100, 800, num=26)
    tg_bounds = []
    # for i in range(len(tg_vec)-1):
    #     if smal_tgbound == True:
    #         tg_bounds.append((tg_vec[i+1]-0.01, tg_vec[i+1]))
    #     else:
    #         tg_bounds.append((tg_vec[i], tg_vec[i+1]))

    workers = 150
    popsize = 50
    mutation = (0.5, 1.)
    recombination = 0.7
    tol = 0.01
    x0_vec =   [[0.02998763398547402, 129.060364854945, -0.009524334449526474],
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
 [0.005718650638369904, 787.4597105231765, -0.00038813713923993576]]

    x0_vec = np.array(x0_vec)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)
    print('tg_bounds=',tg_bounds)

    print('\nworkers=',workers)
    print('popsize=',popsize)
    print('mutation=',mutation)
    print('recombination=',recombination)
    print('tol=',tol)
    # print('x0=',x0)

    result_opt = []
    fidelity = []
    drive_param = []
    for i in tqdm(range(x0_vec.shape[0])):
        # if x0_vec[i,1] > tg_bound[0] and x0_vec[i,1] < tg_bound[1]:
        x0 = x0_vec[i]
        tg_bound = (x0_vec[i][1] - 0.000001, x0_vec[i][1] + 0.000001)

        bounds = (amp_bounds, tg_bound, detune_bounds)
        res = sp.optimize.differential_evolution(
            func=ut.get_fidelity_cnot_1A0,
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

        print(res, '\n')
        fidelity.append(res.fun)
        drive_param.append(res.x)
        print('fidelity = ', fidelity)
        print('drive_param = ', np.array(drive_param).tolist())


###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_dual_anneal():
    amp_bounds = (0, 0.04)
    detune_bounds = (-0.3, 0.)
    x0_vec = np.array([[0.02998763398547402, 129.060364854945, -0.009524334449526474]])

    result_opt = []
    fidelity = []
    drive_param = []
    for i in tqdm(range(x0_vec.shape[0])):
        x0 = x0_vec[i]
        tg_bound = (x0_vec[i][1] - 0.000001, x0_vec[i][1] + 0.000001)

        bounds = (amp_bounds, tg_bound, detune_bounds)
        res = sp.optimize.dual_annealing(
            func=ut.get_fidelity_cnot_1A0,
            bounds=bounds,
            args=args,
            x0=x0
            )

        print(res, '\n')
        fidelity.append(res.fun)
        drive_param.append(res.x)
        print('fidelity = ', fidelity)
        print('drive_param = ', np.array(drive_param).tolist())



###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_shgo():
    amp_bounds = (0, 0.04)
    detune_bounds = (-0.3, 0.)
    x0_vec = np.array([[0.02998763398547402, 129.060364854945, -0.009524334449526474]])
    options = {'disp': True}
    result_opt = []
    fidelity = []
    drive_param = []
    for i in tqdm(range(x0_vec.shape[0])):
        tg_bound = (x0_vec[i][1] - 0.000001, x0_vec[i][1] + 0.000001)

        bounds = (amp_bounds, tg_bound, detune_bounds)
        res = sp.optimize.shgo(
            func=ut.get_fidelity_cnot_1A0,
            bounds=bounds,
            args=args,
            workers=10,
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
def fidelity_de_repeat(smal_tgbound=True):
    amp_bounds = (0, 0.3)
    detune_bounds = (-0.3, 0.)
    tg_vec = np.linspace(77.85, 116.43, num=4)
    tg_bounds = []
    if smal_tgbound:
        for i in range(len(tg_vec)):
            tg_bounds.append((tg_vec[i]-0.00001, tg_vec[i]))
    else:
        for i in range(len(tg_vec)-1):
            tg_bounds.append((tg_vec[i], tg_vec[i+1]))

    workers = 75
    popsize = 50
    mutation = (1., 1.99)
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


    Fidelity = {}
    Drive_param = {}
    repets = 10
    for idx in range(repets):
        print('\n\n\n***&&&***repeats=%d\n\n\n'%idx)
        fidelity = []
        drive_param = []
        for jdx, tg_bound in tqdm(enumerate(tg_bounds)):
            if idx==0:
                x0 = None  # [0.02998763398547402, tg_vec[jdx]-0.00000001, -0.009524334449526474]
            else:
                x0 = Drive_param[idx][jdx]

            bounds = (amp_bounds, tg_bound, detune_bounds)
            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cnot_1A0,
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
def fidelity_de_x0_repeat(smal_tgbound=True):
    amp_bounds = (0, 0.5)
    detune_bounds = (-0.5, 0.)
    workers = 75
    popsize = 50
    mutation = (1.8, 1.99)
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
[0.3304264822054793, 19.999997129222958, -0.2999376576562004],
[0.05829400492926, 26.429992967437347, -0.2999994700575538],
[0.02587163786523947, 32.857139167618385, -0.24406692231805796],
[0.0223306136277554, 39.287137774489146, -0.24301204900771484]
])
# -0.28700771815538834 [0.23304264822054793, 19.999997129222958, -0.2999376576562004]
# -0.32892950681327915 [0.175829400492926, 26.429992967437347, -0.2999994700575538]
# -0.36673519976530283 [0.07587163786523947, 32.857139167618385, -0.24406692231805796]
# -0.45868003422217696 [0.0723306136277554, 39.287137774489146, -0.24301204900771484]
    print('x0_vec = ', x0_vec)
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
            tg_bound = (x0_vec[jdx][1] - 0.000001, x0_vec[jdx][1] + 0.000001)

            bounds = (amp_bounds, tg_bound, detune_bounds)
            res = sp.optimize.differential_evolution(
                func=ut.get_fidelity_cnot_1A0,
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
        Fidelity[idx+1] = fidelity
        Drive_param[idx+1] = drive_param
    print('Fidelity = ', Fidelity)
    print('Drive_param = ', Drive_param)
    print('\namp_bounds=',amp_bounds)
    print('detune_bounds=',detune_bounds)


if __name__ == '__main__':
    args_all = ut.get_operator_two_zeropi()
    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states ] = args_all

    truc = len(trunc_states)
    state_tot = [qt.basis(truc, i) for i in range(truc)]

    [w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
    n_theta2_20_14] = ut.get_transition_freq(args_all)

    n1n1 = True
    if n1n1 == True:
        H_qbt_drive = [ H0, [2*np.pi*n_theta1_truc, ut.drive_gauss_A],
                            [2*np.pi*n_theta1_truc, ut.drive_gauss_B]  ]
    else:
        H_qbt_drive = [ H0, [2*np.pi*n_theta2_truc, ut.drive_gauss_A],
                            [2*np.pi*n_theta1_truc, ut.drive_gauss_B]  ]
    print('n1n1=', n1n1)

    args = (w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
            n_theta2_20_14, state_tot, H_qbt_drive, n1n1)


    fidelity_de_x0_repeat()








# ###################################################################
# ## Optimize fidelity with differential evolution
# ###################################################################
# def calculate_fidelity_de():
#     amp_bounds = (0., 0.03)
#     tg_bounds = (250, 350)
#     detune_bounds = (-0.001, 0.)
#     bounds = (amp_bounds, tg_bounds, detune_bounds)


#     workers = 100
#     popsize = 50
#     mutation = (0.5, 1.)
#     recombination = 0.7
#     tol = 0.01
#     x0 = [ 2.26098334e-02,  3.30785266e+02, -2.57668268e-04]
#     # print('gate_time:', tg)
#     print('\npopsize=',popsize)
#     print('mutation=',mutation)
#     print('recombination=',recombination)
#     print('tol=',tol)
#     print('\namp_bounds=',amp_bounds)
#     print('tg_bounds=',tg_bounds)
#     print('detune_bounds=',detune_bounds)

#     res = sp.optimize.differential_evolution(
#             func=ut.get_fidelity_raman,
#             bounds=bounds,
#             args=args,
#             disp=True,
#             callback=ut.print_soln,
#             workers=workers,
#             init="sobol",
#             popsize=popsize,
#             mutation=mutation,
#             recombination=recombination,
#             tol=tol,
#             x0=x0,
#             polish=False, # 'True' will make the for-loop break
#             )
#     print(res)