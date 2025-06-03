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
import utils_two_qubit_gate as ut
ut.set_fig_font() ### Set various sizes in plotting 


if __name__ == '__main__':
    Ec0 = 1.0
    tg = 112
    drive_ab = False
    state0 = '20'
    state1 = '50'
    one_drive=False
    args = (Ec0, tg, drive_ab, state0, state1, one_drive)

    ####################################################################
    ### test
    ####################################################################
    # detune_A = 0.
    # drive_amp_A = 0.
    # detune_B = 0.094
    # drive_amp_B = 0.0089
    # args_indep = (detune_A, drive_amp_A, detune_B, drive_amp_B)
    # fidelity = ut.get_fidelity(args_indep, *args)
    # print('Ec0:', Ec0)
    # print('gate time:', tg,'(ns)')
    # print('detune: ', detune)
    # print('drive amp: ', drive_amp)
    # print('fidelity=',fidelity)


    ####################################################################
    ### Optimize
    ####################################################################
    # detune_bounds = (0, 0.1)
    # amp_bounds = (0, 0.02)
    # args_indep = (detune_bounds, amp_bounds, detune_bounds, amp_bounds)
    # res2 = sp.optimize.differential_evolution(ut.get_fidelity, args_indep,
    #                                             disp=True, callback=ut.print_soln, popsize=30, workers=100,
    #                                             # init="halton",
    #                                             args = args,
    #                                             polish=False, mutation = (0.1, 1.5)  )
    # print(res2)

    
    ####################################################################
    ### Optimize while sweeping tg
    ####################################################################
    detune_bounds = (-0.1, 0.1)
    amp_bounds = (0, 0.02)
    args_indep = (detune_bounds, amp_bounds, detune_bounds, amp_bounds)
    tg_vec = np.linspace(50, 300, num=21)
    fidelity = []
    drive_param = []
    for tg in tg_vec:
        args = (Ec0, tg, drive_ab, state0, state1, one_drive)
        res = sp.optimize.differential_evolution(ut.get_fidelity, args_indep,
                                                    disp=True, callback=ut.print_soln, popsize=30, workers=100,
                                                    # init="halton",
                                                    args = args,
                                                    polish=False, mutation = (0.1, 1.5)  )
        print('\nGate time:', tg)
        print(res)
        fidelity.append(res.fun)
        drive_param.append(res.x)
    print('gate time:\n', tg_vec)
    print('fidelity:\n', fidelity)
    print('drive_param:\n', drive_param)

















