import sys
sys.path.append('../')

import numpy as np
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import utils_2Q_gate_zp as ut
import qutip as qt
import scqubits as scq
from datetime import datetime
import os
import pytz
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
    ############################################################
    # f_xgate = pd.read_csv('data/data_fig/data_xgate_theta_truc1=50_truc2=29.txt')
    # f_xgate = pd.read_csv('data/data_fig/data_xgate_theta_truc1=80_truc2=44.txt')
    # f_xgate = pd.read_csv('data/data_fig/data_xgate_phi+theta_truc1=40_truc2=40.txt')
    # para_tot = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()

#     tg_vec = np.array([40, 50, 60, 45, 55, 70, 80])
#     para_tot = np.array([
#  0.081606, 0.106285, -0.299703, -0.247238
#     ])
#     para_tot = np.column_stack((tg_vec, para_tot))

    truc1 = 500
    n_cpu = 30
    drive_phi = True
    drive_theta = False
    drag = 0
    para_tot = np.array([
        [20,0.236014,0.210179,0.325496,0.377853],
        [25,0.221721,0.211883,0.329296,0.371963],
        [30,0.213381,0.20848,0.33484,0.370759],
        [35,0.23764,0.174945,0.323456,0.383149],
        ])
    n_job = len(para_tot)
    print('truc1=', truc1, ', n_job=', n_job, ', n_cpu=', n_cpu)
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
    print('\nmode of DRAG (0 is no drag, 1 is one drag) =',drag)
    print('para_tot =')
    for para in para_tot:
        print(para.tolist())
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)
    print('hilbert space: num of truncation =', len(hspace_charge))



    ### parallel sweep
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, drag]
    f_theta = Parallel(n_jobs=n_job, verbose=5)(delayed(ut.xgate_fidelity_parallel)(arg, args)
                                                for arg in para_tot)

    ### print result
    print('\nlog of gate error (truc=500) = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')

    fidelity_real = 1-10**np.array(f_theta)
    print('\nfidelity = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

    ############################################################
    truc1 = 300
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, drag]
    f_80 = Parallel(n_jobs=n_job, verbose=5)(delayed(ut.xgate_fidelity_parallel)(arg, args)
                                                for arg in para_tot)
    ### print result
    print('\nlog of gate error (truc=300) = ')
    for i in range(0, len(f_80), 4):
        print(', '.join(map(str, np.round(f_80[i:i+4], 8))), ',')



    ############################################################
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
