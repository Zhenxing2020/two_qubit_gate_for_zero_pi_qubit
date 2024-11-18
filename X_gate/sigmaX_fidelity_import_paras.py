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


    para_tot = np.array([
[45, 0.154128, 0.091293, -0.295173, -0.270381]
        ])
    print('para_tot =')
    for para in para_tot:
        print(para.tolist())

    drag = 0
    print('\nmode of DRAG (0 is no drag, 1 is one drag) =',drag)

    truc1 = 300
    n_job = len(para_tot)
    n_cpu = 30
    drive_phi = False
    drive_theta = True
    print('truc1=', truc1, ', n_job=', n_job, ', n_cpu=', n_cpu)
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)

    ## print hilbert space
    print('hilbert space: num of truncation =', len(hspace_charge))
    # dim = 10
    # data = hspace_charge
    # for i in range(0, len(data), dim):
    #     print(', '.join(map(str, np.round(data[i:i+dim], 8) )), ',')

    ### parallel sweep
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, drag]
    f_theta = Parallel(n_jobs=n_job, verbose=5)(delayed(ut.xgate_fidelity_parallel)(arg, args)
                                                for arg in para_tot)

    ### print result
    print('\nlog of gate error (truc1=300) = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')

    fidelity_real = 1-10**np.array(f_theta)
    print('\nfidelity = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

    ############################################################
    truc1 = 80
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, drag]
    f_80 = Parallel(n_jobs=n_job, verbose=5)(delayed(ut.xgate_fidelity_parallel)(arg, args)
                                                for arg in para_tot)
    ### print result
    print('\nlog of gate error (truc1=80) = ')
    for i in range(0, len(f_80), 4):
        print(', '.join(map(str, np.round(f_80[i:i+4], 8))), ',')



    ############################################################
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
