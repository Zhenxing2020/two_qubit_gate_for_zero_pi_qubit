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
print(os.path.basename(__file__)) # Print the name of the current Python file
print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
############################################################
# f_xgate = pd.read_csv('data/data_fig/data_xgate_theta_truc1=50_truc2=29.txt')
f_xgate = pd.read_csv('data/data_fig/data_xgate_theta_truc1=80_truc2=44.txt')
# f_xgate = pd.read_csv('data/data_fig/data_xgate_phi+theta_truc1=40_truc2=40.txt')
para_tot = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()

truc1 = 50
n_job = 16
n_cpu = 8
drive_phi = False
drive_theta = True
print('truc1=', truc1, ', n_job=', n_job, ', n_cpu=', n_cpu)
print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
[H0, drive_term, w_trans_1, w_trans_2] = ut.zero_pi_intialize(drive_phi, drive_theta, truncation=truc1,)

## find hilbert space
thresh = 0.01  
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
dim = 10
data = hilbert_space
for i in range(0, len(data), dim):
    print(', '.join(map(str, np.round(data[i:i+dim], 8) )), ',')


### parallel sweep
args = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu]
f_theta = Parallel(n_jobs=n_job, verbose=5)(delayed(ut.xgate_fidelity_arg5)(arg, args)
                                            for arg in para_tot)

### print result
print('\nlog of gate error = ')
for i in range(0, len(f_theta), 4):
    print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')

fidelity_real = 1-10**np.array(f_theta)
print('\nfidelity = ')
for i in range(0, len(f_theta), 4):
    print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')


############################################################
print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
