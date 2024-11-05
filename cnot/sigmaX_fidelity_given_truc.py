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
print(os.path.basename(__file__)) # Print the name of the current Python file
print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

EL        = 0.377 # GHz
EJ        = 6.013 # Soft Zero Pi (Gyenis)
EC_phi    = 1.142
EC_theta  = 0.092
E_CJ = 2 * EC_phi
E_C = 2./(1./EC_theta -1./EC_phi)
dEJ = 0.
ng = 0.
vmax=0.1
thresh = 0.01
phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, 100)

############################################################
drive_phi = False
drive_theta = True
tg, drive_amp_A, drive_amp_B, detune_A, detune_B = [20, 0.239886, 0.0551455, -0.0051449, -0.00191928] # truc=50
print('drive_phi = ', drive_phi, ';   drive_theta = ', drive_theta)

truc_vec = [300, 400, 500]
fidelity = []
for idx, truc in tqdm(enumerate(truc_vec)):
    zero_pi = scq.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=E_CJ, EC = E_C, dEJ=dEJ,
                        ng=ng, flux=0., ncut=30, truncated_dim=truc)
    n_Theta = zero_pi.matrixelement_table(operator='n_theta_operator', evals_count=truc)
    n_Phi = zero_pi.matrixelement_table(operator='i_d_dphi_operator', evals_count=truc)
    evals = 2*np.pi * zero_pi.eigenvals(evals_count=truc)
    H0 = qt.Qobj(np.diag(evals))
    n_phi = 2*np.pi * qt.Qobj(n_Phi)
    n_theta = 2*np.pi * qt.Qobj(n_Theta)
    states = [qt.basis(truc, i) for i in range(truc)]

    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta
    if drive_phi and drive_theta:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = 0.976*n_phi+ 0.024*n_theta
    argz = [H0, drive_term, tg, w_trans_1, w_trans_2, drive_amp_A, drive_amp_B, detune_A, detune_B]
    hilbert_space = np.arange(truc)

    fidelity.append(ut.xgate_fidelity_given_hspace(hilbert_space, argz))

    print('\ntruc = ', np.array(truc_vec[:idx+1]).tolist())
    print('\nfidelity = ')
    for i in range(0, len(fidelity), 4):
        print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
