import sys
sys.path.append('../')
import scqubits as scq
import qutip as qt
import numpy as np
from tqdm import tqdm
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import scipy as sp
import utils_2Q_gate_zp as ut
import os, pytz, pytz, datetime
from datetime import datetime
import pandas as pd

###################################################################
## Optimize fidelity with differential evolution
###################################################################
def fidelity_de():
    fidelity = []
    drive_param = []
    fidelity_full = []
    fidelity_300 = []
    n_cpu = 1
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, drag]
    # for jdx, tg in tqdm(enumerate(tg_vec)):
    for jdx, tg in tqdm(enumerate(x0_vec[:,0])): # if there is x0
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)
        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_parallel,
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
            x0=x0_vec[jdx],
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        ### print fidelity
        print(res, '\n')
        # print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        print('\ntg = ', np.array((x0_vec[:,0])[:jdx+1]).tolist())
        print(f'\nlog of gate error (truc1={len(hspace_charge)}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

        print(f'\ndrive_param (truc1={len(hspace_charge)}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        ## get fidelity of optimal point in full system
        if drag == 0:
            [alpha_A, alpha_B] = [0, 0]
            [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = drive_param[jdx]
        elif drag == 1:
            alpha_B = 0
            [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = drive_param[jdx]
        else:
            [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = drive_param[jdx]

        n_cpu2 = 30
        argz = [H0_full, drive_full, w_trans_1, w_trans_2, hspace_full, n_cpu2, tg,
                drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
        fidelity_full.append(ut.xgate_fidelity_fast(argz))
        print(f'\nlog of gate error (truc_full={truc_full}) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')

        # truc3 = 300
        # hspace_300 = np.arange(truc3).tolist()
        # argz = [H0_full, drive_full, w_trans_1, w_trans_2, hspace_300, n_cpu2, tg,
        #         drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
        # fidelity_300.append(ut.xgate_fidelity_fast(argz))
        # print(f'\nlog of gate error (truc_full={truc3}) = ')
        # for i in range(0, len(fidelity_300), 4):
        #     print(', '.join(map(str, np.round(fidelity_300[i:i+4], 8))), ',')

        print("\n***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')
    print('amp1_bounds=',amp1_bounds, ', amp2_bounds=',amp2_bounds,)
    print('detune1_bounds=',detune1_bounds, ', detune2_bounds=', detune2_bounds)


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    drive_phi, drive_theta, drag = False, True,  0
    # drive_phi, drive_theta, drag = True, False, 0
    # amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.1, 0.45), (0.025, 0.08), (-0.15, 0.01), (-0.06, 0.01)] # theta small tg
    # amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0, 0.13), (0., 0.035), (-0.0025, 0.0025), (-0.005, 0.005)] # theta large tg
    # amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.2, 0.25), (0.1, 0.225), (0.325, 0.45), (0.35, 0.5)] # phi small
    amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.03, 0.42), (0.008, 0.08), (-0.09, 0.0025), (-0.045, 0.0045)] # phi big
    tg_bound = (-0.01, 0.01)
    # tg_vec = np.arange(10, 35, step=5).tolist() # theta
    # tg_vec = np.arange(25, 125, step=5).tolist() # theta
    # tg_vec = np.arange(20, 120, step=10).tolist() # phi
    # tg_vec = [200] # np.arange(120, 210, step=10).tolist() # phi

    f_xgate = pd.read_csv('data/data_xgate_theta.txt')
    x0_vec = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2',
                        'detune_1', 'detune_2']].to_numpy()

    workers, popsize = 100, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    truc1, truc_full = 300, 500 # theta
    print('drive_phi=', drive_phi, ', drive_theta = ', drive_theta, ', Drag=', drag)
    print('amp1_bounds=',amp1_bounds, ', amp2_bounds=',amp2_bounds, ', tg_bound=', tg_bound)
    print('detune1_bounds=',detune1_bounds, ', detune2_bounds=', detune2_bounds)
    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)
    if 'params' in globals():
        print('params = ')
        for i in params:
            print(np.round(i,6).tolist(),',')
    if 'tg_vec' in globals():
        print('tg_vec = ')
        for i in range(0, len(tg_vec), 4):
            print(', '.join(map(str, tg_vec[i:i+4])), ',')

#####################################################################
    truncation=1000
    folder = 'data/'
    new_specdata = scq.read(folder + f'zeropi_specdata_truc={truncation}_3ncut.h5')
    n_theta = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
    n_phi = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
    evals = 2*np.pi* new_specdata.energy_table
    n_Theta = 2*np.pi* n_theta.matrixelem_table
    n_Phi = 2*np.pi* n_phi.matrixelem_table
    evals = evals - evals[0]

    H0 = qt.Qobj(np.diag(evals[:truc1]))
    H0_full = qt.Qobj(np.diag(evals[:truc_full]))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_Phi[:truc1, :truc1]
        drive_full = n_Phi[:truc_full, :truc_full]
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_Theta[:truc1, :truc1]
        drive_full = n_Theta[:truc_full, :truc_full]

    ## find hilbert space
    thresh = 0.01
    hspace_charge = [0, 2]
    for s in hspace_charge:
        for i in range(truc1):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
#####################################################################


    # [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(
    #     drive_phi, drive_theta, truncation=truc1, ncut=90, phi_cut=300)
    # [H0_full, drive_full, _, _, _] = ut.zero_pi_initialize(
    #     drive_phi, drive_theta, truncation=truc_full, ncut=90, phi_cut=300)
    hspace_full = np.arange(truc_full).tolist()
    print('truc1 =', truc1, ', truc2 (in optimization) =', len(hspace_charge))
    print('truc1_full =', truc_full, ', truc2_full =', len(hspace_full))

    ### optimize
    fidelity_de()




