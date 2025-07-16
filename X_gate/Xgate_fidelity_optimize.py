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
import os, pytz
from datetime import datetime
import pandas as pd

###################################################################
## Optimize fidelity with differential evolution
###################################################################
def fidelity_de(max_int=0.1):
    fidelity = []
    drive_param = []
    fidelity_full = []
    n_cpu = 1
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, max_int]
    for jdx, tg in tqdm(enumerate(tg_vec)):
    # for jdx, tg in tqdm(enumerate(x0_vec[:,0])): # if there is x0
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
        print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        # print('\ntg = ', np.array((x0_vec[:,0])[:jdx+1]).tolist())
        print(f'\nlog of gate error (truc1={len(hspace_charge)}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')
        print(f'\ndrive_param (truc1={len(hspace_charge)}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        ## get fidelity of optimal point in full system
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = drive_param[jdx]
        n_cpu2 = 30
        argz = [H0_full, drive_full, w_trans_1, w_trans_2, hspace_full, n_cpu2,
                tg, drive_amp_A, drive_amp_B, detune_A, detune_B]
        fidelity_full.append(ut.xgate_fidelity_log(argz))
        print(f'\nlog of gate error (truc_full={truc_full}) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')

        para_tot = np.column_stack((drive_param, fidelity_full))
        for i in para_tot:
            print(np.round(i,6).tolist(),',')

        print('\namp1_bounds=',amp1_bounds, ', amp2_bounds=',amp2_bounds,)
        print('detune1_bounds=',detune1_bounds, ', detune2_bounds=', detune2_bounds)
        print("***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # drive_phi, drive_theta, drive_0  = True, False, False
    drive_phi, drive_theta, drive_0  = False, True, True
    if drive_theta:
        # amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.01, 0.05), (0.025, 0.2), (-0.17, -0.2), (0.1, 0.3)] # theta big
        tg_vec = [50] # theta
        amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.00, 0.2), (0.00, 0.2), (0, 0.2), (0, 0.2)] # theta small
        # tg_vec = np.arange(40, 150, step=10).tolist() # theta
    else:
        amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.15, 0.3), (0, 0.3), (0.3, 0.5), (0.3, 0.5)] # phi
        tg_vec = np.arange(10, 50, step=10).tolist() # phi

    tg_bound = (-1, 1)
    
    folder = 'data_xgate_theta_3ncut.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('../../data/'+folder)
    x0_vec = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()[[8],:] #[[10,8,6,4,2],:]
    # breakpoint()

    workers, popsize = 1, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    truc1, truc_full = 150, 500 # theta
    print('drive_phi=', drive_phi, ', drive_theta = ', drive_theta, ', drive_0 = ', drive_0)
    print('amp1_bounds=',amp1_bounds, ', amp2_bounds=',amp2_bounds, ', tg_bound=', tg_bound)
    print('detune1_bounds=',detune1_bounds, ', detune2_bounds=', detune2_bounds)
    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)
    if 'x0_vec' in globals():
        print('params = ')
        for i in x0_vec:
            print(np.round(i,6).tolist(),',')
    if 'tg_vec' in globals():
        print('tg_vec = ')
        for i in range(0, len(tg_vec), 4):
            print(', '.join(map(str, tg_vec[i:i+4])), ',')

#####################################################################
    truncation=1000
    folder = '../../data/3ncut_one_zeropi/'
    if drive_0:
        evals = 2*np.pi* scq.read(folder + f'zeropi_0_specdata_truc={truncation}_3ncut.h5').energy_table
        n_Theta = 2*np.pi* scq.read(folder + f'zeropi_0_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_Phi = 2*np.pi* scq.read(folder + f'zeropi_0_n_phi_truc={truncation}_3ncut.h5').matrixelem_table
    else:
        evals = 2*np.pi* scq.read(folder + f'zeropi_1_specdata_truc={truncation}_3ncut.h5').energy_table
        n_Theta = 2*np.pi* scq.read(folder + f'zeropi_1_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_Phi = 2*np.pi* scq.read(folder + f'zeropi_1_n_phi_truc={truncation}_3ncut.h5').matrixelem_table
    evals = evals - evals[0]

    H0 = qt.Qobj(np.diag(evals[:truc1]))
    H0_full = qt.Qobj(np.diag(evals[:truc_full]))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_Phi[:truc1, :truc1]
        drive_full = n_Phi[:truc_full, :truc_full]
    if drive_theta:
        w_trans_1 = evals[1] - evals[0]
        w_trans_2 = evals[2] - evals[1]
        # w_trans_1 = evals[7] - evals[0]
        # w_trans_2 = evals[7] - evals[2]
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
    hspace_full = np.arange(truc_full).tolist()
    max_int = 0.2
    print('truc1 =', truc1, ', truc2 (in optimization) =', len(hspace_charge))
    print('truc1_full =', truc_full, ', truc2_full =', len(hspace_full))
    print('Max intermediate state population =', 0.1)

    ## TEST
    n_cpu = 1
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, max_int]
    arg = x0_vec[0]
    res = ut.xgate_fidelity_parallel(arg, *args)
    breakpoint()

    ### optimize
    # fidelity_de(max_int=max_int)




