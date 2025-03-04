import sys
sys.path.append('../')
from datetime import datetime
import pytz, os
import numpy as np
import scipy as sp
from tqdm import tqdm
import scqubits as scq
import scqubits.settings as settings
import qutip as qt
from multiprocessing import Pool
from joblib import Parallel, delayed
import pandas as pd
import utils_2Q_gate_zp as ut
# Update scqubits settings
settings.OVERLAP_THRESHOLD = 0.3
max_step, nsteps = 1e-3, 1e4

def import_para():
    drive_phi, drive_theta = False, True
    # drive_phi, drive_theta =  True, False
    # folder = 'data_xgate_theta.txt' if drive_theta else 'data_xgate_phi.txt'
    # f_xgate = pd.read_csv('data/'+folder)
    # para_tot = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2',
    #                     'detune_1', 'detune_2']].to_numpy()[:3]
    para_tot = np.array([
        [10.108462, 0.41212, 0.073451, -0.084451, -0.044706] ,
        [15.074864, 0.303821, 0.065666, -0.036206, -0.016563] ,
        [20.041257, 0.241329, 0.059044, -0.019402, -0.009802] ,
        [25.020473, 0.180208, 0.046132, -0.003789, 0.001544] ,
    ])

    n_cpu, n_job = 50, 30*len(para_tot)
    print('n_job=', n_job, ', n_cpu=', n_cpu)
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
    print('para_tot =')
    for para in para_tot:
        print(para.tolist(), ',')

#####################################################################
    truncation, truc1 = 1000, 300
    folder = 'data/'
    new_specdata = scq.read(folder + f'zeropi_specdata_truc={truncation}_3ncut.h5')
    n_theta = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
    n_phi = scq.read(folder + f'zeropi_n_phi_truc={truncation}_3ncut.h5')
    evals = 2*np.pi* new_specdata.energy_table
    n_Theta = 2*np.pi* n_theta.matrixelem_table
    n_Phi = 2*np.pi* n_phi.matrixelem_table
    evals = evals - evals[0]

    H0 = qt.Qobj(np.diag(evals[:truc1]))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_Phi[:truc1, :truc1]
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_Theta[:truc1, :truc1]

    ## find hilbert space
    thresh = 0.01
    hspace_charge = [0, 2]
    for s in hspace_charge:
        for i in range(truc1):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
#####################################################################
    hspace = np.arange(truc1).tolist()

    hilbert_space = hspace_charge
    print('\ntruc1_a=', truc1, '; truc1_b=', len(hspace_charge))
    args = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu]
    f_theta = Parallel(n_jobs=n_job, verbose=0)(delayed(xgate_fidelity_parallel)(arg, *args)
                                                for arg in para_tot)
    print(f'log of gate error (truc={len(hilbert_space)}) = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')




def import_para_noise():
    """
    Imports parameters and computes gate fidelities for noisy systems.
    """
    # drive_phi, drive_theta, truc = True, False, 300
    drive_phi, drive_theta, truc = False, True, 150
    drive_0 = True
    folder = 'data_xgate_theta_3ncut.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/'+folder)
    params = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
                      ]].to_numpy()[[-5], :]
    # [[ 1,2, 4,5, 7,8, 10,11, 13,14, 16,17,18],:]  , 0,3,9,12,17,
    #
    num_cpus, n_job = 4, 1*len(params)
    logi_state = [0, 2]

    truncation=1000
    folder = '../../data/3ncut_one_zeropi/'
    if drive_0:
        evals = 2*np.pi* scq.read(folder + f'zeropi_0_specdata_truc={truncation}_3ncut.h5').energy_table
        n_theta = 2*np.pi* scq.read(folder + f'zeropi_0_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_phi = 2*np.pi* scq.read(folder + f'zeropi_0_n_phi_truc={truncation}_3ncut.h5').matrixelem_table
    else:
        evals = 2*np.pi* scq.read(folder + f'zeropi_1_specdata_truc={truncation}_3ncut.h5').energy_table
        n_theta = 2*np.pi* scq.read(folder + f'zeropi_1_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_phi = 2*np.pi* scq.read(folder + f'zeropi_1_n_phi_truc={truncation}_3ncut.h5').matrixelem_table

    evals = evals - evals[0]
    gate_target = qt.sigmax()

    H0 = qt.Qobj(np.diag(evals))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta

    thresh = 0.01
    hspace_charge = [0, 2]  # Start with the ground and first excited states
    for s in hspace_charge:
        for i in range(truc):
            if np.abs(drive_term[s, i] / (2 * np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
    # hspace_charge = np.arange(truc).tolist()

    t1_other = 5 # μs

    tphi_logi = 100 # μs
    gamma_decay_logi =  1 / 1600e3
    gamma_dephase_logi = 1 / 1e3 / tphi_logi
    gamma_decay_other =  1 / 1e3 / t1_other
    gamma_dephase_other = 1 / 1e3 / t1_other
    idx_2 = 2 if drive_theta else 1

    hspace_len = len(hspace_charge)
    if drive_theta:
        gamma_decay_old   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (hspace_len-3)
        gamma_dephase_old = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (hspace_len-3)
    else:
        gamma_decay_old   = [0,  gamma_decay_logi]  + [gamma_decay_other]  * (hspace_len-2)
        gamma_dephase_old = [0,  gamma_dephase_logi] + [gamma_dephase_other] * (hspace_len-2)

    folder_1 = 'data/data_gamma_'
    folder_2 = 'theta.txt' if drive_theta else 'phi.txt'
    gamma_new = pd.read_csv(folder_1 + folder_2)

    ### 't1_50us_47', 't1_50us_27', 't1_50us_07'
    ### 'tphi_50us_02', 'tphi_50us_07', 'tphi_1e6'
    gamma_decay_new = gamma_new['t1_50us_47'].to_numpy()
    gamma_dephase_new = gamma_new['tphi_1e6'].to_numpy()
    # gamma_dephase_new = gamma_new['tphi_50us_02'].to_numpy()

    print("gamma_decay_new[2] = ", gamma_decay_new[2], ", gamma_dephase_new[2] = ", gamma_dephase_new[2])
    gamma_decay_new = gamma_decay_new *50 /t1_other
    gamma_dephase_new = gamma_dephase_new *50 /t1_other
    # gamma_dephase_new[idx_2] = gamma_dephase_logi
    # gamma_decay_new[idx_2] = gamma_decay_logi

    jump_t1   = []
    jump_tphi = []
    for i in range(1,hspace_len):
        jump_t1.append( np.sqrt(gamma_decay_new[i]) * qt.basis(hspace_len,0) * qt.basis(hspace_len,i).dag() )
        jump_tphi.append( np.sqrt(2*gamma_dephase_new[i]) * qt.basis(hspace_len,i).proj() )

    logi_idx = [hspace_charge.index(s) for s in logi_state]
    H0_truc = ut.truncate_2(H0, hspace_charge)
    drive_truc = ut.truncate_2(drive_term, hspace_charge)
    H_qbt_drive = [H0_truc, [drive_truc, ut.drive_gauss_A],
                            [drive_truc, ut.drive_gauss_B],]
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta, '; truc = ', truc)
    print('hspace_len=', hspace_len)
    print('params =')
    for para in params:
        print(para.tolist(), ',')
    print("num_cpus = ", num_cpus, ";   n_job = ", n_job)
    print("gamma_decay_logi = ", gamma_decay_logi, ", gamma_dephase_logi = ", gamma_dephase_logi)
    print("gamma_decay_other = ", gamma_decay_other, ", gamma_dephase_other = ", gamma_dephase_other)
    print(f"T1_logi = {1/gamma_decay_logi} ns") if gamma_decay_logi != 0 else None
    print(f"Tphi_logi = {1/gamma_dephase_logi} ns") if gamma_dephase_logi != 0 else None
    print(f"T1_other = {1/gamma_decay_other} ns") if gamma_decay_other != 0 else None
    print(f"Tphi_other = {1/gamma_dephase_other} ns") if gamma_dephase_other != 0 else None

    ############################################################
    # c_op_list = [qt.Qobj(np.zeros((truc, truc)))]
    # c_op_list = []
    # args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx, gate_target]
    # f_ideal = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args)
    #                                             for args_indep in params)
    # print('\nf_ideal = [')
    # for i in range(0, len(f_ideal), 4):
    #     print(', '.join(map(str, f_ideal[i:i+4])), ',')
    # print(']')
    # print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    ############################################################
    c_op_list = jump_t1 + jump_tphi
    args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx, gate_target]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_noise = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, f_noise[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

if __name__ == '__main__':
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_para()
    import_para_noise()





# # Optimize fidelity with differential evolution
# def fidelity_optimize_x():
#     """
#     Perform optimization of fidelity using differential evolution.

#     The function optimizes gate parameters for high fidelity in a quantum system
#     using the `differential_evolution` method from `scipy.optimize`. It also evaluates
#     fidelity for various truncations of the Hilbert space and prints results.
#     """

#     # Drive parameters
#     drive_phi, drive_theta = False, True

#     # Parameter bounds
#     amp1_bounds = (0.1, 0.45)
#     amp2_bounds = (0.025, 0.08)
#     detune1_bounds = (-0.15, 0.01)
#     detune2_bounds = (-0.06, 0.01)
#     tg_bound = (0, 0.01)

#     # Target gate times
#     tg_vec = np.arange(10, 20, step=5).tolist()

#     # Optimization parameters
#     workers, popsize = 100, 10
#     recombination, tol, mutation = 0.7, 0.01, (0.5, 1.0)

#     # Truncations
#     truc1, truc_full = 30, 50

#     # charge and phase basis
#     ncut, phi_cut = 30, 100

#     print('drive_phi =', drive_phi, ', drive_theta =', drive_theta)
#     print('amp1_bounds =', amp1_bounds, ', amp2_bounds =', amp2_bounds, ', tg_bound =', tg_bound)
#     print('detune1_bounds =', detune1_bounds, ', detune2_bounds =', detune2_bounds)
#     print('workers =', workers, ', popsize =', popsize)
#     print('recombination =', recombination, ', tol =', tol, ', mutation =', mutation)

#     # Initialize system
#     [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = zero_pi_initialize(
#         drive_phi, drive_theta, truncation=truc1, ncut=ncut, phi_cut=phi_cut
#     )
#     [H0_full, drive_full, _, _, _] = zero_pi_initialize(
#         drive_phi, drive_theta, truncation=truc_full, ncut=ncut, phi_cut=phi_cut
#     )
#     hspace_full = np.arange(truc_full).tolist()

#     print('truc1 =', truc1, ', truc2 (in optimization) =', len(hspace_charge))
#     print('truc1_full =', truc_full, ', truc2_full =', len(hspace_full))
# #############################################################

#     fidelity = []
#     drive_param = []
#     fidelity_full = []
#     n_cpu = 1
#     args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu]

#     for jdx, tg in tqdm(enumerate(tg_vec)):
#         tg_bounds = (tg + tg_bound[0], tg + tg_bound[1])
#         bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)

#         # Optimize fidelity using differential evolution
#         res = sp.optimize.differential_evolution(
#             func=xgate_fidelity_parallel,
#             bounds=bounds,
#             args=args,
#             disp=True,
#             callback=print_soln,
#             init="sobol",
#             workers=workers,
#             popsize=popsize,
#             mutation=mutation,
#             recombination=recombination,
#             tol=tol,
#             polish=False,
#         )

#         fidelity.append(res.fun)
#         drive_param.append(res.x)

#         # Print optimization results
#         print(res, '\n')
#         print(f'\ntg = {np.array(tg_vec[:jdx + 1]).tolist()}')
#         print(f'\nlog of gate error (truc1={len(hspace_charge)}) = ')
#         for i in range(0, len(fidelity), 4):
#             print(', '.join(map(str, np.round(fidelity[i:i + 4], 8))), ',')

#         print(f'\ndrive_param (truc1={len(hspace_charge)}) = ')
#         for param in drive_param:
#             print(np.round(param, 6).tolist(), ',')

#         # Evaluate fidelity for optimal parameters in the full system
#         tg, drive_amp_A, drive_amp_B, detune_A, detune_B = drive_param[jdx]

#         n_cpu2 = 30
#         argz = [
#             H0_full, drive_full, w_trans_1, w_trans_2, hspace_full, n_cpu2,
#             tg, drive_amp_A, drive_amp_B, detune_A, detune_B
#         ]
#         fidelity_full.append(xgate_fidelity_log(argz))

#         print(f'\nlog of gate error (truc_full={truc_full}) = ')
#         for i in range(0, len(fidelity_full), 4):
#             print(', '.join(map(str, np.round(fidelity_full[i:i + 4], 8))), ',')

#         print("\n***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')

#     print('amp1_bounds =', amp1_bounds, ', amp2_bounds =', amp2_bounds)
#     print('detune1_bounds =', detune1_bounds, ', detune2_bounds =', detune2_bounds)