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


def import_para_noise():
    drive_phi, drive_theta, truc = False, True, 30
    # drive_phi, drive_theta, truc = True, False, 30
    fast, parallel = True, True
    folder = 'data/data_xgate_theta.txt' if drive_theta else 'data/data_xgate_phi.txt'
    f_xgate = pd.read_csv(folder)
    params = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()#[-5:]
    n_cpu, n_job = 4, 4*len(params)
    logi_state = [0, 2]
    # [H0, drive_term, w_trans_1, w_trans_2, _] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc)
    folder = f'data/data_one_zeropi_truncation=1000/'
    evals = 2*np.pi* pd.read_csv(folder+ 'evals.txt').to_numpy().flatten()
    n_theta = 2*np.pi* pd.read_csv(folder+ 'n_theta.txt').to_numpy()
    n_phi = 2*np.pi* pd.read_csv(folder+ 'n_phi.txt').map(complex).to_numpy()

    gamma2 =  1 / 1600e3
    gammas =  1 / 2e3
    gamma2_p = 0 / 100e3
    # ratio = 10
    # gammas =  gamma_2 * ratio
    gammas_p = 1 / 9e3

    jump_t1   = []
    jump_tphi = []
    gamma_t1   = [0, gammas,  gamma2]  + [gammas]  * (truc-3)
    gamma_tphi = [0, gammas_p, gamma2_p] + [gammas_p] * (truc-3)
    for i in range(1,truc):
        jump_t1.append( np.sqrt(gamma_t1[i]) * qt.basis(truc,0) * qt.basis(truc,i).dag() )
        jump_tphi.append( np.sqrt(2*gamma_tphi[i]) * qt.basis(truc,i).proj() )

    H0 = qt.Qobj(np.diag(evals))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta
    hilbert_space = np.arange(truc).tolist()
    H0_truc = ut.truncate_2(H0, hilbert_space)
    drive_truc = ut.truncate_2(drive_term, hilbert_space)
    H_qbt_drive = [H0_truc, [drive_truc, ut.drag_A],
                            [drive_truc, ut.drag_B],]
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta, '; truc = ', truc)
    print('fast=', fast, '; parallel = ', parallel)
    print('params =')
    for para in params:
        print(para.tolist(), ',')
    print("n_cpu = ", n_cpu, ";   n_job = ", n_job)
    print("gamma_2 = ", gamma2, "gamma_p2 = ", gamma2_p)
    print(f"T1_2 = {1/gamma2} ns") if gamma2 != 0 else None
    print(f"T1_other = {1/gammas} ns") if gammas != 0 else None
    print(f"Tphi_2 = {1/gamma2_p} ns") if gamma2_p != 0 else None
    print(f"Tphi_other = {1/gammas_p} ns") if gammas_p != 0 else None

    ############################################################
    if fast:
        c_op_list = []
        args = [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state, parallel]
        f_ideal = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_noise_fast)(args_indep, *args)
                                                    for args_indep in params)
        print('\nf_ideal = [')
        for i in range(0, len(f_ideal), 4):
            print(', '.join(map(str, f_ideal[i:i+4])), ',')
        print(']')
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
        ############################################################
        c_op_list = jump_t1 + jump_tphi
        args = [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state, parallel]
        f_noise = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_noise_fast)(args_indep, *args)
                                                    for args_indep in params)
        print('\nf_noise = [')
        for i in range(0, len(f_noise), 4):
            print(', '.join(map(str, f_noise[i:i+4])), ',')
        print(']')
    else:
        c_op_list = []
        args = [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state]
        f_ideal = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_noise)(args_indep, *args)
                                                    for args_indep in params)
        print('\nf_ideal = [')
        for i in range(0, len(f_ideal), 4):
            print(', '.join(map(str, f_ideal[i:i+4])), ',')
        print(']')
        print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
        ############################################################
        c_op_list = jump_t1 + jump_tphi
        args = [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state]
        f_noise = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_noise)(args_indep, *args)
                                                    for args_indep in params)
        print('\nf_noise = [')
        for i in range(0, len(f_noise), 4):
            print(', '.join(map(str, f_noise[i:i+4])), ',')
        print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

def check_eval():
    truncation=10
    EL        = 0.377 # GHz
    EJ        = 6.013 # Soft Zero Pi (Gyenis)
    EC_phi    = 1.142
    EC_theta  = 0.092
    E_CJ = 2 * EC_phi
    E_C = 2./(1./EC_theta -1./EC_phi)
    phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, 100)
    zero_pi = scq.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=E_CJ, EC = E_C, dEJ=0.,
                            ng=0., flux=0., ncut=30, truncated_dim=truncation)
    def get_eval(args, index=10):
        zero_pi.ncut, zero_pi.grid.pt_count = args
        evals = zero_pi.eigenvals(evals_count=index+1)
        evals = evals - evals[0]
        return evals[index]
    ncut_vec = np.arange(10, 100, 5).tolist()
    phi_grid_vec = np.arange(20, 300, 5).tolist()

    level_index = 300
    print('level_index=', level_index)
    print('ncut_vec=', ncut_vec)
    print('phi_grid_vec=', phi_grid_vec)
    print('size=', len(ncut_vec)*len(phi_grid_vec))
    grid1, grid2 = np.meshgrid(ncut_vec, phi_grid_vec)
    grid = np.stack([grid1.ravel(), grid2.ravel()], axis=-1)
    eval_grid = Parallel(n_jobs=50, verbose=0)(delayed(get_eval)(args, level_index)
                                                for args in grid)
    eval_grid = np.reshape(eval_grid, (len(ncut_vec), len(phi_grid_vec)))
    eval_error = eval_grid - eval_grid[-1, -1]

    idx_phase = phi_grid_vec.index(110)
    idx_charge = ncut_vec.index(30)
    folder = f'data/eval/'
    pd.DataFrame(eval_error).to_csv(folder+ 'eval_error.txt', sep=',', index=False, header=True)
    print(f'Eigenvalue error: Level={level_index} (error={eval_error[idx_charge, idx_phase]})')

def import_para():
    ############################################################
    f_xgate = pd.read_csv('data/data_xgate_theta.txt')
    # f_xgate = pd.read_csv('data/data_xgate_phi.txt')
    para_tot = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2',
                        'detune_1', 'detune_2']].to_numpy()
    n_cpu, n_job = 50, 30*len(para_tot)
    drive_phi, drive_theta, drag = False, True, 0
    # drive_phi, drive_theta, drag =  True, False, 0

    print('n_job=', n_job, ', n_cpu=', n_cpu)
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta, '; DRAG =',drag)
    print('para_tot =')
    for para in para_tot:
        print(para.tolist(), ',')

    ############################################################
    truc1 = 300
    ratio = 1.8
    ncut, phi_cut = int(ratio*30), int(ratio*100)
    print('ncut=', ncut, ', phi_cut=', phi_cut)
    [H0, drive_term, w_trans_1, w_trans_2, hspace] = ut.zero_pi_initialize(
        drive_phi, drive_theta, truncation=truc1, ncut=ncut, phi_cut=phi_cut)
    hspace = np.arange(truc1).tolist()
    print('\ntruc1_a=', truc1, '; truc1_b=', len(hspace))
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace, n_cpu, drag]
    res = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_parallel)(arg, *args)
                                                for arg in para_tot)
    # res = np.reshape(res, (len(para_tot), 2))
    # f_theta = res[:,0]
    # error_leakage = res[:,1]

    f_theta = res
    print(f'log of gate error (truc={len(hspace)}) = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')
    # print('leakage_error=')
    # for i in range(0, len(error_leakage), 4):
    #     print(', '.join(map(str, np.round(error_leakage[i:i+4], 8))), ',')

    # print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    ############################################################
    # truc1 = 2000
    # [H0, drive_term, w_trans_1, w_trans_2, hspace] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)
    # hspace = np.arange(truc1).tolist()
    # print('\ntruc1_a=', truc1, '; truc1_b=', len(hspace))
    # args = [H0, drive_term, w_trans_1, w_trans_2, hspace, n_cpu, drag]
    # f_theta = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_parallel)(arg, *args)
    #                                             for arg in para_tot)
    # print(f'log of gate error (truc={truc1}) = ')
    # for i in range(0, len(f_theta), 4):
    #     print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')
    # print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_para()
    import_para_noise()
    # check_eval()
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




