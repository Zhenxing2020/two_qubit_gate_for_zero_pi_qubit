import argparse
import sys
import os
from pathlib import Path

XGATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = XGATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(XGATE_DIR)

import numpy as np
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import utils_2Q_gate_zp as ut
import qutip as qt
import scqubits as scq
from datetime import datetime
import pytz
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd


def import_para_noise(
    drive_phi=False,
    drive_theta=True,
    truc=30,
    fast=True,
    parallel=True,
    num_cpu=4,
    qubit_0=True,
    num_rows=1,
):
    folder = "data/data_xgate_theta.txt" if drive_theta else "data/data_xgate_phi.txt"
    f_xgate = pd.read_csv(folder)
    params = f_xgate[["tg", "drive_amp_1", "drive_amp_2", "detune_1", "detune_2"]].to_numpy()[:num_rows]
    n_cpu, n_job = num_cpu, 4 * len(params)
    logi_state = [0, 2]
    # [H0, drive_term, w_trans_1, w_trans_2, _] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc)
    folder = "data/data_one_zeropi_truncation=1000/"
    if qubit_0:
        evals = 2 * np.pi * pd.read_csv(folder + "evals.txt").to_numpy().flatten()
        n_theta = 2 * np.pi * pd.read_csv(folder + "n_theta.txt").to_numpy()
        n_phi = 2 * np.pi * pd.read_csv(folder + "n_phi.txt").map(complex).to_numpy()
    else:
        evals, n_theta, n_phi, _ = ut.load_qubit_data_xgate(qubit_0=qubit_0)

    gamma_decay_logi =  1 / 1600e3
    gamma_dephase_logi = 0
    gamma_decay_other =  1 / 10e3
    gamma_dephase_other = 1 / 10e3

    jump_t1   = []
    jump_tphi = []
    gamma_t1   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (truc-3)
    gamma_tphi = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (truc-3)
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
    print("gamma_2 = ", gamma_decay_logi, "gamma_p2 = ", gamma_dephase_logi)
    print(f"T1_2 = {1/gamma_decay_logi} ns") if gamma_decay_logi != 0 else None
    print(f"T1_other = {1/gamma_decay_other} ns") if gamma_decay_other != 0 else None
    print(f"Tphi_2 = {1/gamma_dephase_logi} ns") if gamma_dephase_logi != 0 else None
    print(f"Tphi_other = {1/gamma_dephase_other} ns") if gamma_dephase_other != 0 else None

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

def check_eval(truncation=10, level_index=300, n_jobs=50):
    EL = 0.377 # GHz
    EJ = 6.013 # Soft Zero Pi (Gyenis)
    EC_phi = 1.142
    EC_theta = 0.092
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

    print('level_index=', level_index)
    print('ncut_vec=', ncut_vec)
    print('phi_grid_vec=', phi_grid_vec)
    print('size=', len(ncut_vec)*len(phi_grid_vec))
    grid1, grid2 = np.meshgrid(ncut_vec, phi_grid_vec)
    grid = np.stack([grid1.ravel(), grid2.ravel()], axis=-1)
    eval_grid = Parallel(n_jobs=n_jobs, verbose=0)(delayed(get_eval)(args, level_index)
                                                for args in grid)
    eval_grid = np.reshape(eval_grid, (len(ncut_vec), len(phi_grid_vec)))
    eval_error = eval_grid - eval_grid[-1, -1]

    idx_phase = phi_grid_vec.index(110)
    idx_charge = ncut_vec.index(30)
    folder = f'data/eval/'
    pd.DataFrame(eval_error).to_csv(folder+ 'eval_error.txt', sep=',', index=False, header=True)
    print(f'Eigenvalue error: Level={level_index} (error={eval_error[idx_charge, idx_phase]})')



def import_para(drive_phi=False, drive_theta=True, n_cpu=50):
    ############################################################
    f_xgate = pd.read_csv('data/data_xgate_theta.txt')
    # f_xgate = pd.read_csv('data/data_xgate_phi.txt')
    para_tot = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2',
                        'detune_1', 'detune_2']].to_numpy()
    n_job = 30 * len(para_tot)
    drive_phi, drive_theta, drag = False, True, 0
    # drive_phi, drive_theta, drag =  True, False, 0

    print('n_job=', n_job, ', n_cpu=', n_cpu)
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta, '; DRAG =',drag)
    print('para_tot =')
    for para in para_tot:
        print(para.tolist(), ',')

    ############################################################
    # ratio = 1.8
    # ncut, phi_cut = int(ratio*30), int(ratio*100)
    # print('ncut=', ncut, ', phi_cut=', phi_cut)
    # [H0, drive_term, w_trans_1, w_trans_2, hspace] = ut.zero_pi_initialize(
    #     drive_phi, drive_theta, truncation=truc1, ncut=ncut, phi_cut=phi_cut)
    # hspace = np.arange(truc1).tolist()

#####################################################################
    truncation, truc1 = 1000, 500
    folder = 'data/'
    new_specdata = scq.read(folder + f'zeropi_specdata_truc={truncation}_3ncut.h5')
    n_theta = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
    n_phi = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
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

    hilbert_space = hspace
    print('\ntruc1_a=', truc1, '; truc1_b=', len(hspace_charge))
    args = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, drag]
    f_theta = Parallel(n_jobs=n_job, verbose=0)(delayed(ut.xgate_fidelity_parallel)(arg, *args)
                                                for arg in para_tot)
    print(f'log of gate error (truc={len(hilbert_space)}) = ')
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

def generate_data(truncation=1000, ncut=90, phi_cut=300):
    print('truncation=', truncation)
    print('ncut, phi_cut =', ncut, phi_cut)
    folder = '../../data/data_3ncut_one_zeropi/'
    EL        = 0.377 # GHz
    EJ        = 5.41 # Soft Zero Pi (Gyenis)
    EC_phi    = 1.142
    EC_theta  = 0.092
    E_CJ = 2 * EC_phi
    E_C = 2./(1./EC_theta -1./EC_phi)
    phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, phi_cut)
    zero_pi = scq.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=E_CJ, EC = E_C, dEJ=0.,
                            ng=0., flux=0., ncut=ncut, truncated_dim=truncation)
    zero_pi.eigensys(evals_count=truncation, return_spectrumdata=True,
                    filename=folder + f'zeropi2_specdata_truc={truncation}.h5')
    zero_pi.matrixelement_table(operator='n_theta_operator', evals_count=truncation, return_datastore=True,
                                          filename=folder + f'zeropi2_n_theta_truc={truncation}.h5')
    zero_pi.matrixelement_table(operator='i_d_dphi_operator', evals_count=truncation, return_datastore=True,
                                        filename=folder + f'zeropi2_n_phi_truc={truncation}.h5')

    eval0 = 2*np.pi* scq.read(folder + f'zeropi_0_specdata_truc={truncation}_3ncut.h5').energy_table
    n_theta0 = 2*np.pi* scq.read(folder + f'zeropi_0_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
    n_phi0 = 2*np.pi* scq.read(folder + f'zeropi_0_n_phi_truc={truncation}_3ncut.h5').matrixelem_table
    eval0 = eval0 - eval0[0]
    print(eval0)
    print(n_theta0)
    print(n_phi0)

def build_parser():
    parser = argparse.ArgumentParser(description="Prepare data/parameters for X-gate workflows.")
    parser.add_argument("--task", choices=["generate-data", "import-para", "import-para-noise", "check-eval"], default="generate-data")
    parser.add_argument("--drive-phi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drive-theta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truc", type=int, default=30)
    parser.add_argument("--qubit-0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-cpu", type=int, default=4)
    parser.add_argument("--num-rows", type=int, default=1)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncation", type=int, default=1000)
    parser.add_argument("--ncut", type=int, default=90)
    parser.add_argument("--phi-cut", type=int, default=300)
    parser.add_argument("--level-index", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=50)
    return parser


def main():
    args = build_parser().parse_args()
    print(os.path.basename(__file__))
    print("Current Mountain Time:", datetime.now(pytz.timezone("America/Denver")))
    if args.task == "generate-data":
        generate_data(truncation=args.truncation, ncut=args.ncut, phi_cut=args.phi_cut)
    elif args.task == "import-para":
        import_para(drive_phi=args.drive_phi, drive_theta=args.drive_theta, n_cpu=args.num_cpu)
    elif args.task == "import-para-noise":
        import_para_noise(
            drive_phi=args.drive_phi,
            drive_theta=args.drive_theta,
            truc=args.truc,
            fast=args.fast,
            parallel=args.parallel,
            num_cpu=args.num_cpu,
            qubit_0=args.qubit_0,
            num_rows=args.num_rows,
        )
    else:
        check_eval(truncation=args.truncation, level_index=args.level_index, n_jobs=args.n_jobs)
    print("Current Mountain Time:", datetime.now(pytz.timezone("America/Denver")))


if __name__ == "__main__":
    main()




