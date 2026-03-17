import os

import sys
sys.path.append('../')
import scqubits.settings as settings
import utils_2Q_gate_zp as ut
import numpy as np
import qutip as qt
from joblib import Parallel, delayed
settings.OVERLAP_THRESHOLD = 0.3


def get_config():
    """
    Centralized configuration dictionary
    """
    cfg = {}

    # ---------------- Gate type ----------------
    cfg['cz_run'] = False   # True -> CZ, False -> CNOT

    # ---------------- Truncation ----------------
    cfg['truc_one_qubit'] = 300
    cfg['truc_full'] = 1000
    cfg['charge_pick'] = True
    cfg['n_truc_list'] = [200, 500, 1000]

    # ---------------- Fidelity options ----------------
    cfg['calculate_ideal'] = True
    cfg['calculate_noise'] = False

    cfg['apply_decay'] = True
    cfg['apply_dephase'] = True
    cfg['decay_enlarge'] = 1.0
    cfg['filter_ratio'] = 0.3

    # ---------------- Noise ----------------
    cfg['t1_tphi_other'] = 170  # us

    # ---------------- Parallel ----------------
    cfg['num_cpus'] = 16
    cfg['max_step_ideal'] = 1e-3
    cfg['max_step_noisy'] = 1e-3

    # ---------------- Pulse params ----------------
    cfg['params'] = np.array([
        [50.008813, 0.090273, 0.046546, -0.027498, -0.025767],
        [100.006167, 0.053076, 0.029316, -0.01188, -0.01157],
        [149.996866, 0.039078, 0.024359, -0.009107, -0.008681],
    ])

    cfg['n_job'] = len(cfg['params'])

    return cfg


def load_qubit_data(cfg):
    return ut.load_qubit_data_2q(
        cfg['truc_one_qubit'],
        cfg['truc_full'],
        cfg['charge_pick']
    )


def get_gate_info(cfg, eval_tot, hspace_full, n_theta0_dress, n_theta1_dress):
    """
    Return drive_term and gate-specific frequencies
    """
    if cfg['cz_run']:
        drive_term = n_theta1_dress
        W_20_50 = (
            eval_tot[hspace_full.index('5-0')]
            - eval_tot[hspace_full.index('2-0')]
        )
        gate_freq = W_20_50
        mid_state = None
    else:
        drive_term = n_theta0_dress
        mid_state = '8-2'
        idx_0 = hspace_full.index('0-2')
        idx_1 = hspace_full.index('2-2')
        idx_2 = hspace_full.index(mid_state)

        W_0_2 = eval_tot[idx_2] - eval_tot[idx_0]
        W_1_2 = eval_tot[idx_2] - eval_tot[idx_1]
        gate_freq = (W_0_2, W_1_2, mid_state)

    return drive_term, gate_freq


def build_truncated_hamiltonian(
    cfg,
    hspace_full,
    eval_tot,
    eket_tot,
    drive_term,
    n_truc,
):
    hspace_select = hspace_full[:n_truc]
    index_select = [hspace_full.index(i) for i in hspace_select]

    H_drive, eket_truc = ut.build_hamiltonian_2q(
        cfg['cz_run'],
        index_select,
        eval_tot,
        eket_tot,
        drive_term,
    )

    return H_drive, eket_truc, hspace_select


def build_collapse_ops(cfg, hspace_0, hspace_1, eket_truc):
    if not cfg['calculate_noise']:
        return []

    (
        n_theta0,
        n_theta1,
        gamma_dephase_02_q0,
        gamma_dephase_02_q1
    ) = ut.load_noise_data_2q(cfg['t1_tphi_other'])

    Gamma = 1 / 1e3 / cfg['t1_tphi_other']
    Gamma_decay_q0 = Gamma / (n_theta0[4, 8] ** 2)
    Gamma_decay_q1 = Gamma / (n_theta1[4, 8] ** 2)

    transition_a, n_theta0_trunc = ut.get_transitions_for_collapse(
        hspace_0, n_theta0, filter_ratio=cfg['filter_ratio']
    )
    transition_b, n_theta1_trunc = ut.get_transitions_for_collapse(
        hspace_1, n_theta1, filter_ratio=cfg['filter_ratio']
    )

    return ut.construct_c_ops_2q(
        len(hspace_0),
        len(hspace_1),
        n_theta0_trunc,
        n_theta1_trunc,
        gamma_dephase_02_q0,
        gamma_dephase_02_q1,
        eket_truc,
        Gamma_decay_q0,
        Gamma_decay_q1,
        transition_a,
        transition_b,
        cfg['apply_decay'],
        cfg['apply_dephase'],
        cfg['decay_enlarge'],
    )

def run_fidelity(cfg, params, H_drive, gate_freq, logi_idx, c_ops, options):
    if cfg['cz_run']:
        return Parallel(n_jobs=cfg['n_job'])(
            delayed(ut.cz_fidelity_log_noise)(
                p, H_drive, gate_freq,
                cfg['num_cpus'], c_ops,
                logi_idx, *options
            ) for p in params
        )
    else:
        W_0_2, W_1_2, mid_state = gate_freq
        return Parallel(n_jobs=cfg['n_job'])(
            delayed(ut.cnot_fidelity_log_noise)(
                p, H_drive, W_0_2, W_1_2,
                cfg['num_cpus'], c_ops,
                logi_idx, mid_state, *options
            ) for p in params
        )

def main():
    cfg = get_config()

    ut.print_time()

    (
        hspace_full, eket_tot, eval_tot,
        n_theta0_dress, n_theta1_dress,
        hspace_0, hspace_1, logi_state
    ) = load_qubit_data(cfg)

    drive_term, gate_freq = get_gate_info(
        cfg, eval_tot, hspace_full, n_theta0_dress, n_theta1_dress
    )

    option_ideal, option_noisy = ut.get_qutip_options(
        cfg['max_step_ideal'],
        cfg['max_step_noisy']
    )

    options = (option_ideal, option_noisy)

    f_list = []

    for n_truc in cfg['n_truc_list']:
        H_drive, eket_truc, hspace_sel = build_truncated_hamiltonian(
            cfg, hspace_full, eval_tot, eket_tot, drive_term, n_truc
        )

        logi_idx = [hspace_sel.index(i) for i in logi_state]
        c_ops = build_collapse_ops(cfg, hspace_0, hspace_1, eket_truc)

        f = run_fidelity(
            cfg,
            cfg['params'],
            H_drive,
            gate_freq,
            logi_idx,
            c_ops,
            options
        )

        ut.print_fidelity(f'fidelity_{n_truc}', f)
        f_list.append(f)

    ut.print_fidelity('fidelity_list', f_list, num_each_row=1, num_digits=8)

if __name__ == '__main__':
    main()
