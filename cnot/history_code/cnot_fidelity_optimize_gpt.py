#!/usr/bin/env python3
# pyright: reportMissingImports=false

"""
CNOT Gate Fidelity Optimization for Two-Qubit Zero-Pi Systems
Rewritten in modular (non-class) form,
fully aligned with the CZ-gate optimization framework.

Author: ChatGPT (based on your CZ code and old CNOT code)
Date: 2025
"""

import os, sys
from datetime import datetime
import pytz
import numpy as np
import scipy as sp
from tqdm import tqdm
import qutip as qt
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3

sys.path.append('../')
import utils_2Q_gate_zp as ut
import ham_data as hd


def load_system_data_cnot(config):
    """
    Load data for CNOT optimization. Similar to CZ but includes:
    - two drive terms (n_theta0, n_theta1)
    - energy differences W_0_2, W_1_2
    - mid_state logic (8-2, 4-5, etc.)
    """

    print("Loading CNOT system data...")

    # Load eigenvalues, eigenstates, logical states
    (hspace_full, eket_tot, eval_tot, _, 
     n_theta1_dress, n_theta0_dress, 
     _, logi_state) = hd.load_two_qubit_data(
         config['folder_load'], return_full=False
    )

    # mid-state definition (8–2, 4–5, 1–4, …)
    mid_state = config['mid_state']
    idx_mid = hspace_full.index(mid_state)

    # logical state indices
    idx_0 = hspace_full.index(config['logical_state_A'])
    idx_1 = hspace_full.index(config['logical_state_B'])

    # transition frequencies
    W_0_2 = eval_tot[idx_mid] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_mid] - eval_tot[idx_1]

    # truncation selection
    if config['use_truc_model']:
        hspace_select = ut.truc_model[config['truc_model_name']][:config['truc_optimize']]
    else:
        hspace_select = hspace_full[:config['truc_optimize']]

    option_ideal, option_noisy = ut.get_qutip_options(
        config['max_step_ideal'], config['max_step_noisy']
    )

    # load initial pulse parameters
    pulse_param = ut.load_drive_params_2q(
        config['cnot_run'], folder=config['folder_pulse']
    )[config['gate_time_indices'], :]

    print(f"Loaded CNOT system with {len(hspace_full)} states.")
    print(f"Optimization truncation = {len(hspace_select)}")
    print(f"W_0_2, W_1_2 = {W_0_2:.3f}, {W_1_2:.3f}")

    return dict(
        hspace_full=hspace_full,
        eket_tot=eket_tot,
        eval_tot=eval_tot,
        n_theta0_dress=n_theta0_dress,
        n_theta1_dress=n_theta1_dress,
        hspace_select=hspace_select,
        logi_state=config['logical_states'],
        mid_state=mid_state,
        idx_mid=idx_mid,
        idx_0=idx_0,
        idx_1=idx_1,
        W_0_2=W_0_2,
        W_1_2=W_1_2,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        pulse_param=pulse_param
    )




def build_hamiltonians_cnot(sys, config):
    """
    Build truncated-space and large-space Hamiltonians for CNOT.

    Drive Hamiltonian:
        H = H0 + A1(t) * n_theta0 + A2(t) * n_theta1
    """

    print("Building CNOT Hamiltonians...")

    # indices for truncation
    index_select = [sys['hspace_full'].index(i) for i in sys['hspace_select']]

    H0 = np.diag(sys['eval_tot'])
    H0_qobj = qt.Qobj(H0)

    # truncate for optimization space
    H0_part = ut.truncate_2(H0_qobj, index_select)
    n0_part = ut.truncate_2(sys['n_theta0_dress'], index_select)
    n1_part = ut.truncate_2(sys['n_theta1_dress'], index_select)

    logi_idx_select = [sys['hspace_select'].index(i) for i in sys['logi_state']]

    # large Hilbert space for verification
    hspace_large = sys['hspace_full'][:config['truc_large']]
    idx_large = list(range(config['truc_large']))

    H0_large = qt.Qobj(np.diag(sys['eval_tot'][:config['truc_large']]))
    n0_large = ut.truncate_2(sys['n_theta0_dress'], idx_large)
    n1_large = ut.truncate_2(sys['n_theta1_dress'], idx_large)
    logi_idx_large = [hspace_large.index(i) for i in sys['logi_state']]

    # drive definition
    H_drive_select = [H0_part, [n0_part, ut.drive_gauss_A], [n1_part, ut.drive_gauss_B]]
    H_drive_large  = [H0_large, [n0_large, ut.drive_gauss_A], [n1_large, ut.drive_gauss_B]]

    return dict(
        H_drive_select=H_drive_select,
        H_drive_large=H_drive_large,
        logi_idx_select=logi_idx_select,
        logi_idx_large=logi_idx_large
    )


def optimize_single_gate_time_cnot(
    gate_time_idx, system_data, hamiltonians, config, drive_param_list
):
    """
    Same structure as CZ version, but using:
        ut.cnot_fidelity_log_noise
    """

    tg_initial = system_data['pulse_param'][gate_time_idx, 0]

    bounds = (
        (tg_initial + config['tg_bound'][0], tg_initial + config['tg_bound'][1]),
        config['A1_bound'],
        config['A2_bound'],
        config['detune1_bound'],
        config['detune2_bound']
    )

    args_truc = [
        hamiltonians['H_drive_select'],
        system_data['W_0_2'],
        system_data['W_1_2'],
        config['workers_eval'],
        [],  # collapse operators
        hamiltonians['logi_idx_select'],
        system_data['mid_state'],
        system_data['option_ideal'],
        system_data['option_noisy']
    ]

    print(f"\n--- Optimizing CNOT gate index {gate_time_idx}, tg={tg_initial:.6f} ---")

    params = dict(
        func=ut.cnot_fidelity_log_noise,
        bounds=bounds,
        args=args_truc,
        disp=True,
        callback=ut.print_soln,
        init="sobol",
        workers=config['workers'],
        popsize=config['popsize'],
        mutation=config['mutation'],
        recombination=config['recombination'],
        tol=config['tol'],
        polish=False,
    )

    # x0 逻辑复用你 CZ 代码
    if config['use_x0'] == 'from_input':
        params['x0'] = system_data['pulse_param'][gate_time_idx, :5]

    result = sp.optimize.differential_evolution(**params)

    print(f"Optimized fidelity = {result.fun:.8f}")
    print(f"x = {result.x}")

    return result.fun, result.x.tolist()


def check_pulse_in_large_cnot(drive_params, system_data, hamiltonians, config):
    tg, A1, A2, d1, d2 = drive_params
    args_all = [
        tg, A1, A2, d1, d2,
        hamiltonians['H_drive_large'],
        system_data['W_0_2'],
        system_data['W_1_2'],
        config['workers_eval'],
        [],
        hamiltonians['logi_idx_large'],
        system_data['mid_state'],
        system_data['option_ideal'],
        system_data['option_noisy']
    ]
    return ut.cnot_fidelity_log(args_all)


def run_fidelity_sweep_cnot(system_data, hamiltonians, config):
    fidelity_list, drive_param_list = [], []
    fidelity_large_list = []

    for idx in tqdm(range(len(system_data['pulse_param'])), desc="CNOT optimize"):
        f, param = optimize_single_gate_time_cnot(
            idx, system_data, hamiltonians, config, drive_param_list
        )
        fidelity_list.append(f)
        drive_param_list.append(param)

        ut.print_pulse_params("CNOT param_opt", drive_param_list)
        ut.print_fidelity("f_truc", fidelity_list)

        f_large = check_pulse_in_large_cnot(
            param, system_data, hamiltonians, config
        )
        fidelity_large_list.append(f_large)
        ut.print_fidelity("f_large", fidelity_large_list)

    print("\n=== CNOT fidelity sweep finished ===")


def get_cnot_config(custom=None):
    config = dict(
        truc_large=1000,
        truc_optimize=300,
        use_truc_model=False,
        truc_model_name='cnot_short_1000',
        cnot_run=True,
        folder_load='../../data/_truc_3000',
        folder_pulse='data/npz/cnot_pulse.txt',

        gate_time_indices=list(range(0, 20)),
        max_step_ideal=1e-3,
        max_step_noisy=1e-3,

        workers=80,
        workers_eval=1,
        popsize=12,
        recombination=0.7,
        tol=0.01,
        mutation=(0.5, 1),

        # CNOT bounds
        tg_bound=(-0.01, 0.01),
        A1_bound=(0.05, 0.5),
        A2_bound=(0.01, 0.5),
        detune1_bound=(-0.5, 0.1),
        detune2_bound=(-0.5, 0.1),

        # mid-state
        mid_state='8-2',
        logical_state_A='0-2',
        logical_state_B='2-2',
        logical_states=['0-0','0-2','2-0','2-2'],

        use_x0='from_input'
    )
    if custom:
        config.update(custom)
    return config


def main():
    print("Running CNOT gate optimizer...")
    ut.print_time()

    config = get_cnot_config()

    system_data = load_system_data_cnot(config)
    hamiltonians = build_hamiltonians_cnot(system_data, config)

    ut.print_pulse_params("Init param", system_data['pulse_param'])

    run_fidelity_sweep_cnot(system_data, hamiltonians, config)

    print("\n=== CNOT OPTIMIZATION COMPLETED ===")


if __name__ == "__main__":
    main()








