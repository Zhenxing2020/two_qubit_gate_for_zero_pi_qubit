#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Unified 2Q Gate Optimizer for Zero-Pi System
Supports:
    - CZ gate optimization
    - CNOT gate optimization

Features:
    - Modular, function-based structure
    - Differential evolution optimizer
    - Automatic saving (npz + csv)
    - Resume from previous runs
    - Simple plotting (fidelity & parameters)

Author: ChatGPT + Zhenxing's original scripts
Date: 2025
"""

# ===== Standard Library Imports =====
import os
import sys
from datetime import datetime
import pytz
import json

# ===== Third-Party Imports =====
import numpy as np
import scipy as sp
from tqdm import tqdm
import qutip as qt
import pandas as pd
from matplotlib import pyplot as plt
import scqubits.settings as settings

settings.OVERLAP_THRESHOLD = 0.3

# ===== Local Imports =====
from pathlib import Path

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)
import utils_2Q_gate_zp as ut
import ham_data as hd
from functools import partial

# === global variables for multiprocessing ===
GLOBAL_system_data = None
GLOBAL_hamiltonians = None
GLOBAL_config = None

# ==============================================================
# SYSTEM LOADING
# ==============================================================

def load_system_data(config):
    """Dispatch to CZ / CNOT system loading routines based on gate_type."""
    gate_type = config["gate_type"]
    if gate_type == "CZ":
        return load_system_data_cz(config)
    elif gate_type == "CNOT":
        return load_system_data_cnot(config)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


def load_system_data_cz(config):
    """System data loader for CZ gate, adapted from the main CZ script."""

    print("Loading system data for CZ...")

    (hspace_full, eket_tot, eval_tot, _, _, _, n_theta1_dress, _, _,
     _, _, logi_state) = hd.load_two_qubit_data(config['folder_load'], return_full=False)

    drive_term = n_theta1_dress

    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    if config['use_truc_model']:
        hspace_select = ut.truc_model[config['truc_model_name']][:config['truc_optimize']]
    else:
        hspace_select = hspace_full[:config['truc_optimize']]

    option_ideal, option_noisy = ut.get_qutip_options(
        config['max_step_ideal'], config['max_step_noisy']
    )

    pulse_param = ut.load_drive_params_2q(
        config['gate_type'] == 'CZ', folder=config.get('folder_pulse', None)
    )[config.get('gate_time_indices', slice(None)), :]

    print(f"Loaded CZ data: total states = {len(hspace_full)}")
    print(f"Optimization truncation = {len(hspace_select)}")
    print(f"W_20_50 = {np.round(W_20_50, 3)}")

    return dict(
        gate_type='CZ',
        hspace_full=hspace_full,
        eket_tot=eket_tot,
        eval_tot=eval_tot,
        drive_term=drive_term,
        logi_state=logi_state,
        hspace_select=hspace_select,
        W_20_50=W_20_50,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        pulse_param=pulse_param,
    )


def load_system_data_cnot(config):
    """System data loader for CNOT using the current two-qubit data layout."""

    print("Loading system data for CNOT...")

    (hspace_full, eket_tot, eval_tot, _, _, n_theta0_dress,
     n_theta1_dress, hspace_0, hspace_1, hspace_n_theta1,
     hspace_n_theta2, logi_state) = hd.load_two_qubit_data(
        config['folder_load'], return_full=False
    )

    if config['mid_state'] in ['8-2', '4-5']:
        idx_0 = hspace_full.index('0-2')
        idx_1 = hspace_full.index('2-2')
    elif config['mid_state'] in ['1-4', '8-0', '4-1']:
        idx_0 = hspace_full.index('0-0')
        idx_1 = hspace_full.index('2-0')
    else:
        raise ValueError(f"Unsupported mid_state for CNOT: {config['mid_state']}")

    if config['mid_state'] in ['8-2', '4-5', '1-4', '8-0', '4-1']:
        drive_term = n_theta0_dress
    else:
        drive_term = n_theta1_dress

    idx_mid = hspace_full.index(config['mid_state'])
    W_0_2 = eval_tot[idx_mid] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_mid] - eval_tot[idx_1]

    if config["reduced_model"] == 'lowest_state':
        index_select = list(range(config['truc_optimize']))
        hspace_select = np.array(hspace_full)[:config['truc_optimize']]
        
    elif config["reduced_model"] == 'charge_pick':
        index_select = hspace_n_theta1[:config['truc_optimize']]
        hspace_select = [np.array(hspace_full)[i] for i in index_select]
        
    elif config["reduced_model"] == 'graph_pick':
        index_select = ut.truc_model[config["graph_model_name"]][:config['truc_optimize']]
        hspace_select = [np.array(hspace_full)[i] for i in index_select]        
        
    else:
        raise ValueError("Unknown reduced_model type. Please choose from 'graph_pick', 'lowest_state', 'charge_pick'.")
    
    # if config['use_truc_model']:
    #     hspace_select = np.array(hspace_full)[ hspace_n_theta1[:config['truc_optimize']]].tolist()
        # hspace_select = np.array(hspace_full)[ ut.truc_model[config['truc_model_name']][:config['truc_optimize']]].tolist()
        # hspace_select = ut.truc_model['cnot_' + config['mid_state'][0] + config['mid_state'][2]][:config['truc_optimize']]
    # else:
    #     hspace_select = hspace_full[:config['truc_optimize']]
    print(f'hspace_full={hspace_full[:30]}')
    print(f"hspace_select_idx={index_select}")
    ut.print_fidelity(f'hspace_select_idx', index_select, num_each_row=20, n_make_blank_line=200)
    print(f'hspace_select={hspace_select[:30]}')
    
    option_ideal, option_noisy = ut.get_qutip_options(config['max_step_ideal'], config['max_step_noisy'])

    if config.get('folder_pulse', None) is not None:
        pulse_param = ut.load_drive_params_2q(
            config['gate_type']=='CZ', folder=config.get('folder_pulse', None)
        )[config.get('gate_time_indices', slice(None)), :]
    else:
        pulse_param = config['x0_array'][config.get('gate_time_indices', slice(None)), :]

    print(f"Loaded CNOT data: total states = {len(hspace_full)}")

    return dict(
        gate_type='CNOT',
        hspace_full=hspace_full,
        eket_tot=eket_tot,
        eval_tot=eval_tot,
        drive_term=drive_term,
        hspace_select=hspace_select,
        logi_state=logi_state,
        W_0_2=W_0_2,
        W_1_2=W_1_2,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        pulse_param=pulse_param,
    )


# ==============================================================
# HAMILTONIAN BUILDING
# ==============================================================

def build_hamiltonians(system_data, config):
    gate_type = config["gate_type"]
    if gate_type == "CZ":
        return build_hamiltonians_cz(system_data, config)
    elif gate_type == "CNOT":
        return build_hamiltonians_cnot(system_data, config)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


def build_hamiltonians_cz(sys, config):
    """Hamiltonian construction for CZ gate, similar to the main CZ script."""
    print("Building Hamiltonians for CZ...")

    logi_idx_select = [sys['hspace_select'].index(i) for i in sys['logi_state']]
    index_select = [sys['hspace_full'].index(i) for i in sys['hspace_select']]

    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', index_select,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    hspace_large = sys['hspace_full'][:config['truc_large']]
    logi_idx_large = [hspace_large.index(i) for i in sys['logi_state']]
    idx_large = np.arange(config['truc_large']).tolist()

    H_drive_large, _ = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', idx_large,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    return dict(
        H_drive_select=H_drive_select,
        H_drive_large=H_drive_large,
        logi_idx_select=logi_idx_select,
        logi_idx_large=logi_idx_large,
    )


def build_hamiltonians_cnot(sys, config):
    """Hamiltonian construction for CNOT optimization and large-space checks."""
    print("Building Hamiltonians for CNOT...")

    logi_idx_select = [sys['hspace_select'].index(i) for i in sys['logi_state']]
    # print(f"logi_idx_select = {logi_idx_select}\n")
    # print(f"sys['hspace_select'] = {sys['hspace_select'][:20]}\n")
    # print(f"sys['logi_state'] = {sys['logi_state']}\n")
    # print("sys['hspace_full'] =", sys['hspace_full'][:20])
    
    index_select = [sys['hspace_full'].index(i) for i in sys['hspace_select']]

    H_drive_select, _ = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', index_select,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    hspace_large = sys['hspace_full'][:config['truc_large']]
    logi_idx_large = [hspace_large.index(i) for i in sys['logi_state']]
    idx_large = np.arange(config['truc_large']).tolist()

    H_drive_large, _ = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', idx_large,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    return dict(
        H_drive_select=H_drive_select,
        H_drive_large=H_drive_large,
        logi_idx_select=logi_idx_select,
        logi_idx_large=logi_idx_large,
    )


# ==============================================================
# FIDELITY EVALUATION HELPERS
# ==============================================================

def evaluate_fidelity_truncated(x, system_data, hamiltonians, config):
    """Wrapper objective function for differential evolution."""
    gate_type = config['gate_type']

    if gate_type == 'CZ':
        # x = [tg, amp, detune]
        args = [
            hamiltonians['H_drive_select'],
            system_data['W_20_50'],
            1,
            [],
            hamiltonians['logi_idx_select'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cz_fidelity_log_noise(x, *args)

    elif gate_type == 'CNOT':
        # x = [tg, A1, A2, d1, d2]
        args = [
            hamiltonians['H_drive_select'],
            system_data['W_0_2'],
            system_data['W_1_2'],
            1,
            [],
            hamiltonians['logi_idx_select'],
            config['mid_state'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cnot_fidelity_log_noise(x, *args)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


def evaluate_fidelity_large(drive_params, system_data, hamiltonians, config):
    """Evaluate fidelity on a larger Hilbert space."""
    gate_type = config['gate_type']

    if gate_type == 'CZ':
        # Evaluate on the large Hilbert space with the SAME rigorous fidelity path
        # as the DE objective (cz_fidelity_log_noise -> cz_fidelity_log: super-
        # operator propagator, 3-pts/ns time grid), just on the large operators.
        # The legacy cz_fidelity_log_old used a coarser (1-pt/ns) grid and a
        # different unitary path, which put a ~1% systematic offset between the
        # optimized-large and objective/fixed-pulse fidelities. num_cpus follows
        # the config inner parallelism (this runs in the master, after the DE pool
        # closes, so a pool here does not nest).
        args = [
            hamiltonians['H_drive_large'],
            system_data['W_20_50'],
            config.get('inner_num_cpus', 1),
            [],
            hamiltonians['logi_idx_large'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cz_fidelity_log_noise(list(drive_params), *args)

    elif gate_type == 'CNOT':
        tg, A1, A2, d1, d2 = drive_params
        num_cpus = 1
        c_op_list = []
        arg_all = [
            tg, A1, A2, d1, d2,
            hamiltonians['H_drive_large'],
            system_data['W_0_2'],
            system_data['W_1_2'],
            num_cpus,
            c_op_list,
            hamiltonians['logi_idx_large'],
            config['mid_state'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cnot_fidelity_log(arg_all)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


# ==============================================================
# OPTIMIZATION CORE
# ==============================================================

def fid_func_global(x):
    """
    Global wrapper for truncated fidelity used by differential evolution.
    It reads complex objects from global variables to avoid pickling issues.
    """    
    # x ??differential_evolution ??????
    # ???????????????????pickle
    return evaluate_fidelity_truncated(
        x,
        system_data=GLOBAL_system_data,
        hamiltonians=GLOBAL_hamiltonians,
        config=GLOBAL_config,
    )

def vectorized_fid(X):
    """
    Example vectorized fidelity using a process pool.
    Not used by default differential evolution, but available if needed.
    """    
    return [fid_func_global(x) for x in X]


def optimize_single_gate_time(
    gate_time_idx, system_data, hamiltonians, config, drive_param_list
):
    """
    Optimize fidelity for a single gate time index using differential evolution.
    """    
    gate_type = config['gate_type']
    if gate_type == 'CZ':
        tg_initial = system_data['pulse_param'][gate_time_idx, 0]
        tg_bounds = (tg_initial + config['tg_bound'][0],
                     tg_initial + config['tg_bound'][1])
        bounds = (tg_bounds, config['amp_bound'], config['detune_bound'])
    else:
        tg_idx = config['tg_opt_vec'][gate_time_idx]
        tg_initial = tg_idx
        tg_bounds = (tg_initial + config['tg_bound'][0],
                     tg_initial + config['tg_bound'][1])
        bounds = (
            tg_bounds,
            config['A1_bound'],
            config['A2_bound'],
            config['detune1_bound'],
            config['detune2_bound'],
        )

    print(f"\n--- Optimizing {gate_type} at index {gate_time_idx}, tg_init={tg_initial:.6f} ---")
    ut.print_time()

    # differential evolution settings
    # ??partial ?????? pickle ?????    # fid_func = partial(
    #     evaluate_fidelity_truncated,
    #     system_data=system_data,
    #     hamiltonians=hamiltonians,
    #     config=config
    # )

    global GLOBAL_system_data, GLOBAL_hamiltonians, GLOBAL_config

    # === ????????????????? ===
    GLOBAL_system_data = system_data
    GLOBAL_hamiltonians = hamiltonians
    GLOBAL_config = config

    params = dict(
        func=fid_func_global,
        bounds=bounds,
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

    # x0 ???
    if config['use_x0'] == 'from_neighbor':
        if gate_time_idx == 0:
            if config['first_x0_from_input']:
                print("Using first x0 from input for initialization (first gate time)")
                if gate_type == 'CZ':
                    params['x0'] = system_data['pulse_param'][gate_time_idx, :3]
                else:
                    params['x0'] = config['x0_array'][gate_time_idx, :]
            else:
                print("No x0 for first gate time")
        else:
            print("Using x0 from previous optimized param")
            params['x0'] = [tg_initial] + drive_param_list[-1][1:]
    elif config['use_x0'] == 'from_input':
        print("Using x0 from input for initialization")
        if gate_type == 'CZ':
            params['x0'] = system_data['pulse_param'][gate_time_idx, :3]
        else:
            params['x0'] = config['x0_array'][gate_time_idx, :]
    else:
        print("No x0 used")

    print(f'params["x0"] = {params.get("x0", "None")}')
    result = sp.optimize.differential_evolution(**params)

    ut.print_time()
    print(f"Optimization completed, fidelity = {result.fun:.8f}")
    print(f"Optimized params = {result.x}")

    return result.fun, result.x.tolist()


# ==============================================================
# SAVE / RESUME / PLOT
# ==============================================================

def save_results(config, system_data, fidelity_list, fidelity_large_list,
                 drive_param_list, last_index):
    """Save current optimization results to both npz and csv."""
    result_file = config['result_file']
    csv_file = config['csv_file']

    np.savez(
        result_file,
        config=json.dumps(config, default=str),
        fidelity=np.array(fidelity_list),
        fidelity_large=np.array(fidelity_large_list),
        drive_params=np.array(drive_param_list, dtype=object),
        last_index=last_index,
    )
    print(f"[SAVE] npz saved to {result_file}")

    if len(drive_param_list) > 0:
        max_len = max(len(p) for p in drive_param_list)
        data = {}
        data['idx'] = np.arange(len(drive_param_list))
        data['f_trunc'] = fidelity_list
        if len(fidelity_large_list) == len(drive_param_list):
            data['f_large'] = fidelity_large_list
        for j in range(max_len):
            data[f'p{j}'] = [p[j] if j < len(p) else np.nan for p in drive_param_list]

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"[SAVE] csv saved to {csv_file}")


def load_resume_if_any(config):
    """Load previous results and continue from the next index when resume is enabled."""
    if not config.get('resume', False):
        return 0, [], [], []

    result_file = config['result_file']
    if not os.path.exists(result_file):
        print(f"[RESUME] No existing file {result_file}, starting from scratch.")
        return 0, [], [], []

    data = np.load(result_file, allow_pickle=True)
    fidelity_list = data.get('fidelity', np.array([])).tolist()
    fidelity_large_list = data.get('fidelity_large', np.array([])).tolist()
    drive_param_list = data.get('drive_params', np.array([], dtype=object)).tolist()
    last_index = int(data.get('last_index', 0))

    start_idx = last_index + 1
    print(f"[RESUME] Loaded from {result_file}, resume from index {start_idx}.")

    return start_idx, fidelity_list, fidelity_large_list, drive_param_list


def plot_results(config, fidelity_list, fidelity_large_list, drive_param_list):
    """
    Plot 3?1 panel:
        1. Fidelity vs gate time tg
        2. Drive amplitude vs gate time tg
        3. Detuning vs gate time tg

    CNOT parameters:
        0: tg
        1: drive_amp1
        2: drive_amp2
        3: detuning1
        4: detuning2

    CZ parameters:
        0: tg
        1: drive_amp1
        2: detuning1
    """
    if not config.get('do_plot', False):
        return

    plot_dir = config['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)

    if len(drive_param_list) == 0:
        print("[PLOT] No parameters to plot.")
        return

    arr = np.array(drive_param_list)   # shape: (N, num_params)
    tg = arr[:, 0]

    # -----------------------------
    # Fidelity subplot
    # -----------------------------
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))

    ax[0].set_title(f"{config['gate_type']} optimization results")

    # truncated fidelity
    fidelity = 10 ** np.array(fidelity_list)
    ax[0].plot(tg, fidelity, ".-", label="truncated")

    # large Hilbert fidelity
    if len(fidelity_large_list) == len(fidelity_list):
        fidelity_large = 10 ** np.array(fidelity_large_list)
        ax[0].plot(tg, fidelity_large, "o", label="large", alpha=0.8)

    ax[0].set_ylabel("Fidelity")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid()

    # -----------------------------
    # Drive amplitude subplot
    # -----------------------------
    gate_type = config["gate_type"].upper()

    if gate_type == "CNOT":
        # param1 = amp1; param2 = amp2
        ax[1].plot(tg, arr[:, 1], ".-", label="drive_amp1")
        ax[1].plot(tg, arr[:, 2], ".-", label="drive_amp2")
    else:  # CZ
        ax[1].plot(tg, arr[:, 1], ".-", label="drive_amp1")

    ax[1].set_ylabel("Drive amplitude")
    ax[1].legend()
    ax[1].grid()

    # -----------------------------
    # Detuning subplot
    # -----------------------------
    if gate_type == "CNOT":
        ax[2].plot(tg, arr[:, 3], ".-", label="detuning1")
        ax[2].plot(tg, arr[:, 4], ".-", label="detuning2")
    else:  # CZ
        ax[2].plot(tg, arr[:, 2], ".-", label="detuning1")

    ax[2].set_ylabel("Detuning")
    ax[2].set_xlabel("Gate time tg")
    ax[2].legend()
    ax[2].grid()

    # -----------------------------
    # Save figure
    # -----------------------------
    fname = os.path.join(plot_dir, f"{config['gate_type']}_summary.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[PLOT] Saved {fname}")


# ==============================================================
# PRINTING
# ==============================================================

def print_configuration_summary(sys, config):
    print("\n" + "=" * 60)
    print(f"{config['gate_type']} GATE FIDELITY OPTIMIZATION CONFIGURATION")
    print("=" * 60)
    print(f"Start time: {datetime.now(pytz.timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"gate_type = {config['gate_type']}")
    
    print(f"truc_large      = {config['truc_large']}")
    print(f"truc_optimize   = {config['truc_optimize']}")    
    print(f"reduced model   = {config['reduced_model']}")
    if config.get('reduced_model') == 'graph_pick':
        print(f"graph_model_name = {config['graph_model_name']}")
    
    print(f"max_step_ideal  = {config['max_step_ideal']}")
    print(f"max_step_noisy  = {config['max_step_noisy']}")

    print(f"workers         = {config['workers']}")
    print(f"popsize         = {config['popsize']}")
    print(f"recombination   = {config['recombination']}")
    print(f"tol             = {config['tol']}")
    print(f"mutation        = {config['mutation']}")
    print(f"resume          = {config['resume']}")
    print(f"do_plot         = {config['do_plot']}")
    
    print(f"use_x0         = {config['use_x0']}")
    print(f"first_x0_from_input = {config['first_x0_from_input']}")
    
    print(f"result_file     = {config['result_file']}")
    
    print(f"folder_pulse   = {config.get('folder_pulse', 'None')}")
    print(f"gate_time_indices = {config.get('gate_time_indices', 'None')}") 

    if config['gate_type'] == 'CZ':
        print(f"W_20_50         = {np.round(sys['W_20_50'], 3)}")
        print(f"amp_bound       = {config['amp_bound']}")
        print(f"detune_bound    = {config['detune_bound']}")
    else:
        print(f"mid_state       = {config['mid_state']}")
        print(f"W_0_2, W_1_2    = {np.round(sys['W_0_2'], 3)}, {np.round(sys['W_1_2'], 3)}")
        print(f"A1_bound        = {config['A1_bound']}")
        print(f"A2_bound        = {config['A2_bound']}")
        print(f"detune1_bound   = {config['detune1_bound']}")
        print(f"detune2_bound   = {config['detune2_bound']}")
        
        print(f"x0_array       = {config['x0_array']}")
        print(f"tg_opt_vec       = {config['tg_opt_vec']}")

    print(f"pulse_param = {sys['pulse_param']}")

    print("=" * 60)


# ==============================================================
# RUN SWEEP
# ==============================================================

def run_fidelity_sweep(system_data, hamiltonians, config):
    print("Starting fidelity sweep...")

    # resume
    start_idx, fidelity_list, fidelity_large_list, drive_param_list = load_resume_if_any(config)

    if config['gate_type'] == 'CZ':
        n_total = len(system_data['pulse_param'])
    else:
        n_total = len(config['tg_opt_vec'])
    print(f"pulse_param = {system_data['pulse_param']}")
    print(f"Total gate times to optimize: {n_total}, starting from index {start_idx}")
    
    for jdx in tqdm(range(start_idx, n_total), desc=f"{config['gate_type']} optimize"):
        f_trunc, params = optimize_single_gate_time(
            jdx, system_data, hamiltonians, config, drive_param_list
        )
        fidelity_list.append(f_trunc)
        drive_param_list.append(params)

        ut.print_pulse_params("param_optimized", drive_param_list)
        ut.print_fidelity(f"f_{config['truc_optimize']}", fidelity_list, num_digits=8)

        f_large = evaluate_fidelity_large(params, system_data, hamiltonians, config)
        fidelity_large_list.append(f_large)
        ut.print_fidelity(f"f_{config['truc_large']}", fidelity_large_list, num_digits=8)

        save_results(config, system_data, fidelity_list, fidelity_large_list,
                     drive_param_list, jdx)

    print("Fidelity sweep completed!")

    plot_results(config, fidelity_list, fidelity_large_list, drive_param_list)


# ==============================================================
# DIRECT FIDELITY CALCULATION (NO OPTIMIZATION)
# ==============================================================

def compute_fidelity_given_params(params, system_data, hamiltonians, config):
    """
        f_trunc   -- truncated Hilbert space fidelity (log10(error))
        f_large   -- large Hilbert space fidelity (log10(error))
    """
    gate_type = config["gate_type"].upper()

    # -------- truncated fidelity --------
    if gate_type == "CZ":
        args = [
            hamiltonians['H_drive_select'],
            system_data['W_20_50'],
            1,
            [],
            hamiltonians['logi_idx_select'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]

        f_trunc = ut.cz_fidelity_log_noise(params, *args)

    elif gate_type == "CNOT":
        args = [
            hamiltonians['H_drive_select'],
            system_data['W_0_2'],
            system_data['W_1_2'],
            1,
            [],
            hamiltonians['logi_idx_select'],
            config['mid_state'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        # print("params =", params)
        # print("args =", args)        
        f_trunc = ut.cnot_fidelity_log_noise(params, *args)

    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    # -------- large fidelity --------
    # f_large = evaluate_fidelity_large(params, system_data, hamiltonians, config)

    return f_trunc # , f_large


# ==============================================================
# BATCH DIRECT FIDELITY CALCULATION (NO OPTIMIZATION)
# ==============================================================

def compute_fidelity_batch(
    params_list,
    system_data,
    hamiltonians,
    config,
    show_progress=True,
    return_numpy=False,
):
    """
    Compute fidelities for multiple pulse parameter sets without optimization.

    Parameters
    ----------
    params_list : array-like, shape (N, P)
        List of pulse parameters.
        CZ:   [tg, amp, detune]
        CNOT: [tg, A1, A2, d1, d2]
    show_progress : bool, optional
        Whether to show a tqdm progress bar.

    return_numpy : bool, optional
        If True, return NumPy arrays instead of Python lists.

    Returns
    -------
    f_trunc_list : list or np.ndarray
        log10(error) fidelities in truncated Hilbert space.
    """

    params_array = np.asarray(params_list)
    if params_array.ndim != 2:
        raise ValueError("params_list must have shape (N, num_params)")

    iterator = params_array
    if show_progress:
        iterator = tqdm(iterator, desc="Computing fidelity batch")

    f_trunc_list = []

    for params in iterator:
        f_trunc = compute_fidelity_given_params(
            params.tolist(),
            system_data,
            hamiltonians,
            config,
        )
        f_trunc_list.append(f_trunc)

    if return_numpy:
        return np.array(f_trunc_list)
    return f_trunc_list

# ==============================================================
# CONFIGURATION
# ==============================================================

def get_optimization_config(gate_type="CNOT", custom_config=None):
    """
    Return the unified configuration dictionary.

    Parameters
    ----------
    gate_type : {"CZ", "CNOT"}
        Type of 2-qubit gate.
    custom_config : dict or None
        If provided, overrides entries in the default config.
    """

    config = dict(
        gate_type=gate_type,

        # truncation
        truc_large=1000,
        truc_optimize=220,
        use_truc_model=False,
        truc_model_name='cz_short_500_detune1',
        reduced_model='charge_pick',
        # charge_model_name="n_theta_dress_charge_truc",

        # qutip options
        max_step_ideal=1e-3,
        max_step_noisy=1e-3,

        # differential evolution parameters
        workers=60,
        popsize=10,
        recombination=0.7,
        tol=0.01,
        mutation=(0.5, 1.0),

        # resume & saving
        resume=False,
        do_plot=True,

        # x0 source: None / 'from_neighbor' / 'from_input'
        use_x0='from_neighbor',
        first_x0_from_input=True,
        
        # folder_load='../../data/_truc_3000',
        folder_load = '../data/December_17_2025_Sorted_Untruc'
    )

    # =====================================================
    # Add timestamped directory (this is the effective path)
    # =====================================================    
    
    beijing_tz = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")

    label_str = f"{gate_type.lower()}_{timestamp}"
    base_dir = f"data/results/{label_str}"
    result_file = os.path.join(base_dir, f"{label_str}.npz")
    csv_file = os.path.join(base_dir, f"{label_str}.csv")
    os.makedirs(base_dir, exist_ok=True)

    config.update(dict(
        timestamp=timestamp,
        result_dir=base_dir,
        result_file=result_file,
        csv_file=csv_file,
        plot_dir=base_dir,
    ))
    # =====================================================

    # ===== CZ-specific parameters =====
    if gate_type == "CZ":
        config.update(dict(
            truc_large=50,
            truc_optimize=50,
            folder_load='../data/Two_qubit_data_Sorted_Truc',
            folder_pulse='../figure/data/data_cz_fidelity_npz_full.txt',
            gate_time_indices=[0, 1],

            tg_bound=(-0.01, 0.01),
            amp_bound=(0.01, 0.05),
            detune_bound=(0.001, 0.04),
        ))

    # ===== CNOT-specific parameters =====
    elif gate_type == "CNOT":
        # Here we follow your old CNOT script style (truncations and related parameters)
        config.update(dict(
            
            mid_state='8-2',
            
            # data-related parameters (kept for compatibility with your previous scripts)
            # truc1=300,
            # truc_tot=1000,
            # truc_full=1000,
            # charge_pick=True,

            # If you have a pulse file, you can enable it and specify path and indices
            # folder_pulse='../figure/data/data_cnot_fidelity_3ncut.txt',
            # gate_time_indices= [0, 15, -1], # np.arange(3).tolist(),  #[0],

            # CNOT parameter bounds (you can adjust as needed)
            tg_bound=(-0.01, 0.01),
            A1_bound=(0.001, 0.1),
            A2_bound=(0.001, 0.1),
            detune1_bound=(-0.05, -0.00001),
            detune2_bound=(-0.05, -0.00001),

            # Initial x0 vector(s) (can be extended to multiple rows for multiple gate times)
            # x0_array=np.array([
            #     [200.00347, 0.0213, 0.0139, -0.005, -0.005],
            # ]),
            # tg_opt_vec=np.arange(200, 250, 2).tolist(),  
            x0_array=np.array([
# [29.998501,0.096133,0.070563,-0.035335,-0.032721] , # good
[29.998501,0.086133,0.070563,-0.035335,-0.032721] , # random
# [300.00454,0.018764,0.012331,-0.001994,-0.001895],
# [149.998717,0.039313,0.025856,-0.008901,-0.00847],
# [149.9928624, 0.0390348, 0.02497799, -0.0085582, -0.00815811],
                # [200.00347, 0.0213, 0.0139, -0.005, -0.005],
            ]),
            tg_opt_vec=np.arange(30, 31, step=10).tolist(),             
        ))
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    if custom_config:
        config.update(custom_config)

    return config

# ==============================================================
# MAIN
# ==============================================================

def main():
    print(f"Starting {os.path.basename(__file__)}")
    ut.print_time()

    # ===== choose gate type here: 'CZ' or 'CNOT' =====
    # gate_type = "CZ"
    gate_type = "CNOT"

    # Override some configuration entries here if needed
    custom_config = {
        # "resume": True,
        # "do_plot": False,
    }
    config = get_optimization_config(gate_type=gate_type, custom_config=custom_config)

    # Step 1: Load system data
    system_data = load_system_data(config)

    # Step 2: Build Hamiltonians
    hamiltonians = build_hamiltonians(system_data, config)

    # Step 3: Print configuration summary
    print_configuration_summary(system_data, config)

    # Step 4: Run sweep
    run_fidelity_sweep(system_data, hamiltonians, config)
    # print("system_data['hspace_full'] = ", system_data['hspace_full'])
    # print("system_data['eval_tot'] = ", system_data['eval_tot'][:10])

    ### Direct fidelity calculation example (no optimization)    
    # cnot = pd.read_csv('data/cnot_fidelity_npz.txt')
    # params_list = cnot[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].values.tolist()[:2]
    # f_trunc = compute_fidelity_batch(
    #     params_list,
    #     system_data,
    #     hamiltonians,
    #     config,
    # )
    # ut.print_fidelity(f'f_{n_truc}', f_trunc, num_digits=8)
    # print("Truncated fidelities:", f_trunc)

    ut.print_time()
    # print("\n" + "=" * 60)
    # print("OPTIMIZATION COMPLETED")
    # print("=" * 60)

if __name__ == '__main__':
    main()
