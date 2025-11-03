#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
CZ Gate Fidelity Optimization for Two-Qubit Zero-Pi Systems
Modular (non-class) version.
Implements fidelity optimization for CZ gates using differential evolution.
All class structures removed; logic is now organized into standalone functions.
Author: Converted from cz_fidelity_optimize.py
Date: 2025
"""
# ===== Standard Library Imports =====
import os
import sys
from datetime import datetime
import pytz
# ===== Third-Party Imports =====
import numpy as np
import scipy as sp
from tqdm import tqdm
import scqubits.settings as settings
import qutip as qt
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sympy import symbols
# ===== Local Imports =====
sys.path.append('../')
import utils_2Q_gate_zp as ut
import ham_data as hd
# Configure scqubits
settings.OVERLAP_THRESHOLD = 0.3


# ==============================================================
# SYSTEM INITIALIZATION
# ==============================================================
def load_system_data(config):
    """
    Load system data and construct required elements for optimization.
    Returns a dictionary with all necessary system components.
    """
    print("Loading system data...")
    (hspace_full, eket_tot, eval_tot, n_theta0_dress, n_theta1_dress,
     hspace_0, hspace_1, logi_state) = hd.load_two_qubit_data(config['folder_load'], return_full=False)
    drive_term = n_theta1_dress
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    # Select Hilbert space for optimization
    if config['use_truc_model']:
        hspace_select = ut.truc_model[config['truc_model_name']][:config['truc_optimize']]
    else:
        hspace_select = hspace_full[:config['truc_optimize']]
    option_ideal, option_noisy = ut.get_qutip_options(
        config['max_step_ideal'], config['max_step_noisy']
    )
    x0_vec = ut.load_drive_params_2q(config['cz_run'])[config['gate_time_indices'], :]
    print(f"Loaded system data with {len(hspace_full)} total states")
    print(f"Using {len(hspace_select)} states for optimization")
    print(f"Transition frequency W_20_50 = {np.round(W_20_50, 3)}")
    return dict(
        hspace_full=hspace_full,
        eket_tot=eket_tot,
        eval_tot=eval_tot,
        drive_term=drive_term,
        logi_state=logi_state,
        hspace_select=hspace_select,
        W_20_50=W_20_50,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        x0_vec=x0_vec
    )
def build_hamiltonians(system_data, config):
    """Construct Hamiltonians for truncated and large Hilbert spaces."""
    print("Building Hamiltonians...")
    hspace_full = system_data['hspace_full']
    hspace_select = system_data['hspace_select']
    eval_tot = system_data['eval_tot']
    eket_tot = system_data['eket_tot']
    drive_term = system_data['drive_term']
    logi_state = system_data['logi_state']
    logi_idx_select = [hspace_select.index(i) for i in logi_state]
    index_select = [hspace_full.index(i) for i in hspace_select]
    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        config['cz_run'], index_select, eval_tot, eket_tot, drive_term
    )
    hspace_large = hspace_full[:config['truc_large']]
    logi_idx_large = [hspace_large.index(i) for i in logi_state]
    idx_large = np.arange(config['truc_large']).tolist()
    H_drive_large, _ = ut.build_hamiltonian_2q(
        config['cz_run'], idx_large, eval_tot, eket_tot, drive_term
    )
    print("Hamiltonians built successfully")
    return dict(
        H_drive_select=H_drive_select,
        H_drive_large=H_drive_large,
        logi_idx_select=logi_idx_select,
        logi_idx_large=logi_idx_large
    )
# ==============================================================
# OPTIMIZATION FUNCTIONS
# ==============================================================
def optimize_single_gate_time(
    gate_time_idx, system_data, hamiltonians, config
):
    """Optimize fidelity for a single gate time."""
    tg_initial = system_data['x0_vec'][gate_time_idx, 0]
    tg_bounds = (tg_initial + config['tg_bound'][0], tg_initial + config['tg_bound'][1])
    bounds = (tg_bounds, config['amp_bound'], config['detune_bound'])
    args_truc = [
        hamiltonians['H_drive_select'], system_data['W_20_50'], 1, [],
        hamiltonians['logi_idx_select'], system_data['option_ideal'], system_data['option_noisy']
    ]
    print(f"Optimizing gate time {gate_time_idx} (tg_initial = {tg_initial:.6f})")
    ut.print_time()
    result = sp.optimize.differential_evolution(
        func=ut.cz_fidelity_log_noise,
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
        x0=system_data['x0_vec'][gate_time_idx, :3],
        polish=False,
    )
    ut.print_time()
    print(f"Optimization completed for gate time {gate_time_idx}")
    print(f"Optimized fidelity: {result.fun:.8f}")
    print(f"Optimized parameters: {result.x}")
    return result.fun, result.x.tolist()
def run_fidelity_sweep(system_data, hamiltonians, config):
    """Run fidelity optimization sweep across all gate times."""
    print("Starting fidelity optimization sweep...")
    fidelity_list, drive_param_list = [], []
    for jdx, tg in tqdm(enumerate(system_data['x0_vec'][:, 0]), desc="Optimizing gate times"):
        fidelity, drive_params = optimize_single_gate_time(jdx, system_data, hamiltonians, config)
        fidelity_list.append(fidelity)
        drive_param_list.append(drive_params)
        print_intermediate_results(system_data, fidelity_list, drive_param_list, jdx, config)
    print("Fidelity optimization sweep completed!")
    return fidelity_list, drive_param_list

# ==============================================================
# PRINTING AND REPORTING
# ==============================================================

def print_configuration_summary(system_data, config):
    """Print a summary of the current configuration."""
    print("\n" + "=" * 60)
    print("CZ FIDELITY OPTIMIZER CONFIGURATION")
    print("=" * 60)
    print(f"Start time: {datetime.now(pytz.timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"\nTruncation parameters:")
    print(f"  - Large truncation: {config['truc_large']}")
    print(f"  - Optimization truncation: {config['truc_optimize']}")
    print(f"\nOptimization bounds:")
    print(f"  - Amplitude bounds: {config['amp_bound']}")
    print(f"  - Detuning bounds: {config['detune_bound']}")
    print(f"  - Gate time bounds: {config['tg_bound']}")
    print(f"\nDifferential evolution parameters:")
    print(f"  - Workers: {config['workers']}")
    print(f"  - Population size: {config['popsize']}")
    print(f"  - Recombination: {config['recombination']}")
    print(f"  - Tolerance: {config['tol']}")
    print(f"  - Mutation: {config['mutation']}")
    print(f"\nSystem parameters:")
    print(f"  - Transition frequency W_20_50: {np.round(system_data['W_20_50'], 3)}")
    print(f"  - Number of gate times to optimize: {len(system_data['x0_vec'])}")
    print(f"  - Hilbert space size (optimization): {len(system_data['hspace_select'])}")
    print(f"  - Hilbert space size (total): {len(system_data['hspace_full'])}")

    print('params:')
    for i in system_data['x0_vec']:
        print(i.tolist(), ',')
    print(system_data['x0_vec'])

    print("=" * 60)

def print_intermediate_results(system_data, fidelity_list, drive_param_list, current_idx, config):
    """Print intermediate optimization results."""
    print(f"\n--- Results after {current_idx + 1} optimizations ---")
    print(f"F_{config['truc_optimize']} = np.array([")
    for i in range(0, len(fidelity_list), 4):
        print(', '.join(map(str, np.round(fidelity_list[i:i+4], 8))), ',')
    print("])")
    print(f"param_{config['truc_optimize']} = np.array([")
    for params in drive_param_list:
        print(np.round(params, 6).tolist(), ',')
    print("])")

# ==============================================================
# CONFIGURATION AND SETUP
# ==============================================================
def get_default_config():
    """Return default configuration parameters for the optimization."""
    return {
        'truc_large': 1000,
        'truc_optimize': 200,
        'use_truc_model': False,
        'truc_model_name': 'cz_short_500_detune1',
        'max_step_ideal': 1e-3,
        'max_step_noisy': 1e-3,
        'amp_bound': (0., 0.1),
        'detune_bound': (-0.1, 0.1),
        'tg_bound': (-0.01, 0.01),
        'workers': 50,
        'popsize': 10,
        'recombination': 0.7,
        'tol': 0.01,
        'mutation': (0.5, 1.5),
        'folder_load': '../../data/_truc_3000',
        'cz_run': True,
        'gate_time_indices': np.arange(181)[0::6].tolist(),
    }

def setup_optimization_params(config):
    """Extract and return optimization parameters from config."""
    return {
        'truc_large': config['truc_large'],
        'truc_optimize': config['truc_optimize'],
        'use_truc_model': config['use_truc_model'],
        'truc_model_name': config['truc_model_name'],
        'max_step_ideal': config['max_step_ideal'],
        'max_step_noisy': config['max_step_noisy'],
        'amp_bound': config['amp_bound'],
        'detune_bound': config['detune_bound'],
        'tg_bound': config['tg_bound'],
        'workers': config['workers'],
        'popsize': config['popsize'],
        'recombination': config['recombination'],
        'tol': config['tol'],
        'mutation': config['mutation'],
        'folder_load': config['folder_load'],
        'cz_run': config['cz_run'],
        'gate_time_indices': config['gate_time_indices'],
    }

# ==============================================================
# MAIN EXECUTION
# ==============================================================

def main():
    """Main function to run CZ fidelity optimization."""
    print(f"Starting {os.path.basename(__file__)}")
    ut.print_time()
    config = get_default_config()
    params = setup_optimization_params(config)
    system_data = load_system_data(params)
    hamiltonians = build_hamiltonians(system_data, params)
    print_configuration_summary(system_data, params)
    run_fidelity_sweep(system_data, hamiltonians, params)
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    main()