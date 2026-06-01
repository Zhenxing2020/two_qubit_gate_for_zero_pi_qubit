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
from pathlib import Path

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)
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

    This function loads the two-qubit data, selects the Hilbert space
    (either truncated or full), prepares the drive term and transition
    frequency, and initializes optimization parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model, truncation, and file paths.

    Returns
    -------
    dict
        A dictionary containing all necessary system components for optimization:
        - hspace_full : list
            Full Hilbert space basis labels.
        - eket_tot : ndarray
            Eigenkets of the total system.
        - eval_tot : ndarray
            Eigenvalues of the total system.
        - drive_term : ndarray
            Drive operator (matrix form).
        - logi_state : list
            Logical state labels.
        - hspace_select : list
            Truncated Hilbert space used for optimization.
        - W_20_50 : float
            Transition frequency between |5-0⟩ and |2-0⟩.
        - option_ideal, option_noisy : qutip.Options
            Solver options for ideal and noisy simulations.
        - pulse_param : ndarray
            Initial drive parameters for each gate time index.
    """
    print("Loading system data...")

    # Load two-qubit system data (returns eigenstates, energies, etc.)
    (hspace_full, eket_tot, eval_tot, _, _, _, n_theta1_dress, _, _,
     _, _, logi_state) = hd.load_two_qubit_data(config['folder_load'], return_full=False)

    # Use the dressed theta_1 term as the drive term
    drive_term = n_theta1_dress

    # Compute energy difference (transition frequency) between specific states
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    # Choose Hilbert space truncation: truncated model or full model
    if config['use_truc_model']:
        hspace_select = ut.truc_model[config['truc_model_name']][:config['truc_optimize']]
    else:
        hspace_select = hspace_full[:config['truc_optimize']]

    # Generate qutip solver options for ideal and noisy simulations
    option_ideal, option_noisy = ut.get_qutip_options(
        config['max_step_ideal'], config['max_step_noisy']
    )

    # Load initial drive parameters corresponding to gate time indices
    pulse_param = ut.load_drive_params_2q(config['cz_run'], 
                                          folder=config['folder_pulse'])[config['gate_time_indices'], :]

    # Print summary information
    print(f"Loaded system data with {len(hspace_full)} total states")
    print(f"Using {len(hspace_select)} states for optimization")
    print(f"Transition frequency W_20_50 = {np.round(W_20_50, 3)}")

    # Return all necessary system components in a dictionary
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
        pulse_param=pulse_param
    )


def build_hamiltonians(sys, config):
    """
    Construct Hamiltonians for truncated and large Hilbert spaces.

    Parameters
    ----------
    sys : dict
        System data dictionary returned from `load_system_data`.
    config : dict
        Configuration dictionary containing truncation sizes and run options.

    Returns
    -------
    dict
        A dictionary containing:
        - H_drive_select : ndarray
            Drive Hamiltonian for the truncated Hilbert space.
        - H_drive_large : ndarray
            Drive Hamiltonian for the large (reference) Hilbert space.
        - logi_idx_select : list[int]
            Indices of logical states within the truncated basis.
        - logi_idx_large : list[int]
            Indices of logical states within the large basis.
    """
    print("Building Hamiltonians...")

    # Identify indices of logical states in the truncated Hilbert space
    logi_idx_select = [sys['hspace_select'].index(i) for i in sys['logi_state']]

    # Indices of selected Hilbert space within the full space
    index_select = [sys['hspace_full'].index(i) for i in sys['hspace_select']]

    # Build drive Hamiltonian for truncated space
    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        config['cz_run'], index_select, sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    # Define the large Hilbert space truncation
    hspace_large = sys['hspace_full'][:config['truc_large']]

    # Logical state indices and full index list for large space
    logi_idx_large = [hspace_large.index(i) for i in sys['logi_state']]
    idx_large = np.arange(config['truc_large']).tolist()

    # Build drive Hamiltonian for the large Hilbert space
    H_drive_large, _ = ut.build_hamiltonian_2q(
        config['cz_run'], idx_large, sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    print("Hamiltonians built successfully")

    # Return both Hamiltonians and their logical indices
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
    gate_time_idx, system_data, hamiltonians, config, drive_param_list
):
    """
    Optimize gate fidelity for a single gate time using differential evolution.

    Parameters
    ----------
    gate_time_idx : int
        Index of the gate time to be optimized.
    system_data : dict
        Dictionary containing system data and initial parameters.
    hamiltonians : dict
        Dictionary containing Hamiltonians and logical state indices.
    drive_param_list : list
        List of drive parameters for initialization.
    config : dict
        Configuration dictionary with optimization parameters.

    Returns
    -------
    tuple
        (fidelity_value, optimized_parameters)
        fidelity_value : float
            The optimized fidelity (objective function value).
        optimized_parameters : list[float]
            The optimized gate parameters [tg, amp, detune].
    """
    # Extract initial gate time and define search bounds
    tg_initial = system_data['pulse_param'][gate_time_idx, 0]
    tg_bounds = (tg_initial + config['tg_bound'][0], tg_initial + config['tg_bound'][1])
    bounds = (tg_bounds, config['amp_bound'], config['detune_bound'])

    # Prepare arguments for fidelity function evaluation
    args_truc = [
        hamiltonians['H_drive_select'], system_data['W_20_50'], 1, [],
        hamiltonians['logi_idx_select'], system_data['option_ideal'], system_data['option_noisy']
    ]

    print(f"\n--- Optimizing gate time index {gate_time_idx} (tg = {tg_initial:.6f}) ---")
    ut.print_time()

    params = dict(
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
        polish=False,
    )
    
    if config['use_x0'] == 'from_neighbor':
        if gate_time_idx == 0:
            if config['first_x0_from_input']:
                print("Using first x0 from input for initialization (first gate time)")
                params['x0'] = system_data['pulse_param'][gate_time_idx, :3]
            else:
                print("No x0 for first gate time")
        else:
            print("Using first x0 from neighbor for initialization")
            params['x0'] = [tg_initial] + drive_param_list[-1][1:]
        
    elif config['use_x0'] == 'from_input':
        print("Using x0 from input for initialization")
        params['x0'] = system_data['pulse_param'][gate_time_idx, :3]
    else:
        print("No x0 for all gate times")

    result = sp.optimize.differential_evolution(**params)

    ut.print_time()
    print(f"Optimization completed for gate time {gate_time_idx}")
    print(f"Optimized fidelity: {result.fun:.8f}")
    print(f"Optimized parameters: {result.x}")

    # Return the optimized fidelity and parameters as a tuple
    return result.fun, result.x.tolist()


def run_fidelity_sweep(system_data, hamiltonians, config):
    """
    Run fidelity optimization across all gate times.

    Iterates through each gate time index, performs independent optimization,
    and stores fidelity and optimized parameters. Intermediate results are
    printed and optionally saved at each step.

    Parameters
    ----------
    system_data : dict
        Dictionary containing system and initialization data.
    hamiltonians : dict
        Dictionary containing Hamiltonians and logical indices.
    config : dict
        Configuration dictionary with optimization settings.

    Returns
    -------
    tuple
        (fidelity_list, drive_param_list)
        fidelity_list : list[float]
            List of optimized fidelities for each gate time.
        drive_param_list : list[list[float]]
            Corresponding optimized drive parameters.
    """
    print("Starting fidelity optimization sweep...")

    # Containers for fidelity and optimized drive parameters
    fidelity_list, drive_param_list = [], []
    fidelity_large_list = []

    # Iterate over all gate times to optimize
    if config['tg_reverse']:
        system_data['pulse_param'] = system_data['pulse_param'][::-1]

    for jdx in tqdm(range(len(system_data['pulse_param'])), desc="Optimizing gate times"):
        
        fidelity, drive_params = optimize_single_gate_time(jdx, system_data, hamiltonians, config, drive_param_list)
        fidelity_list.append(fidelity)
        drive_param_list.append(drive_params)

        # Print or save intermediate results
        print_intermediate_results(fidelity_list, drive_param_list, jdx, config)

        # Evaluate fidelity in the large Hilbert space for reference
        ut.print_time()
        fidelity_large = check_pulse_in_large(drive_params, system_data, hamiltonians, config)
        fidelity_large_list.append(fidelity_large)   
        ut.print_fidelity(f'f_{config["truc_large"]}', fidelity_large_list, num_digits=8)
        
        ut.print_time()

    print("Fidelity optimization sweep completed!")


def check_pulse_in_large(drive_params, system_data, hamiltonians, config):
    """
    Check pulse fidelity in the large Hilbert space.

    Parameters
    ----------
    drive_params : list
        Drive parameters [tg, drive_amp, detune].

    Returns
    -------
    float
        Fidelity value in the large Hilbert space.
    """
    tg, drive_amp, detune = drive_params
    n_cpu_parallel = 16

    arg_all = [tg, drive_amp, detune,
               n_cpu_parallel, np.arange(config['truc_large']), system_data['W_20_50'], 
               hamiltonians['H_drive_large'], hamiltonians['logi_idx_large']]
    fidelity_large = ut.cz_fidelity_log_old(arg_all)
    return fidelity_large

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
    print(f"  - Number of gate times to optimize: {len(system_data['pulse_param'])}")
    print(f"  - Hilbert space size (optimization): {len(system_data['hspace_select'])}")
    print(f"  - Hilbert space size (total): {len(system_data['hspace_full'])}")
    print(f"  - reverse tg optimization: {config['tg_reverse']}")
    print(f"  - use_x0: {config['use_x0']}")
    print(f"  - folder pulse: {config['folder_pulse']}")
    ut.print_pulse_params(f"param_input", system_data['pulse_param'])        

    print("=" * 60)

def print_intermediate_results(fidelity_list, drive_param_list, current_idx, config):
    """Print intermediate optimization results."""
    print(f"\n--- Results after {current_idx + 1} optimizations ---")
        
    ut.print_pulse_params(f"param_optimized", drive_param_list)        
    ut.print_fidelity(f'f_{config["truc_optimize"]}', fidelity_list, num_digits=8)        

# ==============================================================
# CONFIGURATION AND SETUP
# ==============================================================

def get_optimization_config(custom_config=None):
    """
    Return the optimization parameter configuration.

    If `custom_config` (dict) is provided, it overrides the corresponding
    entries in the default configuration.

    Parameters
    ----------
    custom_config : dict, optional
        User-specified configuration values to override defaults.

    Returns
    -------
    config : dict
        The final optimization parameter configuration.
    """
    # Default configuration
    config = {
        'truc_large': 50, # 1000,
        'truc_optimize': 50, # 300,
        'use_truc_model': False,
        'truc_model_name': 'cz_short_500_detune1',
        'max_step_ideal': 1e-3,
        'max_step_noisy': 1e-3,
        'tg_bound': (-0.01, 0.01),
        'workers': 5,
        'popsize': 10,
        'recombination': 0.7,
        'tol': 0.01,
        'mutation': (0.5, 1),
        'folder_load': '../data/Two_qubit_data_Sorted_Truc',
        'cz_run': True,
        'folder_pulse': 'data/npz/cz_pulse_neighbor.txt',
        'tg_reverse': False,
        'use_x0': 'from_input',  # Options: None, 'from_neighbor', 'from_input'
        # if use 'from_neighbor', the first one will use from input, make sure it gives nice fidelity
        'first_x0_from_input': True,
        
        # 'gate_time_indices': np.arange(10,20).tolist(),
        # 'amp_bound': (0.035, 0.045), # (0., 0.1),
        # 'detune_bound': (0.02, 0.1), # (-0.1, -0.05),
        
        # 'gate_time_indices': np.arange(20,30).tolist(),
        # 'amp_bound': (0.01, 0.1), 
        # 'detune_bound': (0.03, 0.1),                  
        
        # 'gate_time_indices': (np.arange(144,175)-20).tolist(),
        'gate_time_indices': [0,1], # (np.arange(20,50) - 20).tolist(),
        'amp_bound': (0.01, 0.05), 
        'detune_bound': (0.001, 0.04),                    
    }

    # Override defaults with user-provided configuration
    if custom_config:
        config.update(custom_config)

    return config

# ==============================================================
# MAIN EXECUTION
# ==============================================================

def main():
    """Main function to run CZ fidelity optimization."""
    print(f"Starting {os.path.basename(__file__)}")
    ut.print_time()

    # Step 1: Load optimization configuration (default + user overrides)
    config = get_optimization_config()

    # Step 2: Load system data including eigenstates, energies, logical states, etc.
    system_data = load_system_data(config)

    # Step 3: Build Hamiltonians for both truncated and large Hilbert spaces
    hamiltonians = build_hamiltonians(system_data, config)

    # Step 4: Print summary of configuration and loaded system parameters
    print_configuration_summary(system_data, config)

    # Step 5: Run fidelity optimization sweep across all specified gate times
    run_fidelity_sweep(system_data, hamiltonians, config)

    print_configuration_summary(system_data, config)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    main()