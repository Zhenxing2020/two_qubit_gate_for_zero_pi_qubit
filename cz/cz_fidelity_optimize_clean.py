#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
CZ Gate Fidelity Optimization for Two-Qubit Zero-Pi Systems

This module implements fidelity optimization for CZ gates in two-qubit zero-pi systems
using differential evolution optimization. It sweeps through different gate times
and optimizes drive parameters (amplitude and detuning) to maximize gate fidelity.

Author: Generated from original cz_fidelity_optimize.py
Date: 2024
"""

# Standard library imports
import os
import sys
from datetime import datetime
import pytz

# Third-party imports
import numpy as np
import scipy as sp
import scipy.sparse as ssp
from tqdm import tqdm
import scqubits as scq
import scqubits.settings as settings
import qutip as qt
from qutip.qip.operations import rz, cz_gate
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from joblib import Parallel, delayed
import itertools
from sympy import symbols

# Local imports
sys.path.append('../')
import utils_2Q_gate_zp as ut
import ham_data as hd

# Configure scqubits settings
settings.OVERLAP_THRESHOLD = 0.3


class CZFidelityOptimizer:
    """
    A class for optimizing CZ gate fidelity in two-qubit zero-pi systems.
    
    This optimizer uses differential evolution to find optimal drive parameters
    (amplitude and detuning) for different gate times to maximize gate fidelity.
    """
    
    def __init__(self, config=None):
        """
        Initialize the CZ fidelity optimizer.
        
        Args:
            config (dict, optional): Configuration dictionary containing optimization
                                   parameters. If None, default values are used.
        """
        # Default configuration
        self.config = config or self._get_default_config()
        
        # Initialize optimization parameters
        self._setup_optimization_params()
        
        # Initialize system data
        self._load_system_data()
        
        # Build Hamiltonians
        self._build_hamiltonians()
        
    def _get_default_config(self):
        """Get default configuration parameters."""
        return {
            # Truncation parameters
            'truc_large': 1000,
            'truc_optimize': 200,
            
            # Model parameters
            'use_truc_model': False,
            'truc_model_name': 'cz_short_500_detune1',
            
            # Solver parameters
            'max_step_ideal': 1e-3,
            'max_step_noisy': 1e-3,
            
            # Optimization bounds
            'amp_bound': (0., 0.1),
            'detune_bound': (-0.1, 0.1),
            'tg_bound': (-0.01, 0.01),
            
            # Differential evolution parameters
            'workers': 50,
            'popsize': 10,
            'recombination': 0.7,
            'tol': 0.01,
            'mutation': (0.5, 1.5),
            
            # Data loading
            'folder_load': '../../data/_truc_3000',
            'cz_run': True,
            
            # Gate time selection
            'gate_time_indices': np.arange(31)[1::3].tolist(),
            # [0, 1, 2],  # Select specific gate times
        }
    
    def _setup_optimization_params(self):
        """Setup optimization parameters from configuration."""
        self.truc_large = self.config['truc_large']
        self.truc_optimize = self.config['truc_optimize']
        self.use_truc_model = self.config['use_truc_model']
        self.truc_model_name = self.config['truc_model_name']
        self.max_step_ideal = self.config['max_step_ideal']
        self.max_step_noisy = self.config['max_step_noisy']
        
        # Optimization bounds
        self.amp_bound = self.config['amp_bound']
        self.detune_bound = self.config['detune_bound']
        self.tg_bound = self.config['tg_bound']
        
        # Differential evolution parameters
        self.workers = self.config['workers']
        self.popsize = self.config['popsize']
        self.recombination = self.config['recombination']
        self.tol = self.config['tol']
        self.mutation = self.config['mutation']
        
        # Data parameters
        self.folder_load = self.config['folder_load']
        self.cz_run = self.config['cz_run']
        self.gate_time_indices = self.config['gate_time_indices']
    
    def _load_system_data(self):
        """
        Load system data including Hamiltonian, eigenstates, and logical states.
        
        This method loads the precomputed system data from files and extracts
        relevant parameters for optimization.
        """
        print("Loading system data...")
        
        # Load two-qubit data
        (self.hspace_full, self.eket_tot, self.eval_tot, self.n_theta0_dress,
         self.n_theta1_dress, self.hspace_0, self.hspace_1, self.logi_state) = \
            hd.load_two_qubit_data(self.folder_load, return_full=False)
        
        # Set drive term
        self.drive_term = self.n_theta1_dress
        
        # Calculate transition frequency
        self.W_20_50 = (self.eval_tot[self.hspace_full.index('5-0')] - 
                       self.eval_tot[self.hspace_full.index('2-0')])
        
        # Select Hilbert space for optimization
        if self.use_truc_model:
            self.hspace_select = ut.truc_model[self.truc_model_name][:self.truc_optimize]
        else:
            self.hspace_select = self.hspace_full[:self.truc_optimize]
        
        # Get QuTiP solver options
        self.option_ideal, self.option_noisy = ut.get_qutip_options(
            self.max_step_ideal, self.max_step_noisy)
        
        # Load initial drive parameters
        self.x0_vec = ut.load_drive_params_2q(self.cz_run)[self.gate_time_indices, :]
        
        print(f"Loaded system data with {len(self.hspace_full)} total states")
        print(f"Using {len(self.hspace_select)} states for optimization")
        print(f"Transition frequency W_20_50 = {np.round(self.W_20_50, 3)}")
    
    def _build_hamiltonians(self):
        """
        Build Hamiltonians for both truncated and full system.
        
        This method constructs the drive Hamiltonians needed for optimization
        and full fidelity calculation.
        """
        print("Building Hamiltonians...")
        
        # Build Hamiltonian for optimization (truncated)
        self.logi_idx_select = [self.hspace_select.index(i) for i in self.logi_state]
        self.index_select = [self.hspace_full.index(i) for i in self.hspace_select]
        
        self.H_drive_select, self.eket_truc = ut.build_hamiltonian_2q(
            self.cz_run, self.index_select, self.eval_tot, 
            self.eket_tot, self.drive_term)
        
        # Build Hamiltonian for full fidelity calculation
        self.hspace_large = self.hspace_full[:self.truc_large]
        self.logi_idx_large = [self.hspace_large.index(i) for i in self.logi_state]
        self.idx_large = np.arange(self.truc_large).tolist()
        
        self.H_drive_large, _ = ut.build_hamiltonian_2q(
            self.cz_run, self.idx_large, self.eval_tot, 
            self.eket_tot, self.drive_term)
        
        print("Hamiltonians built successfully")
    
    def optimize_single_gate_time(self, gate_time_idx):
        """
        Optimize fidelity for a single gate time.
        
        Args:
            gate_time_idx (int): Index of the gate time to optimize
            
        Returns:
            tuple: (fidelity, optimized_parameters) where fidelity is the optimized
                   log gate error and optimized_parameters is [tg, amp, detune]
        """
        # Get initial parameters for this gate time
        tg_initial = self.x0_vec[gate_time_idx, 0]
        
        # Set bounds for this specific gate time
        tg_bounds = (tg_initial + self.tg_bound[0], tg_initial + self.tg_bound[1])
        bounds = (tg_bounds, self.amp_bound, self.detune_bound)
        
        # Prepare arguments for optimization function
        args_truc = [
            self.H_drive_select, self.W_20_50, 1, [],  # n_cpu_optimize=1, c_op_list=[]
            self.logi_idx_select, self.option_ideal, self.option_noisy
        ]
        
        print(f"Optimizing gate time {gate_time_idx} (tg_initial = {tg_initial:.6f})")
        ut.print_time()
        
        # Run differential evolution optimization
        result = sp.optimize.differential_evolution(
            func=ut.cz_fidelity_log_noise,
            bounds=bounds,
            args=args_truc,
            disp=True,
            callback=ut.print_soln,
            init="sobol",
            workers=self.workers,
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            tol=self.tol,
            x0=self.x0_vec[gate_time_idx, :3],
            polish=False,  # Set to False to avoid breaking the loop
        )
        
        ut.print_time()
        
        print(f"Optimization completed for gate time {gate_time_idx}")
        print(f"Optimized fidelity: {result.fun:.8f}")
        print(f"Optimized parameters: {result.x}")
        
        return result.fun, result.x.tolist()
    
    def run_fidelity_sweep(self):
        """
        Run fidelity optimization sweep across all gate times.
        
        This method performs optimization for each gate time and collects results.
        
        Returns:
            tuple: (fidelity_list, drive_param_list) containing optimization results
        """
        print("Starting fidelity optimization sweep...")
        print(f"Optimizing {len(self.x0_vec)} gate times")
        
        fidelity_list = []
        drive_param_list = []
        
        # Optimize each gate time
        for jdx, tg in tqdm(enumerate(self.x0_vec[:, 0]), 
                           desc="Optimizing gate times"):
            fidelity, drive_params = self.optimize_single_gate_time(jdx)
            
            fidelity_list.append(fidelity)
            drive_param_list.append(drive_params)
            
            # Print intermediate results
            self._print_intermediate_results(fidelity_list, drive_param_list, jdx)
        
        print("Fidelity optimization sweep completed!")
        return fidelity_list, drive_param_list
    
    def _print_intermediate_results(self, fidelity_list, drive_param_list, current_idx):
        """
        Print intermediate optimization results.
        
        Args:
            fidelity_list (list): List of optimized fidelities
            drive_param_list (list): List of optimized drive parameters
            current_idx (int): Current optimization index
        """
        print(f"\n--- Results after {current_idx + 1} optimizations ---")
        
        # Print gate times
        print("Gate times:")
        for i in range(0, len(fidelity_list), 4):
            gate_times = np.array(self.x0_vec[:, 0])[:current_idx + 1][i:i+4]
            print(', '.join(map(str, np.round(gate_times, 8))), ',')
        
        # Print log gate errors
        print(f"Log gate errors (trunc={self.truc_optimize}):")
        for i in range(0, len(fidelity_list), 4):
            print(', '.join(map(str, np.round(fidelity_list[i:i+4], 8))), ',')
        
        # Print drive parameters
        print(f"Drive parameters (trunc={self.truc_optimize}):")
        for params in drive_param_list:
            print(np.round(params, 6).tolist(), ',')
    
    def print_configuration_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("CZ FIDELITY OPTIMIZER CONFIGURATION")
        print("="*60)
        
        print(f"File: {os.path.basename(__file__)}")
        print(f"Start time: {datetime.now(pytz.timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        print(f"\nTruncation parameters:")
        print(f"  - Large truncation: {self.truc_large}")
        print(f"  - Optimization truncation: {self.truc_optimize}")
        
        print(f"\nOptimization bounds:")
        print(f"  - Amplitude bounds: {self.amp_bound}")
        print(f"  - Detuning bounds: {self.detune_bound}")
        print(f"  - Gate time bounds: {self.tg_bound}")
        
        print(f"\nDifferential evolution parameters:")
        print(f"  - Workers: {self.workers}")
        print(f"  - Population size: {self.popsize}")
        print(f"  - Recombination: {self.recombination}")
        print(f"  - Tolerance: {self.tol}")
        print(f"  - Mutation: {self.mutation}")
        
        print(f"\nSystem parameters:")
        print(f"  - Transition frequency W_20_50: {np.round(self.W_20_50, 3)}")
        print(f"  - Number of gate times to optimize: {len(self.x0_vec)}")
        print(f"  - Hilbert space size (optimization): {len(self.hspace_select)}")
        print(f"  - Hilbert space size (total): {len(self.hspace_full)}")
        
        print("="*60)


def main():
    """
    Main function to run CZ fidelity optimization.
    
    This function initializes the optimizer and runs the fidelity sweep.
    """
    print(f"Starting {os.path.basename(__file__)}")
    ut.print_time()
    
    # Create optimizer instance
    optimizer = CZFidelityOptimizer()
    
    # Print configuration summary
    optimizer.print_configuration_summary()
    
    # Run fidelity optimization sweep
    fidelity_results, drive_param_results = optimizer.run_fidelity_sweep()
    
    # Print final results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    
    print(f"\nFinal results:")
    print(f"Workers: {optimizer.workers}, Population size: {optimizer.popsize}")
    print(f"Recombination: {optimizer.recombination}, Tolerance: {optimizer.tol}")
    print(f"Mutation: {optimizer.mutation}")
    
    print(f"\nOptimized fidelities (log gate errors):")
    for i, fidelity in enumerate(fidelity_results):
        print(f"Gate time {i}: {fidelity:.8f}")
    
    print(f"\nOptimized drive parameters:")
    for i, params in enumerate(drive_param_results):
        print(f"Gate time {i}: {params}")
    
    ut.print_time()
    print("Optimization completed successfully!")


if __name__ == '__main__':
    main()

