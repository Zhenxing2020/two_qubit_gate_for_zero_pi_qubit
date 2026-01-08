import sys
sys.path.append('../')
import os
import pytz
from datetime import datetime
import numpy as np
import pandas as pd
import scipy as sp
import qutip as qt
import scqubits as scq
import scqubits.settings as settings
from tqdm import tqdm
import utils_2Q_gate_zp as ut
# Set threshold for matrix element overlap
settings.OVERLAP_THRESHOLD = 0.3



def fidelity_de():
    """
    Optimize gate fidelity using differential evolution.

    Returns:
        None
    """
    fidelity = []
    drive_param = []
    fidelity_full = []
    args = [H_truc, w_trans_1, w_trans_2, num_cpus, [], logi_idx, option_ideal, option_noisy]
    for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)

        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_log_noise,
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
            polish=False
        )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        # Evaluate full system fidelity
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = res.x
        argz = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, 
                H_full, w_trans_1, w_trans_2, num_cpus, 
                [], logi_idx_full, option_ideal, option_noisy]
        fidelity_full.append(ut.xgate_fidelity_log(argz))

        # Print progress
        print(f"\nOptimal result for tg={tg}:")
        print(res)
        print(f"Gate errors (log10) (truc1={len(hspace_truc)}):")
        print(', '.join(map(str, np.round(fidelity[-4:], 8))))
        print(f"Drive parameters:")
        print(np.round(res.x, 6).tolist())
        print(f"Full system error (truc_full={n_full}):")
        print(', '.join(map(str, np.round(fidelity_full[-4:], 8))))
        ut.print_fidelity(f'f_optimize', fidelity)
        ut.print_fidelity(f'f_optimize', fidelity_full)
        ut.print_time()

if __name__ == '__main__':
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    drive_phi, drive_theta  = False, True

    # Parameter bounds
    if drive_theta:
        amp1_bounds, amp2_bounds = (0.01, 0.05), (0.025, 0.2)
        detune1_bounds, detune2_bounds = (-0.17, -0.2), (0.1, 0.3)
        tg_vec = [2,3,4] # np.arange(40, 50, step=1).tolist()
    else:
        amp1_bounds, amp2_bounds = (0.15, 0.3), (0, 0.3)
        detune1_bounds, detune2_bounds = (0.3, 0.5), (0.3, 0.5)
        tg_vec = np.arange(10, 50, step=10).tolist()

    tg_bound = (-0.01, 0.01)

    # Differential evolution hyperparameters
    num_cpus = 1 # Lower num_cpus <4 can reduce num of workers while >4 won’t change the num.
    workers, popsize = 100, 10
    recombination, tol, mutation = 0.7, 0.01, (0.5, 1.0)
    n_truc, n_full = 150, 300

    max_step_ideal = 1e-3 # Set max_step to 0 for parallel execution
    nsteps_ideal = 10/ max_step_ideal  # Set nsteps to a large number for parallel execution
    option_ideal = qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, store_states=True, num_cpus=1)  
    option_noisy = None
    evals, n_theta, n_phi, logi_state = ut.load_qubit_data_xgate() # Load spectrum and matrix elements
    w_trans_1, w_trans_2, drive_term = ut.compute_drive_xgate(evals, n_theta, n_phi, drive_phi, drive_theta)  

    # Construct Hamiltonians
    hspace_truc = ut.get_truncated_subspace_xgate(drive_term, n_truc)
    H_truc, drive_truc, logi_idx = ut.build_hamiltonian_xgate(evals, drive_term, hspace_truc, logi_state)

    hspace_full = np.arange(n_full).tolist()
    H_full, drive_full, logi_idx_full = ut.build_hamiltonian_xgate(evals, drive_term, hspace_full, logi_state)

    # Print configuration
    print("drive_phi=", drive_phi, ", drive_theta=", drive_theta)
    print("tg_vec:", tg_vec)
    print("amp1_bounds=", amp1_bounds, ", amp2_bounds=", amp2_bounds)
    print("detune1_bounds=", detune1_bounds, ", detune2_bounds=", detune2_bounds)
    print("n_truc=", n_truc, ", n_full=", n_full, ", n_optimize)=", len(hspace_truc))
    print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')

    # Run optimization
    fidelity_de()
