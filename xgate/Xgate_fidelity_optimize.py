import sys
import os
import argparse
from pathlib import Path
XGATE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = XGATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(XGATE_DIR)
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
    args = [H_truc, w_trans_1, w_trans_2, num_cpus, [], logi_idx,
            option_ideal, option_noisy, use_qt_fidelity]
    for jdx, initial in tqdm(enumerate(initial_params), total=len(initial_params)):
        tg = initial[0]
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)

        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_log_noise,
            bounds=bounds,
            args=args,
            disp=True,
            callback=ut.print_soln,
            init="sobol",
            x0=initial,
            workers=workers,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=False
        )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        # Re-evaluate the optimized pulse with the selected fidelity mode.
        fidelity_full.append(ut.xgate_fidelity_log_noise(res.x, *args))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", choices=("phi", "theta"), required=True)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="resume at this zero-based row of the pulse table",
    )
    parser.add_argument(
        "--use-qt-fidelity",
        type=lambda x: x.lower() in {"true", "1", "yes"},
        default=True,
    )
    cli = parser.parse_args()
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    drive_phi, drive_theta = cli.drive == "phi", cli.drive == "theta"
    use_qt_fidelity = cli.use_qt_fidelity

    pulse_file = (
        "../figure/data/data_xgate_phi_mstep_1e3_npz.txt"
        if drive_phi
        else "../figure/data/data_xgate_theta_mstep_3e4_npz.txt"
    )
    fidelity_column = "f_charge_160_npz" if drive_phi else "f_157_charge"
    pulse_table = pd.read_csv(pulse_file)
    initial_params = pulse_table[
        ["tg", "drive_amp_1", "drive_amp_2", "detune_1", "detune_2"]
    ].to_numpy()[cli.start_index:]
    initial_fidelity = pulse_table[fidelity_column].to_numpy()[cli.start_index:]

    # Parameter bounds
    if drive_theta:
        amp1_bounds, amp2_bounds = (0, 0.45), (0, 0.1)
        detune1_bounds, detune2_bounds = (-0.1, 0.05), (-0.05, 0.05)
        tg_vec = initial_params[:, 0].tolist()
    else:
        amp1_bounds, amp2_bounds = (0.15, 0.3), (0, 0.3)
        detune1_bounds, detune2_bounds = (0.3, 0.5), (0.3, 0.5)
        tg_vec = initial_params[:, 0].tolist()

    tg_bound = (-0.01, 0.01)

    # Differential evolution hyperparameters
    num_cpus = 1 # Lower num_cpus <4 can reduce num of workers while >4 won’t change the num.
    workers, popsize = 100, 10
    recombination, tol, mutation = 0.7, 0.01, (0.5, 1.0)
    n_truc, n_full = 300, 300

    # Theta-drive dynamics require the same 3e-4 ns solver step used by the
    # fidelity runner; 1e-3 can produce a numerically inconsistent objective.
    max_step_ideal = 3e-4 if drive_theta else 1e-3
    nsteps_ideal = 10/ max_step_ideal  # Set nsteps to a large number for parallel execution
    option_ideal = qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, store_states=True, num_cpus=1)  
    option_noisy = None
    evals, n_theta, n_phi, logi_state = ut.load_qubit_data_xgate() # Load spectrum and matrix elements
    w_trans_1, w_trans_2, drive_term = ut.compute_drive_xgate(
        evals, n_theta, n_phi, drive_phi, drive_theta
    )

    # Construct Hamiltonians
    hspace_truc = ut.get_truncated_subspace_xgate(drive_term, n_truc)
    H_truc, drive_truc, logi_idx = ut.build_hamiltonian_xgate(evals, drive_term, hspace_truc, logi_state)

    hspace_full = np.arange(n_full).tolist()
    H_full, drive_full, logi_idx_full = ut.build_hamiltonian_xgate(evals, drive_term, hspace_full, logi_state)

    # Print configuration
    print("drive_phi=", drive_phi, ", drive_theta=", drive_theta)
    print("use_qt_fidelity=", use_qt_fidelity)
    print("pulse_file=", pulse_file)
    print("initial_fidelity_column=", fidelity_column)
    print("start_index=", cli.start_index)
    ut.print_fidelity("initial_fidelity", initial_fidelity)
    print("tg_vec:", tg_vec)
    print("amp1_bounds=", amp1_bounds, ", amp2_bounds=", amp2_bounds)
    print("detune1_bounds=", detune1_bounds, ", detune2_bounds=", detune2_bounds)
    print("n_truc=", n_truc, ", n_full=", n_full, ", n_optimize)=", len(hspace_truc))
    print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')

    # Run optimization
    fidelity_de()
