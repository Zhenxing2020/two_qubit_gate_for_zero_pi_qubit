"""Run X-gate fidelity or population simulation with CLI parameters."""

import argparse
import os
import sys
from pathlib import Path

# os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "128"

import numpy as np
import qutip as qt
import scqubits.settings as settings
import pandas as pd
from joblib import Parallel, delayed

XGATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = XGATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(XGATE_DIR)
from datetime import datetime
import utils_2Q_gate_zp as ut

settings.OVERLAP_THRESHOLD = 0.3

# Unit convention used by this script and the imported `ut.*` helpers:
# - simulation time (`tg`, `pop_tg`, `tlist`, `max_step`) is in ns
# - coherence time (`t1`, `tphi`) is in us
# - transition / drive frequencies (`w_trans_*`, `drive_freq_*`) are angular frequencies in rad/ns
#   because `ut.load_qubit_data_xgate()` converts spectra to `2*pi*GHz`
# - user-facing detunings (`detune_*`) are in GHz and converted to rad/ns via `2*pi*detune`
# - decay / dephasing rates are therefore in 1/ns


def _parse_tg_list(text: str):
    """Parse gate-index list from '0,3,5' or '0:18'."""
    if ":" in text:
        start, stop = text.split(":", maxsplit=1)
        return list(range(int(start), int(stop)))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _load_qubit_data(qubit_0: bool):
    """Load spectrum and matrix elements with stable unpacking."""
    data = ut.load_qubit_data_xgate(qubit_0=qubit_0)
    if len(data) == 4:
        return data
    evals, n_theta, n_phi = data
    return evals, n_theta, n_phi, [0, 2]


def _build_hamiltonian_inputs(
    drive_phi: bool,
    drive_theta: bool,
    n_full: int,
    t1: float,
    charge_truc: bool,
    qubit_0: bool,
):
    # `t1` is provided in us, so `gamma_t1` is converted to a rate in 1/ns.
    gamma_t1 = 1 / 1e3 / t1
    tphi = t1
    # `ut.load_qubit_data_xgate()` returns `evals`, `n_theta`, `n_phi` in rad/ns.
    evals, n_theta, n_phi, logi_state = _load_qubit_data(qubit_0=qubit_0)
    # `ut.compute_drive_xgate()` returns transition frequencies in rad/ns.
    w_trans_1, w_trans_2, drive_term, gamma_t1_mat = ut.compute_drive_xgate(
        evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1
    )
    hspace = np.arange(n_full).tolist()
    if charge_truc:
        hspace = ut.get_truncated_subspace_xgate(drive_term, n_full)
    h_qbt_drive, drive_truc, logi_idx = ut.build_hamiltonian_xgate(evals, drive_term, hspace, logi_state)
    return {
        "w_trans_1": w_trans_1,
        "w_trans_2": w_trans_2,
        "gamma_t1_mat": gamma_t1_mat,
        "tphi": tphi,
        "hspace": hspace,
        "h_qbt_drive": h_qbt_drive,
        "drive_truc": drive_truc,
        "logi_idx": logi_idx,
    }


def run_xgate_fidelity(args):
    max_step_ideal = args.max_step_ideal
    max_step_noisy = args.max_step_noisy
    if max_step_ideal is None or max_step_noisy is None:
        # QuTiP solver step size in ns.
        default_step = 3e-4 if args.drive_theta else 1e-3
        max_step_ideal = max_step_ideal or default_step
        max_step_noisy = max_step_noisy or default_step

    # `ut.get_qutip_options()` expects `max_step_*` in ns.
    option_ideal, option_noisy = ut.get_qutip_options(max_step_ideal, max_step_noisy)
    setup = _build_hamiltonian_inputs(
        drive_phi=args.drive_phi,
        drive_theta=args.drive_theta,
        n_full=args.n_full,
        t1=args.t1,
        charge_truc=args.charge_truc,
        qubit_0=args.qubit_0,
    )
    # `ut.load_drive_params_xgate()` returns rows as:
    # [tg (ns), drive_amp_A, drive_amp_B, detune_A (GHz), detune_B (GHz)].
    params_all = ut.load_drive_params_xgate(args.drive_theta, args.drive_phi)
    params = params_all[args.tg_list, :]
    n_hspace = len(setup["hspace"])
    n_job = len(params)
    parallel_jobs = n_job if args.parallel_jobs <= 0 else args.parallel_jobs
    solver_num_cpus = args.num_cpus
    if parallel_jobs != 1 and args.num_cpus != 1:
        # Avoid nested process pools: joblib parallelizes over gate times.
        solver_num_cpus = 1

    print("drive_phi=", args.drive_phi, "; drive_theta =", args.drive_theta, "; n_full =", args.n_full, "; charge_truc =", args.charge_truc)
    print(f"T1 = Tphi = {args.t1} us, num_cpus = {solver_num_cpus}")
    print(f"calculate_ideal = {args.calculate_ideal}, calculate_noise = {args.calculate_noise}")
    print(f"apply_decay = {args.apply_decay}, apply_dephase = {args.apply_dephase}")
    print("n_hspace =", n_hspace, "; n_job =", n_job, "; parallel_jobs =", parallel_jobs)
    ut.print_fidelity(f"hspace ({args.n_full}/{n_hspace})", setup["hspace"], num_each_row=10)
    ut.print_fidelity("params", params.tolist(), num_each_row=1)

    if args.calculate_ideal:
        ideal_args = [
            setup["h_qbt_drive"],
            setup["w_trans_1"],
            setup["w_trans_2"],
            solver_num_cpus,
            [],
            setup["logi_idx"],
            option_ideal,
            option_noisy,
        ]
        f_ideal = Parallel(n_jobs=parallel_jobs)(
            delayed(ut.xgate_fidelity_log_noise)(args_indep, *ideal_args) for args_indep in params
        )
        ut.print_fidelity(f"f_ideal_{n_hspace}", f_ideal, num_digits=8)
        ut.print_time()

    if args.calculate_noise:
        state_idx_tphi, gamma_dephase_new = ut.load_dephasing_data_xgate(args.drive_theta)
        c_op_list = ut.construct_c_ops_xgate(
            n_hspace,
            setup["drive_truc"],
            setup["gamma_t1_mat"],
            gamma_dephase_new,
            setup["tphi"],
            setup["hspace"],
            state_idx_tphi,
            args.apply_decay,
            args.apply_dephase,
        )
        noisy_args = [
            setup["h_qbt_drive"],
            setup["w_trans_1"],
            setup["w_trans_2"],
            solver_num_cpus,
            c_op_list,
            setup["logi_idx"],
            option_ideal,
            option_noisy,
        ]
        f_noise = Parallel(n_jobs=parallel_jobs)(
            delayed(ut.xgate_fidelity_log_noise)(args_indep, *noisy_args) for args_indep in params
        )
        ut.print_fidelity(f"f_{int(args.t1)}us_{n_hspace}", f_noise, num_digits=8)


def run_xgate_population(args):
    setup = _build_hamiltonian_inputs(
        drive_phi=args.drive_phi,
        drive_theta=args.drive_theta,
        n_full=args.n_full,
        t1=args.t1,
        charge_truc=args.charge_truc,
        qubit_0=args.qubit_0,
    )
    n_hspace = len(setup["hspace"])
    # Population-mode solver step sizes are also in ns.
    option_ideal = qt.Options(max_step=args.max_step_ideal, nsteps=1 / args.max_step_ideal, store_states=True, num_cpus=2)
    option_noisy = qt.Options(max_step=args.max_step_noisy, nsteps=1 / args.max_step_noisy, store_states=True, num_cpus=4)
    print("drive_phi=", args.drive_phi, "; drive_theta =", args.drive_theta, "; n_full =", args.n_full, "; charge_truc =", args.charge_truc)
    print(f"T1 = Tphi = {args.t1} us, tg = {args.pop_tg:.0f}")
    print(f"calculate_ideal = {args.calculate_ideal}")
    ut.print_fidelity(f"hspace ({args.n_full}/{n_hspace})", setup["hspace"], num_each_row=10)

    tlist = np.linspace(0, args.pop_tg, num=5 * int(args.pop_tg))  # ns
    pulse_args = {
        "drive_amp_A": args.pop_drive_amp_a,  # effective drive amplitude used with `drive_term`
        "drive_freq_A": setup["w_trans_1"] + 2 * np.pi * args.pop_detune_a,  # rad/ns = rad/ns + 2*pi*GHz
        "drive_amp_B": args.pop_drive_amp_b,  # effective drive amplitude used with `drive_term`
        "drive_freq_B": setup["w_trans_2"] + 2 * np.pi * args.pop_detune_b,  # rad/ns = rad/ns + 2*pi*GHz
        "gate_time": args.pop_tg,  # ns
    }
    states = [qt.basis(n_hspace, i) for i in range(n_hspace)]
    c_op_list = None
    if not args.calculate_ideal:
        state_idx_tphi, gamma_dephase_new = ut.load_dephasing_data_xgate(args.drive_theta)
        c_op_list = ut.construct_c_ops_xgate(
            n_hspace,
            setup["drive_truc"],
            setup["gamma_t1_mat"],
            gamma_dephase_new,
            args.t1,
            setup["hspace"],
            state_idx_tphi,
            args.apply_decay,
            args.apply_dephase,
        )

    result = {}
    for jdx, state_j in enumerate(setup["logi_idx"]):
        result[jdx] = qt.mesolve(
            setup["h_qbt_drive"],
            states[state_j],
            tlist,
            c_ops=c_op_list,
            e_ops=[state * state.dag() for state in states],
            args=pulse_args,
            options=option_ideal if args.calculate_ideal else option_noisy,
        )

    print('np.shape(c_op_list)=',  np.shape(c_op_list))
    state_phi = [0, 2, 9, 10, 19, 33, 37, 47]
    state_phi_theta = [0, 2, 9]
    state_theta = [0, 2, 7, 25]
    col_phi = ["tg"] + [f"{i}" for i in state_phi] + ["other"]
    col_theta = ["tg"] + [f"{i}" for i in state_theta] + ["other"]
    col_phi_theta = ["tg"] + [f"{i}" for i in state_phi_theta] + ["other"]
    if args.drive_phi and args.drive_theta:
        columns, state_interest = col_phi_theta, state_phi_theta
    elif args.drive_phi and not args.drive_theta:
        columns, state_interest = col_phi, state_phi
    else:
        columns, state_interest = col_theta, state_theta

    ut.top_population(np.array(result[0].expect).sum(axis=1), setup["hspace"])
    ut.top_population(np.array(result[1].expect).sum(axis=1), setup["hspace"])
    state_interest_idx = [setup["hspace"].index(i) for i in state_interest]
    pop = np.array(result[0].expect)
    pop_interest = pop[state_interest_idx, :]
    pop_other = np.delete(pop, state_interest_idx, axis=0).sum(axis=0)
    pop_save = np.vstack((tlist, pop_interest, pop_other)).T

    output_dir = Path("data/population")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / (
        f"population_phi={args.drive_phi}_theta={args.drive_theta}_tg={args.pop_tg:.0f}_T1={args.t1}us_n={n_hspace}_{ts}.txt"
    )
    pd.DataFrame(pop_save, columns=columns).to_csv(output_file, sep=",", index=False, header=True)
    print(f"data saved in {output_file}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run X-gate simulations.")
    parser.add_argument("--mode", choices=["fidelity", "population"], default="fidelity")
    return parser


def main():
    parser = build_parser()
    parsed_args = parser.parse_args()
    mode = parsed_args.mode

    # Keep simulation inputs here so they are easier to read and edit.
    common_args = {
        "mode": mode,
        "drive_phi": False,
        "drive_theta": True,
        "qubit_0": True,
        # "n_full": 150, # 150 for theta, 300 for phi
        "t1": 3,  # us
        "charge_truc": True,
        "calculate_ideal": True, # must set to False if run experiment noisy xgate 
        "calculate_noise": False,
        "num_cpus": 4,
        "parallel_jobs": 1,
        "apply_decay": True,
        "apply_dephase": True,
    }

    fidelity_args = {
        "max_step_ideal": 3e-4,  # ns 3e-4 for theta, 1e-3 for phi
        "max_step_noisy": 3e-4,  # ns 3e-4 for theta, 1e-3 for phi
    }
        ################## theta and phi ##################
    if common_args["drive_theta"] and common_args["drive_phi"]:
        fidelity_args["tg_list"] = [0] 
        common_args["n_full"] = 500   # default=50
        population_args = {
            "pop_tg": 828.759495,  # ns
            "pop_drive_amp_a": 0.013563,  
            "pop_drive_amp_b": 0.034964,  
            "pop_detune_a": -0.003029,  # GHz; 
            "pop_detune_b": -0.003182,  # GHz;
        }        
        ################## theta ##################
    elif common_args["drive_theta"] and not common_args["drive_phi"]:
        fidelity_args["tg_list"] = list(range(18)) #[::4], # [:1] 
        common_args["n_full"] = 50 # default=150 for fidelity, 1000 for population
        population_args = {
            "pop_tg": 25.020473,  # ns
            "pop_drive_amp_a": 0.180208,  
            "pop_drive_amp_b": 0.046132,  
            "pop_detune_a": -0.003789,  # GHz; 
            "pop_detune_b": 0.001544,  # GHz;
        }        
        ################## phi ##################
    elif common_args["drive_phi"] and not common_args["drive_theta"]:
        fidelity_args["tg_list"] = list(range(19))  
        common_args["n_full"] = 50
        fidelity_args.update(
            {
                "max_step_ideal": 1e-3,  # ns 3e-4 for theta, 1e-3 for phi
                "max_step_noisy": 1e-3,  # ns 3e-4 for theta, 1e-3 for phi
            })   
        population_args = {
            "pop_tg": 120.004766,  # ns
            "pop_drive_amp_a": 0.209273,  
            "pop_drive_amp_b": 0.104767,  
            "pop_detune_a": 0.341188,  # GHz; 
            "pop_detune_b": 0.360503,  # GHz;
        }    

    if mode == "population":    
        common_args["n_full"] = 1000
            ################## theta and phi ##################
        if common_args["drive_theta"] and common_args["drive_phi"]:
            fidelity_args["tg_list"] = [0] 
            common_args["n_full"] = 500   # default=50
     
            ################## theta ##################
        elif common_args["drive_theta"] and not common_args["drive_phi"]:
            fidelity_args["tg_list"] = [0] # [3] #[::4], # [:1] 
   
            ################## phi ##################
        elif common_args["drive_phi"] and not common_args["drive_theta"]:
            fidelity_args["tg_list"] = [0] # [10]


    args = argparse.Namespace(**common_args, **fidelity_args, **population_args)

    if not args.drive_phi and not args.drive_theta:
        parser.error("At least one drive must be enabled: --drive-theta or --drive-phi.")

    print(os.path.basename(__file__))
    print("NUMEXPR_NUM_THREADS =", os.environ.get("NUMEXPR_NUM_THREADS"))
    print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    print("mode =", args.mode)
    print("fidelity_args =", fidelity_args)
    ut.print_time()

    if args.mode == "fidelity":
        run_xgate_fidelity(args)
    else:
        run_xgate_population(args)

    ut.print_time()


if __name__ == "__main__":
    main()






