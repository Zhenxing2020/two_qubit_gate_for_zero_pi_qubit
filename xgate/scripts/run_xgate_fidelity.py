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
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

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


def _parse_bool(text: str):
    """Parse a command-line boolean."""
    value = text.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


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
    params_all = getattr(args, "params_override", None)
    if params_all is None:
        params_all = ut.load_drive_params_xgate(args.drive_theta, args.drive_phi)
    params = params_all[args.tg_list, :]
    n_hspace = len(setup["hspace"])
    n_job = len(params)
    parallel_jobs = n_job if args.parallel_jobs <= 0 else args.parallel_jobs
    solver_num_cpus = args.num_cpus
    # if parallel_jobs != 1 and args.num_cpus != 1:
        # Avoid nested process pools: joblib parallelizes over gate times.
        # solver_num_cpus = 1

    print("drive_phi=", args.drive_phi, "; drive_theta =", args.drive_theta, "; n_full =", args.n_full, "; charge_truc =", args.charge_truc)
    print(f"T1 = Tphi = {args.t1} us, num_cpus = {solver_num_cpus}")
    print(f"calculate_ideal = {args.calculate_ideal}, calculate_noise = {args.calculate_noise}")
    print(f"apply_decay = {args.apply_decay}, apply_dephase = {args.apply_dephase}")
    print(f"use_qt_fidelity = {args.use_qt_fidelity}")
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
            args.use_qt_fidelity,
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
            args.use_qt_fidelity,
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
    # nsteps is the allowed number of internal substeps between saved tlist
    # points, not the total trajectory step count.  Long Phi pulses can need
    # substantially more than 1/max_step before reaching the next output time.
    option_ideal = qt.Options(max_step=args.max_step_ideal, nsteps=10_000_000, store_states=False, num_cpus=2)
    option_noisy = qt.Options(max_step=args.max_step_noisy, nsteps=10_000_000, store_states=False, num_cpus=4)
    print("drive_phi=", args.drive_phi, "; drive_theta =", args.drive_theta, "; n_full =", args.n_full, "; charge_truc =", args.charge_truc)
    print(f"T1 = Tphi = {args.t1} us, tg = {args.pop_tg:.0f}")
    print(f"calculate_ideal = {args.calculate_ideal}")
    ut.print_fidelity(f"hspace ({args.n_full}/{n_hspace})", setup["hspace"], num_each_row=10)

    # Use the same number of saved trajectory points for every requested gate.
    tlist = np.linspace(0, args.pop_tg, num=500)  # ns
    pulse_args = {
        "drive_amp_A": args.pop_drive_amp_a,  # effective drive amplitude used with `drive_term`
        "drive_freq_A": setup["w_trans_1"] + 2 * np.pi * args.pop_detune_a,  # rad/ns = rad/ns + 2*pi*GHz
        "drive_amp_B": args.pop_drive_amp_b,  # effective drive amplitude used with `drive_term`
        "drive_freq_B": setup["w_trans_2"] + 2 * np.pi * args.pop_detune_b,  # rad/ns = rad/ns + 2*pi*GHz
        "gate_time": args.pop_tg,  # ns
    }
    state_phi = [0, 2, 9, 10, 19, 33, 37, 47]
    state_phi_theta = [0, 2, 9]
    state_theta = [0, 2, 7, 25, 4, 1, 5]
    if args.drive_phi and args.drive_theta:
        state_interest = state_phi_theta
    elif args.drive_phi and not args.drive_theta:
        state_interest = state_phi
    else:
        state_interest = state_theta
    state_interest_idx = [setup["hspace"].index(i) for i in state_interest]
    e_ops = [qt.basis(n_hspace, i).proj() for i in state_interest_idx]
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
            qt.basis(n_hspace, state_j),
            tlist,
            c_ops=c_op_list,
            e_ops=e_ops,
            args=pulse_args,
            options=option_ideal if args.calculate_ideal else option_noisy,
        )

    print('np.shape(c_op_list)=',  np.shape(c_op_list))
    col_phi = ["tg"] + [f"{i}" for i in state_phi] + ["other"]
    col_theta = ["tg"] + [f"{i}" for i in state_theta] + ["other"]
    col_phi_theta = ["tg"] + [f"{i}" for i in state_phi_theta] + ["other"]
    if args.drive_phi and args.drive_theta:
        columns = col_phi_theta
    elif args.drive_phi and not args.drive_theta:
        columns = col_phi
    else:
        columns = col_theta

    pop = np.array(result[0].expect)
    pop_interest = pop
    pop_other = 1.0 - pop_interest.sum(axis=0)
    pop_save = np.vstack((tlist, pop_interest, pop_other)).T

    output_dir = Path("data/population")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"population_phi={args.drive_phi}_theta={args.drive_theta}_tg={args.pop_tg:.0f}_T1={args.t1}us_n={n_hspace}.txt"
    )
    pd.DataFrame(pop_save, columns=columns).to_csv(output_file, sep=",", index=False, header=True)
    print(f"data saved in {output_file}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row, label in zip(pop_interest, columns[1:-1]):
        ax.plot(tlist, row, label=label)
    ax.plot(tlist, pop_other, "--", label="other")
    ax.set(xlabel="Time (ns)", ylabel="Population", ylim=(-0.02, 1.02))
    ax.set_title(
        f"X-gate population, n={n_hspace}, $t_g$={args.pop_tg:.3f} ns"
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    plot_file = output_file.with_suffix(".png")
    fig.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved in {plot_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-qt-fidelity",
        type=_parse_bool,
        default=None,
        metavar="{true,false}",
        help="use QuTiP fidelity (true) or trace-decreasing fidelity (false)",
    )
    parser.add_argument("--drive-phi", type=_parse_bool, default=None)
    parser.add_argument("--drive-theta", type=_parse_bool, default=None)
    parser.add_argument("--calculate-noise", type=_parse_bool, default=None)
    parser.add_argument("--calculate-ideal", type=_parse_bool, default=None)
    parser.add_argument("--n-full", type=int, default=None)
    parser.add_argument("--mode", choices=("fidelity", "population"), default=None)
    parser.add_argument("--charge-truc", type=_parse_bool, default=None)
    parser.add_argument("--tg-list", type=_parse_tg_list, default=None)
    parser.add_argument("--t1", type=float, default=None, help="T1=Tphi in us")
    parser.add_argument(
        "--pulse-file", default=None,
        help="CSV containing tg, drive_amp_1, drive_amp_2, detune_1, detune_2",
    )
    parser.add_argument(
        "--parallel-jobs", type=int, default=None,
        help="number of outer Joblib jobs",
    )
    cli_args = parser.parse_args()

    mode = cli_args.mode or "fidelity"

    # Keep simulation inputs here so they are easier to read and edit.
    common_args = {
        "mode": mode,
        "drive_phi": False,
        "drive_theta": True,
        "qubit_0": True,
        # "n_full": 150, # 150 for theta, 300 for phi
        "t1": 170,  # us
        "charge_truc": True,
        "calculate_ideal": True, # must set to False if run experiment noisy xgate 
        "calculate_noise": False,
        "num_cpus": 4,
        "parallel_jobs": 10,
        "apply_decay": True,
        "apply_dephase": True,
        # May be overridden by --use-qt-fidelity true/false.
        "use_qt_fidelity": False,
    }
    if cli_args.use_qt_fidelity is not None:
        common_args["use_qt_fidelity"] = cli_args.use_qt_fidelity
    if cli_args.drive_phi is not None:
        common_args["drive_phi"] = cli_args.drive_phi
    if cli_args.drive_theta is not None:
        common_args["drive_theta"] = cli_args.drive_theta
    if cli_args.calculate_noise is not None:
        common_args["calculate_noise"] = cli_args.calculate_noise
    if cli_args.calculate_ideal is not None:
        common_args["calculate_ideal"] = cli_args.calculate_ideal
    if cli_args.t1 is not None:
        common_args["t1"] = cli_args.t1
    if cli_args.parallel_jobs is not None:
        common_args["parallel_jobs"] = cli_args.parallel_jobs
    if cli_args.charge_truc is not None:
        common_args["charge_truc"] = cli_args.charge_truc

    fidelity_args = {
        "max_step_ideal": 3e-4,  # ns 3e-4 for theta, 1e-3 for phi
        "max_step_noisy": 3e-4,  # ns 3e-4 for theta, 1e-3 for phi
    }
        ################## theta and phi ##################
    if common_args["drive_theta"] and common_args["drive_phi"]:
        fidelity_args["tg_list"] = [0] 
        common_args["n_full"] = 50   # default=50
        population_args = {
            "pop_tg": 828.759495,  # ns
            "pop_drive_amp_a": 0.013563,  
            "pop_drive_amp_b": 0.034964,  
            "pop_detune_a": -0.003029,  # GHz; 
            "pop_detune_b": -0.003182,  # GHz;
        }        
        ################## theta ##################
    elif common_args["drive_theta"] and not common_args["drive_phi"]:
        fidelity_args["tg_list"] =  list(range(18)) #[-10,-1] # [::4], # [:1] 
        common_args["n_full"] = 300 # default=150 for fidelity, 1000 for population
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

    # Apply size/index overrides after drive-specific defaults.
    if cli_args.n_full is not None:
        common_args["n_full"] = cli_args.n_full
    if cli_args.tg_list is not None:
        fidelity_args["tg_list"] = cli_args.tg_list

    if cli_args.pulse_file is not None:
        pulse_table = pd.read_csv(cli_args.pulse_file)
        params_override = pulse_table[
            ["tg", "drive_amp_1", "drive_amp_2", "detune_1", "detune_2"]
        ].to_numpy()
        # The loader below reads this Namespace field when supplied.
        common_args["params_override"] = params_override
        if cli_args.tg_list is None:
            fidelity_args["tg_list"] = list(range(len(params_override)))

    if mode == "population":    
        common_args["n_full"] = 30 # 1000
            ################## theta and phi ##################
        if common_args["drive_theta"] and common_args["drive_phi"]:
            fidelity_args["tg_list"] = [0] 
            common_args["n_full"] = 500   # default=50
    
            ################## theta ##################
        elif common_args["drive_theta"] and not common_args["drive_phi"]:
            if cli_args.tg_list is None:
                fidelity_args["tg_list"] = [0]
     
            ################## phi ##################
        elif common_args["drive_phi"] and not common_args["drive_theta"]:
            if cli_args.tg_list is None:
                fidelity_args["tg_list"] = [10]

        # In population mode a pulse file supplies the optimized row selected
        # by --tg-list (one row is expected for an individual trajectory).
        if cli_args.pulse_file is not None:
            selected = params_override[fidelity_args["tg_list"]]
            if len(selected) != 1:
                raise ValueError("population mode requires exactly one --tg-list index")
            row = selected[0]
            population_args.update({
                "pop_tg": float(row[0]),
                "pop_drive_amp_a": float(row[1]),
                "pop_drive_amp_b": float(row[2]),
                "pop_detune_a": float(row[3]),
                "pop_detune_b": float(row[4]),
            })
        # Apply this last: the historical population default above must not
        # overwrite an explicit truncation requested on the command line.
        if cli_args.n_full is not None:
            common_args["n_full"] = cli_args.n_full


    args = argparse.Namespace(**common_args, **fidelity_args, **population_args)

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
