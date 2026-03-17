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

import utils_2Q_gate_zp as ut

settings.OVERLAP_THRESHOLD = 0.3


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
    gamma_t1 = 1 / 1e3 / t1
    tphi = t1
    evals, n_theta, n_phi, logi_state = _load_qubit_data(qubit_0=qubit_0)
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
        default_step = 3e-4 if args.drive_theta else 1e-3
        max_step_ideal = max_step_ideal or default_step
        max_step_noisy = max_step_noisy or default_step

    option_ideal, option_noisy = ut.get_qutip_options(max_step_ideal, max_step_noisy)
    setup = _build_hamiltonian_inputs(
        drive_phi=args.drive_phi,
        drive_theta=args.drive_theta,
        n_full=args.n_full,
        t1=args.t1,
        charge_truc=args.charge_truc,
        qubit_0=args.qubit_0,
    )
    params_all = ut.load_drive_params_xgate(args.drive_theta)
    params = params_all[args.tg_list, :]
    n_hspace = len(setup["hspace"])
    n_job = len(params)
    parallel_jobs = n_job if args.parallel_jobs <= 0 else args.parallel_jobs

    print("drive_phi=", args.drive_phi, "; drive_theta =", args.drive_theta, "; n_full =", args.n_full, "; charge_truc =", args.charge_truc)
    print(f"T1 = Tphi = {args.t1} us, num_cpus = {args.num_cpus}")
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
            args.num_cpus,
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
            args.num_cpus,
            c_op_list,
            setup["logi_idx"],
            option_ideal,
            option_noisy,
        ]
        f_noise = Parallel(n_jobs=parallel_jobs)(
            delayed(ut.xgate_fidelity_log_noise)(args_indep, *noisy_args) for args_indep in params
        )
        ut.print_fidelity(f"f_{args.t1}us_{n_hspace}", f_noise, num_digits=8)


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
    option_ideal = qt.Options(max_step=args.pop_max_step_ideal, nsteps=1 / args.pop_max_step_ideal, store_states=True, num_cpus=1)
    option_noisy = qt.Options(max_step=args.pop_max_step_noisy, nsteps=1 / args.pop_max_step_noisy, store_states=True, num_cpus=1)
    print("drive_phi=", args.drive_phi, "; drive_theta =", args.drive_theta, "; n_full =", args.n_full, "; charge_truc =", args.charge_truc)
    print(f"T1 = Tphi = {args.t1} us, tg = {args.pop_tg:.0f}")
    print(f"calculate_ideal = {args.calculate_ideal}")
    ut.print_fidelity(f"hspace ({args.n_full}/{n_hspace})", setup["hspace"], num_each_row=10)

    tlist = np.linspace(0, args.pop_tg, num=5 * int(args.pop_tg))
    pulse_args = {
        "drive_amp_A": args.pop_drive_amp_a,
        "drive_freq_A": setup["w_trans_1"] + 2 * np.pi * args.pop_detune_a,
        "drive_amp_B": args.pop_drive_amp_b,
        "drive_freq_B": setup["w_trans_2"] + 2 * np.pi * args.pop_detune_b,
        "gate_time": args.pop_tg,
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
    output_file = output_dir / (
        f"population_phi={args.drive_phi}_theta={args.drive_theta}_tg={args.pop_tg:.0f}_{args.t1}us_n={n_hspace}.txt"
    )
    pd.DataFrame(pop_save, columns=columns).to_csv(output_file, sep=",", index=False, header=True)
    print(f"data saved in {output_file}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run X-gate simulations with explicit CLI parameters.")
    parser.add_argument("--mode", choices=["fidelity", "population"], default="fidelity")
    parser.add_argument("--drive-phi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drive-theta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qubit-0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-full", type=int, default=150)
    parser.add_argument("--t1", type=float, default=3.0, help="T1 in us; Tphi follows T1.")
    parser.add_argument("--charge-truc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--calculate-ideal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calculate-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--parallel-jobs", type=int, default=0, help="<=0 means number of selected tg rows.")
    parser.add_argument("--apply-decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-dephase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-step-ideal", type=float, default=None)
    parser.add_argument("--max-step-noisy", type=float, default=None)
    parser.add_argument("--tg-list", type=_parse_tg_list, default=list(range(18)), help="Gate row indices: '0,3,5' or '0:18'.")

    parser.add_argument("--pop-tg", type=float, default=828.759495)
    parser.add_argument("--pop-drive-amp-a", type=float, default=0.013563)
    parser.add_argument("--pop-drive-amp-b", type=float, default=0.034964)
    parser.add_argument("--pop-detune-a", type=float, default=-0.003029)
    parser.add_argument("--pop-detune-b", type=float, default=-0.003182)
    parser.add_argument("--pop-max-step-ideal", type=float, default=1e-4)
    parser.add_argument("--pop-max-step-noisy", type=float, default=3e-4)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.drive_phi and not args.drive_theta:
        parser.error("At least one drive must be enabled: --drive-theta or --drive-phi.")

    print(os.path.basename(__file__))
    print("NUMEXPR_NUM_THREADS =", os.environ.get("NUMEXPR_NUM_THREADS"))
    print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    ut.print_time()

    if args.mode == "fidelity":
        run_xgate_fidelity(args)
    else:
        run_xgate_population(args)

    ut.print_time()


if __name__ == "__main__":
    main()






