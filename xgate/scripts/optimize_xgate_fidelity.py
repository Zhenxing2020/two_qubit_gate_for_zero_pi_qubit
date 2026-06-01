import argparse
import os
import sys
from pathlib import Path

import numpy as np
import qutip as qt
import scipy as sp
import scqubits.settings as settings
from tqdm import tqdm

XGATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = XGATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(XGATE_DIR)

import utils_2Q_gate_zp as ut

settings.OVERLAP_THRESHOLD = 0.3


def parse_float_list(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def run_differential_evolution(args, h_truc, h_full, w_trans_1, w_trans_2, logi_idx, logi_idx_full, option_ideal, option_noisy, hspace_truc):
    fidelity = []
    fidelity_full = []
    objective_args = [h_truc, w_trans_1, w_trans_2, args.num_cpus, [], logi_idx, option_ideal, option_noisy]
    for tg in tqdm(args.tg_vec):
        tg_bounds = (tg + args.tg_bound_min, tg + args.tg_bound_max)
        bounds = (tg_bounds, tuple(args.amp1_bounds), tuple(args.amp2_bounds), tuple(args.detune1_bounds), tuple(args.detune2_bounds))
        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_log_noise,
            bounds=bounds,
            args=objective_args,
            disp=True,
            callback=ut.print_soln,
            init="sobol",
            workers=args.workers,
            popsize=args.popsize,
            mutation=tuple(args.mutation),
            recombination=args.recombination,
            tol=args.tol,
            polish=False,
        )
        fidelity.append(res.fun)
        tg_opt, drive_amp_a, drive_amp_b, detune_a, detune_b = res.x
        full_args = [
            tg_opt, drive_amp_a, drive_amp_b, detune_a, detune_b,
            h_full, w_trans_1, w_trans_2, args.num_cpus, [], logi_idx_full, option_ideal, option_noisy,
        ]
        fidelity_full.append(ut.xgate_fidelity_log(full_args))

        print(f"\nOptimal result for tg={tg_opt}:")
        print(res)
        print(f"Gate errors (log10) (truc1={len(hspace_truc)}):")
        print(", ".join(map(str, np.round(fidelity[-4:], 8))))
        print("Drive parameters:")
        print(np.round(res.x, 6).tolist())
        print(f"Full system error (truc_full={args.n_full}):")
        print(", ".join(map(str, np.round(fidelity_full[-4:], 8))))
        ut.print_fidelity("f_optimize_truc", fidelity)
        ut.print_fidelity("f_optimize_full", fidelity_full)
        ut.print_time()


def build_parser():
    parser = argparse.ArgumentParser(description="Optimize X-gate fidelity with differential evolution.")
    parser.add_argument("--drive-phi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drive-theta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qubit-0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tg-vec", type=parse_float_list, default=[2.0, 3.0, 4.0], help="Comma-separated tg seeds.")
    parser.add_argument("--amp1-bounds", type=parse_float_list, default=[0.01, 0.05])
    parser.add_argument("--amp2-bounds", type=parse_float_list, default=[0.025, 0.2])
    parser.add_argument("--detune1-bounds", type=parse_float_list, default=[-0.17, -0.2])
    parser.add_argument("--detune2-bounds", type=parse_float_list, default=[0.1, 0.3])
    parser.add_argument("--tg-bound-min", type=float, default=-0.01)
    parser.add_argument("--tg-bound-max", type=float, default=0.01)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--recombination", type=float, default=0.7)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--mutation", type=parse_float_list, default=[0.5, 1.0], help="Two values: min,max.")
    parser.add_argument("--n-truc", type=int, default=150)
    parser.add_argument("--n-full", type=int, default=300)
    parser.add_argument("--max-step-ideal", type=float, default=1e-3)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.drive_phi and not args.drive_theta:
        raise ValueError("At least one of --drive-theta or --drive-phi must be enabled.")

    print(os.path.basename(__file__))
    print("NUMEXPR_NUM_THREADS =", os.environ.get("NUMEXPR_NUM_THREADS"))
    print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    ut.print_time()

    option_ideal = qt.Options(max_step=args.max_step_ideal, nsteps=10 / args.max_step_ideal, store_states=True, num_cpus=1)
    option_noisy = None
    evals, n_theta, n_phi, logi_state = ut.load_qubit_data_xgate(qubit_0=args.qubit_0)
    w_trans_1, w_trans_2, drive_term, _ = ut.compute_drive_xgate(evals, n_theta, n_phi, args.drive_phi, args.drive_theta, gamma_t1=0)

    hspace_truc = ut.get_truncated_subspace_xgate(drive_term, args.n_truc)
    h_truc, _, logi_idx = ut.build_hamiltonian_xgate(evals, drive_term, hspace_truc, logi_state)
    hspace_full = np.arange(args.n_full).tolist()
    h_full, _, logi_idx_full = ut.build_hamiltonian_xgate(evals, drive_term, hspace_full, logi_state)

    print("drive_phi=", args.drive_phi, ", drive_theta=", args.drive_theta)
    print("tg_vec:", args.tg_vec)
    print("amp1_bounds=", args.amp1_bounds, ", amp2_bounds=", args.amp2_bounds)
    print("detune1_bounds=", args.detune1_bounds, ", detune2_bounds=", args.detune2_bounds)
    print("n_truc=", args.n_truc, ", n_full=", args.n_full, ", n_optimize)=", len(hspace_truc))
    print(f"Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}")

    run_differential_evolution(
        args=args,
        h_truc=h_truc,
        h_full=h_full,
        w_trans_1=w_trans_1,
        w_trans_2=w_trans_2,
        logi_idx=logi_idx,
        logi_idx_full=logi_idx_full,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        hspace_truc=hspace_truc,
    )


if __name__ == "__main__":
    main()
