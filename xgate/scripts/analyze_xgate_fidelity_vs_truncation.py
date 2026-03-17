import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import qutip as qt
import scqubits as scq
import scqubits.settings as settings
from tqdm import tqdm

XGATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = XGATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(XGATE_DIR)

import utils_2Q_gate_zp as ut

settings.OVERLAP_THRESHOLD = 0.3


def parse_int_list(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def run_analysis(args):
    e_cj = 2 * args.ec_phi
    e_c = 2.0 / (1.0 / args.ec_theta - 1.0 / args.ec_phi)
    phi_grid = scq.Grid1d(-6 * np.pi, 6 * np.pi, args.phi_points)
    print("drive_phi =", args.drive_phi, "; drive_theta =", args.drive_theta)
    print("tg, drive_amp_A, drive_amp_B, detune_A, detune_B =", args.tg, args.drive_amp_a, args.drive_amp_b, args.detune_a, args.detune_b)
    print("truc_vec =", args.truc_vec)

    fidelity = []
    for idx, truc in tqdm(enumerate(args.truc_vec)):
        zero_pi = scq.ZeroPi(
            grid=phi_grid,
            EJ=args.ej,
            EL=args.el,
            ECJ=e_cj,
            EC=e_c,
            dEJ=args.dej,
            ng=args.ng,
            flux=args.flux,
            ncut=args.ncut,
            truncated_dim=truc,
        )
        n_theta = 2 * np.pi * qt.Qobj(zero_pi.matrixelement_table(operator="n_theta_operator", evals_count=truc))
        n_phi = 2 * np.pi * qt.Qobj(zero_pi.matrixelement_table(operator="i_d_dphi_operator", evals_count=truc))
        evals = 2 * np.pi * zero_pi.eigenvals(evals_count=truc)
        h0 = qt.Qobj(np.diag(evals))

        if args.drive_phi and args.drive_theta:
            w_trans_1 = evals[9] - evals[0]
            w_trans_2 = evals[9] - evals[2]
            drive_term = args.mix_phi_weight * n_phi + (1.0 - args.mix_phi_weight) * n_theta
        elif args.drive_phi:
            w_trans_1 = evals[9] - evals[0]
            w_trans_2 = evals[9] - evals[2]
            drive_term = n_phi
        else:
            w_trans_1 = evals[7] - evals[0]
            w_trans_2 = evals[7] - evals[2]
            drive_term = n_theta

        optimize_arg = [args.drive_amp_a, args.drive_amp_b, args.detune_a, args.detune_b]
        optimize_ctx = [h0, drive_term, args.tg, w_trans_1, w_trans_2, np.arange(truc)]
        fidelity.append(ut.xgate_fidelity_optimize(optimize_arg, optimize_ctx))

        print("\ntruc =", np.array(args.truc_vec[: idx + 1]).tolist())
        print("\nlog of gate error =")
        for i in range(0, len(fidelity), 4):
            print(", ".join(map(str, np.round(fidelity[i : i + 4], 8))), ",")
        fidelity_real = np.array(fidelity)
        print("\nfidelity =")
        for i in range(0, len(fidelity), 4):
            print(", ".join(map(str, np.round((1 - 10**fidelity_real)[i : i + 4], 8))), ",")


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze X-gate fidelity versus truncation.")
    parser.add_argument("--drive-phi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drive-theta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tg", type=float, default=20.0)
    parser.add_argument("--drive-amp-a", type=float, default=0.247986)
    parser.add_argument("--drive-amp-b", type=float, default=0.058597)
    parser.add_argument("--detune-a", type=float, default=-0.019272)
    parser.add_argument("--detune-b", type=float, default=-0.016288)
    parser.add_argument("--truc-vec", type=parse_int_list, default=[80], help="Comma-separated truncations, e.g. 40,60,80")
    parser.add_argument("--mix-phi-weight", type=float, default=0.976, help="Phi weight if both drives are enabled.")
    parser.add_argument("--el", type=float, default=0.377)
    parser.add_argument("--ej", type=float, default=6.013)
    parser.add_argument("--ec-phi", type=float, default=1.142)
    parser.add_argument("--ec-theta", type=float, default=0.092)
    parser.add_argument("--dej", type=float, default=0.0)
    parser.add_argument("--ng", type=float, default=0.0)
    parser.add_argument("--flux", type=float, default=0.0)
    parser.add_argument("--ncut", type=int, default=30)
    parser.add_argument("--phi-points", type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.drive_phi and not args.drive_theta:
        raise ValueError("At least one of --drive-theta or --drive-phi must be enabled.")
    print(os.path.basename(__file__))
    print("Current Mountain Time:", datetime.now(pytz.timezone("America/Denver")))
    run_analysis(args)
    print("Current Mountain Time:", datetime.now(pytz.timezone("America/Denver")))


if __name__ == "__main__":
    main()
