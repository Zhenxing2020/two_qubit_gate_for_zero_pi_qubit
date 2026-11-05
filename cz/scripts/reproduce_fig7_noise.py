#!/usr/bin/env python3
"""Recalculate the paper's CZ reference pulses with its 60-state noise model."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qutip as qt
from qutip.qip.operations import cz_gate

from run_2q_fidelity import build_collapse_ops, load_system_data, select_gate_quantities
import utils_2Q_gate_zp as ut


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/data/Two_qubit_data_Sorted_Truc")
DEFAULT_NOISE = ROOT / "data/flux_derivative_truc500/gamma_phi_2Q_truc500.npz"
DEFAULT_PULSES = ROOT / "figure/data/data_cz_fidelity_leakage.txt"
DEFAULT_OUTPUT = ROOT / "cz/data/fig7_30us_reproduction.csv"
PAPER_TABLE_F_OPEN = {91.95: 0.99830, 182.01: 0.99908}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--noise", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--pulses", type=Path, default=DEFAULT_PULSES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-kraus", type=Path,
                        help="save the evolved map and raw/projected Kraus operators to a compressed NPZ")
    parser.add_argument("--time-us", type=float, default=30.0,
                        help="base time in microseconds; Tphi stays at this value")
    parser.add_argument("--t1-rate-scale", type=float, default=1.0,
                        help="multiply decay rates by this factor, so T1 = time / factor")
    parser.add_argument("--nonlogical-rate-scale", type=float, default=1.0,
                        help="multiply decay and dephasing rates of states outside {0,2}")
    parser.add_argument("--state2-t1-ms", type=float,
                        help="set the 2->1 lifetime to this value separately for each qubit")
    parser.add_argument("--t1-82-us", type=float,
                        help="set the 8->2 lifetime to this value separately for each qubit")
    parser.add_argument("--t1-us-q0", type=float,
                        help="set qubit 0 reference T1 directly, in microseconds")
    parser.add_argument("--t1-us-q1", type=float,
                        help="set qubit 1 reference T1 directly, in microseconds")
    parser.add_argument("--include-weak-state2-decay", action="store_true",
                        help="include 2->1 and 8->2 decay jumps excluded by the paper's matrix-element cutoff")
    parser.add_argument("--states", type=int, default=60)
    parser.add_argument("--max-step-ns", type=float, default=0.05)
    parser.add_argument("--solver-tolerance", type=float, default=1e-10,
                        help="absolute and relative ODE tolerances")
    parser.add_argument("--solver-nsteps", type=int, default=1000000,
                        help="maximum internal ODE steps between output times")
    parser.add_argument("--collapse-fro-cutoff", type=float, default=1e-7)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel density-matrix evolutions (1 to 16)")
    parser.add_argument("--diagnostic-phase-scores", action="store_true",
                        help="also score a common channel-wide Z correction and no Z correction")
    parser.add_argument("--gate-times-ns", type=float, nargs="+", default=[91.95])
    args = parser.parse_args()
    if args.output is not None and not args.output.is_absolute():
        args.output = ROOT / args.output
    if args.save_kraus is not None and not args.save_kraus.is_absolute():
        args.save_kraus = ROOT / args.save_kraus
    if (args.time_us <= 0 or args.t1_rate_scale <= 0 or args.max_step_ns <= 0
            or args.nonlogical_rate_scale <= 0 or args.solver_tolerance <= 0
            or args.solver_nsteps <= 0 or args.states < 4):
        parser.error("time, T1 rate scale, max step, and state count must be positive")
    if args.state2_t1_ms is not None and args.state2_t1_ms <= 0:
        parser.error("state-2 T1 must be positive")
    if args.t1_82_us is not None and args.t1_82_us <= 0:
        parser.error("8->2 T1 must be positive")
    if (args.t1_us_q0 is None) != (args.t1_us_q1 is None):
        parser.error("--t1-us-q0 and --t1-us-q1 must be supplied together")
    if args.t1_us_q0 is not None and (args.t1_us_q0 <= 0 or args.t1_us_q1 <= 0):
        parser.error("per-qubit T1 values must be positive")
    if (args.state2_t1_ms is not None or args.t1_82_us is not None) and args.t1_rate_scale != 1:
        parser.error("use a pinned transition lifetime or --t1-rate-scale")
    if args.state2_t1_ms is not None and args.t1_82_us is not None:
        parser.error("pin either 2->1 or 8->2, not both")
    if args.t1_us_q0 is not None and (
            args.state2_t1_ms is not None or args.t1_82_us is not None
            or args.t1_rate_scale != 1):
        parser.error("use direct per-qubit T1 values by themselves")
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if args.state2_t1_ms is not None or args.t1_82_us is not None:
        args.include_weak_state2_decay = True
    if args.output is None:
        args.output = (ROOT / "cz/data/fig7_t1_rate_sweep" /
                       f"state2_t1_{args.state2_t1_ms:g}ms_tphi_{args.time_us:g}us.csv"
                       if args.state2_t1_ms is not None else
                       DEFAULT_OUTPUT if args.t1_rate_scale == 1 and not args.include_weak_state2_decay else
                       ROOT / "cz/data/fig7_t1_rate_sweep" /
                       f"t1_rate_x{args.t1_rate_scale:g}"
                       f"{'_state2_decay' if args.include_weak_state2_decay else ''}.csv")
    tphi_us = args.time_us
    t1_us = tphi_us / args.t1_rate_scale

    cfg = {
        "cz_run": True,
        "folder_load": str(args.data),
        "noise_dephase_path": str(args.noise),
        "t1_tphi_other": args.time_us,
        "t1_us": t1_us,
        "tphi_us": tphi_us,
        "filter_ratio": 0.3,
        "apply_decay": True,
        "apply_dephase": True,
        "decay_enlarge": 1,
        "collapse_setup_workers": 1,
        "nonlogical_rate_scale": args.nonlogical_rate_scale,
        "extra_decay_pairs": [(1, 2), (2, 8)] if args.include_weak_state2_decay else [],
    }
    start = time.monotonic()
    data = load_system_data(cfg)
    if args.state2_t1_ms is not None:
        target_us = 1000 * args.state2_t1_ms
        t1_us_q0 = target_us * abs(data["n_theta0"][1, 2] / data["n_theta0"][4, 8]) ** 2
        t1_us_q1 = target_us * abs(data["n_theta1"][1, 2] / data["n_theta1"][4, 8]) ** 2
        cfg.update(t1_us_q0=t1_us_q0, t1_us_q1=t1_us_q1)
    elif args.t1_82_us is not None:
        # build_collapse_ops defines its reference lifetime through the 8->4
        # matrix element. Convert the requested 8->2 lifetime into that
        # existing per-qubit reference convention.
        t1_us_q0 = args.t1_82_us * abs(data["n_theta0"][2, 8] / data["n_theta0"][4, 8]) ** 2
        t1_us_q1 = args.t1_82_us * abs(data["n_theta1"][2, 8] / data["n_theta1"][4, 8]) ** 2
        cfg.update(t1_us_q0=t1_us_q0, t1_us_q1=t1_us_q1)
    elif args.t1_us_q0 is not None:
        t1_us_q0, t1_us_q1 = args.t1_us_q0, args.t1_us_q1
        cfg.update(t1_us_q0=t1_us_q0, t1_us_q1=t1_us_q1)
    else:
        t1_us_q0 = t1_us_q1 = t1_us
    print(
        f"Tphi={tphi_us:.8g} us; reference T1 q0={t1_us_q0:.8g} us, "
        f"q1={t1_us_q1:.8g} us; decay-rate scales "
        f"q0={tphi_us/t1_us_q0:.8g}, q1={tphi_us/t1_us_q1:.8g}",
        flush=True,
    )
    gate = select_gate_quantities(cfg, data)
    indices = data["hspace_n_theta1"][: args.states]
    logical = [indices.index(data["hspace_full"].index(label)) for label in data["logi_state"]]
    H, eigenvectors = ut.build_hamiltonian_2q(
        True, indices, data["eval_tot"], data["eket_tot"], gate["drive_term"]
    )
    collapse = build_collapse_ops(cfg, data, eigenvectors)
    fro_norms = np.array([np.linalg.norm(op.full()) for op in collapse])
    kept = [op for op, norm in zip(collapse, fro_norms) if norm > args.collapse_fro_cutoff]
    omitted_norm_sq = float(np.sum(fro_norms[fro_norms <= args.collapse_fro_cutoff] ** 2))
    print(
        f"Setup {time.monotonic()-start:.1f}s; {len(kept)}/{len(collapse)} collapse "
        f"operators retained; omitted sum ||L||_F^2={omitted_norm_sq:.3e} ns^-1",
        flush=True,
    )
    ideal_options, noisy_options = ut.get_qutip_options(
        args.max_step_ns, args.max_step_ns
    )
    ideal_options.atol = ideal_options.rtol = args.solver_tolerance
    noisy_options.atol = noisy_options.rtol = args.solver_tolerance
    ideal_options.nsteps = noisy_options.nsteps = args.solver_nsteps
    table = pd.read_csv(args.pulses)
    results: list[dict[str, float | int]] = []
    for requested_time in args.gate_times_ns:
        row = table.iloc[(table.tg - requested_time).abs().argmin()]
        if abs(float(row.tg) - requested_time) > 0.1:
            raise ValueError(f"No saved pulse within 0.1 ns of {requested_time} ns")
        pulse = [float(row.tg), float(row.drive_amp), float(row.detune)]
        pulse_args = {
            "drive_amp_A": pulse[1],
            "drive_freq_A": gate["W_20_50"] + 2 * np.pi * pulse[2],
            "gate_time": pulse[0],
        }
        times = np.array([0.0, pulse[0]])
        print(f"Evolving {pulse[0]:.6f} ns pulse", flush=True)
        t0 = time.monotonic()
        superoperator = ut.get_propagator(
            H, times, args.workers, kept, pulse_args, logical,
            ideal_options, noisy_options,
        )
        # get_propagator maps four logical inputs into the full reduced space.
        # Its Kraus operators are (states x 4), so project output rows only.
        raw_kraus = qt.to_kraus(superoperator)
        kraus = [
            qt.Qobj(op.full()[logical, :], dims=[[2, 2], [2, 2]])
            for op in raw_kraus
        ]
        if args.save_kraus is not None:
            args.save_kraus.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.save_kraus.with_suffix(args.save_kraus.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    full_superoperator=superoperator.full(),
                    raw_kraus=np.stack([op.full() for op in raw_kraus]),
                    projected_kraus=np.stack([op.full() for op in kraus]),
                    projected_superoperator=qt.kraus_to_super(kraus).full(),
                    logical_indices=np.asarray(logical, dtype=int),
                    gate_time_ns=float(pulse[0]),
                    drive_amplitude=float(pulse[1]),
                    detune_ghz=float(pulse[2]),
                    tphi_us=float(tphi_us),
                    t1_us_q0=float(t1_us_q0),
                    t1_us_q1=float(t1_us_q1),
                    target_t1_82_us=(float(args.t1_82_us)
                                     if args.t1_82_us is not None else np.nan),
                    t1_rate_scale=float(args.t1_rate_scale),
                    collapse_fro_cutoff=float(args.collapse_fro_cutoff),
                    collapse_ops=len(kept),
                    max_step_ns=float(args.max_step_ns),
                    solver_tolerance=float(args.solver_tolerance),
                )
            temporary.replace(args.save_kraus)
            print(f"Saved reusable channel data to {args.save_kraus}", flush=True)
        raw_output_trace = float(np.real(
            qt.vector_to_operator(
                superoperator * qt.operator_to_vector(qt.qeye(4))
            ).tr()
        ))
        kraus_input_trace = float(np.real(sum(
            (op.dag() * op).tr() for op in raw_kraus
        )))
        choi = qt.to_choi(superoperator).full()
        choi_min_eigenvalue = float(np.linalg.eigvalsh((choi + choi.conj().T) / 2).min())
        dominant_kraus, dominant_angles, dominant_kraus_index, dominant_kraus_weight = (
            ut.cz_phase_correct_dominant(kraus)
        )
        target = cz_gate()
        dominant_fidelity = ut.average_gate_fidelity_trace_decreasing(
            dominant_kraus, target
        )
        _, optimized_angles, full_fidelity, optimization_seed, optimization = (
            ut.cz_phase_optimize_channel(kraus, target)
        )
        common_fidelity = legacy_fidelity = uncorrected_fidelity = np.nan
        common_z_angle_q0 = common_z_angle_q1 = np.nan
        dominant_z_angle_q0, dominant_z_angle_q1 = dominant_angles
        if args.diagnostic_phase_scores:
            common_kraus, common_angles = ut.cz_phase_correct_channel(kraus)
            common_fidelity = ut.average_gate_fidelity_trace_decreasing(common_kraus, target)
            legacy_fidelity = ut.average_gate_fidelity_trace_decreasing(
                ut.cz_phase_correct(kraus), target
            )
            uncorrected_fidelity = ut.average_gate_fidelity_trace_decreasing(kraus, target)
            common_z_angle_q0, common_z_angle_q1 = common_angles
        table_f_open = next(
            (fidelity for gate_time, fidelity in PAPER_TABLE_F_OPEN.items()
             if abs(pulse[0] - gate_time) < 0.1),
            np.nan,
        ) if (args.time_us == 30 and args.t1_rate_scale == 1
              and args.state2_t1_ms is None and args.t1_82_us is None) else np.nan
        saved_fidelity = (
            float(row["f2_60_30us"])
            if (args.time_us == 30 and args.t1_rate_scale == 1
                and args.state2_t1_ms is None and args.t1_82_us is None
                and "f2_60_30us" in row) else np.nan
        )
        result = {
            "tg_ns": pulse[0],
            "amplitude": pulse[1],
            "detune_ghz": pulse[2],
            "time_us": args.time_us,
            "t1_us": (t1_us if args.state2_t1_ms is None
                       and args.t1_82_us is None and args.t1_us_q0 is None else np.nan),
            "t1_us_q0": t1_us_q0,
            "t1_us_q1": t1_us_q1,
            "tphi_us": tphi_us,
            "t1_rate_scale": (args.t1_rate_scale if args.state2_t1_ms is None
                              and args.t1_82_us is None else np.nan),
            "nonlogical_rate_scale": args.nonlogical_rate_scale,
            "t1_rate_scale_q0": tphi_us / t1_us_q0,
            "t1_rate_scale_q1": tphi_us / t1_us_q1,
            "target_state2_t1_ms": args.state2_t1_ms if args.state2_t1_ms is not None else np.nan,
            "target_t1_82_us": args.t1_82_us if args.t1_82_us is not None else np.nan,
            "include_weak_state2_decay": args.include_weak_state2_decay,
            "states": args.states,
            "collapse_ops": len(kept),
            "collapse_fro_cutoff": args.collapse_fro_cutoff,
            "workers": args.workers,
            "max_step_ns": args.max_step_ns,
            "solver_tolerance": args.solver_tolerance,
            "solver_nsteps": args.solver_nsteps,
            "paper_table_f_open": table_f_open,
            "paper_table_error": 1 - table_f_open,
            "saved_leakage_fidelity": saved_fidelity,
            "saved_curve_error": (10 ** float(row["f_60_30us"])
                                  if (args.time_us == 30 and args.t1_rate_scale == 1
                                      and args.state2_t1_ms is None
                                      and args.t1_82_us is None) else np.nan),
            "trace_decreasing_fidelity": full_fidelity,
            "trace_decreasing_error": 1 - full_fidelity,
            "dominant_z_trace_decreasing_fidelity": dominant_fidelity,
            "dominant_z_trace_decreasing_error": 1 - dominant_fidelity,
            "common_z_trace_decreasing_fidelity": common_fidelity,
            "common_z_trace_decreasing_error": 1 - common_fidelity,
            "legacy_per_kraus_fidelity": legacy_fidelity,
            "legacy_per_kraus_error": 1 - legacy_fidelity,
            "uncorrected_trace_decreasing_fidelity": uncorrected_fidelity,
            "uncorrected_trace_decreasing_error": 1 - uncorrected_fidelity,
            "dominant_kraus_index": dominant_kraus_index,
            "dominant_kraus_weight": dominant_kraus_weight,
            "common_z_angle_q0": common_z_angle_q0,
            "common_z_angle_q1": common_z_angle_q1,
            "dominant_z_angle_q0": dominant_z_angle_q0,
            "dominant_z_angle_q1": dominant_z_angle_q1,
            "optimized_z_angle_q0": optimized_angles[0],
            "optimized_z_angle_q1": optimized_angles[1],
            "optimization_seed": optimization_seed,
            "optimization_function_evaluations": optimization.nfev,
            "fidelity_minus_saved": full_fidelity - saved_fidelity,
            "raw_output_trace": raw_output_trace,
            "kraus_input_trace": kraus_input_trace,
            "choi_min_eigenvalue": choi_min_eigenvalue,
            "evolution_seconds": time.monotonic() - t0,
        }
        results.append(result)
        print(result, flush=True)
        if not 0 <= full_fidelity <= 1:
            raise RuntimeError("Unphysical channel fidelity; diagnostic result was not saved")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
