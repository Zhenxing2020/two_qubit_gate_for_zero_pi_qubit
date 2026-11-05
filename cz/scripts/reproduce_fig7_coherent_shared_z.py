#!/usr/bin/env python3
"""Recalculate one coherent CZ pulse with optimized shared local-Z angles."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import qutip as qt
from qutip.qip.operations import cz_gate

from run_2q_fidelity import load_system_data, select_gate_quantities
import utils_2Q_gate_zp as ut


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/data/Two_qubit_data_Sorted_Truc")
DEFAULT_PULSES = ROOT / "figure/data/data_cz_fidelity_leakage.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-time-ns", type=float, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pulses", type=Path, default=DEFAULT_PULSES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=60)
    parser.add_argument("--max-step-ns", type=float, default=0.001)
    parser.add_argument("--solver-tolerance", type=float, default=1e-10)
    parser.add_argument("--solver-nsteps", type=int, default=1000000)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    started = time.monotonic()
    cfg = {"cz_run": True, "folder_load": str(args.data)}
    data = load_system_data(cfg)
    gate = select_gate_quantities(cfg, data)
    indices = list(data["hspace_n_theta1"][: args.states])
    logical = [indices.index(data["hspace_full"].index(label)) for label in data["logi_state"]]
    hamiltonian, _ = ut.build_hamiltonian_2q(
        True, indices, data["eval_tot"], data["eket_tot"], gate["drive_term"]
    )
    ideal_options, noisy_options = ut.get_qutip_options(
        args.max_step_ns, args.max_step_ns
    )
    ideal_options.atol = ideal_options.rtol = args.solver_tolerance
    ideal_options.nsteps = args.solver_nsteps

    table = pd.read_csv(args.pulses)
    row = table.iloc[(table.tg - args.gate_time_ns).abs().argmin()]
    if abs(float(row.tg) - args.gate_time_ns) > 0.1:
        raise ValueError(f"No saved pulse within 0.1 ns of {args.gate_time_ns} ns")
    pulse_args = {
        "drive_amp_A": float(row.drive_amp),
        "drive_freq_A": gate["W_20_50"] + 2 * np.pi * float(row.detune),
        "gate_time": float(row.tg),
    }
    projected = ut.get_propagator(
        hamiltonian, np.array([0.0, float(row.tg)]), 1, [], pulse_args,
        logical, ideal_options, noisy_options,
    )
    kraus = [qt.Qobj(projected.full(), dims=[[2, 2], [2, 2]])]
    target = cz_gate()
    _, optimized_angles, optimized_fidelity, seed, optimization = (
        ut.cz_phase_optimize_channel(kraus, target)
    )
    legacy = ut.cz_phase_correct(kraus)
    legacy_fidelity = ut.average_gate_fidelity_trace_decreasing(legacy, target)
    uncorrected_fidelity = ut.average_gate_fidelity_trace_decreasing(kraus, target)
    trace_mass = float(np.real((kraus[0].dag() * kraus[0]).tr()))

    result = pd.DataFrame([{
        "tg_ns": float(row.tg),
        "amplitude": float(row.drive_amp),
        "detune_ghz": float(row.detune),
        "states": args.states,
        "workers": 1,
        "max_step_ns": args.max_step_ns,
        "solver_tolerance": args.solver_tolerance,
        "trace_decreasing_fidelity": optimized_fidelity,
        "trace_decreasing_error": 1 - optimized_fidelity,
        "legacy_fidelity": legacy_fidelity,
        "legacy_error": 1 - legacy_fidelity,
        "uncorrected_fidelity": uncorrected_fidelity,
        "uncorrected_error": 1 - uncorrected_fidelity,
        "optimized_z_angle_q0": optimized_angles[0],
        "optimized_z_angle_q1": optimized_angles[1],
        "optimization_seed": seed,
        "optimization_function_evaluations": optimization.nfev,
        "logical_trace_mass": trace_mass,
        "average_logical_survival": trace_mass / 4,
        "average_leakage": 1 - trace_mass / 4,
        "evolution_seconds": time.monotonic() - started,
    }])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
