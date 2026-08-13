"""Optimize the 828 ns simultaneous phi+theta X gate.

This script has two modes: an unconstrained fidelity optimization and an
optimization for which the largest transient population of bare eigenstate 9
(for either logical input state) is constrained to 12 percent by default.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import qutip as qt
import scipy.optimize as opt
import scqubits.settings as settings


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(os.environ.get("XGATE_PROJECT_DIR", HERE.parent)).resolve()
sys.path.insert(0, str(DEFAULT_PROJECT))
import utils_2Q_gate_zp as ut  # noqa: E402
settings.OVERLAP_THRESHOLD = 0.3
# The legacy data loaders use paths relative to the xgate directory.
os.chdir(DEFAULT_PROJECT / "xgate")


# Initial pulse from scripts/run_xgate_fidelity.py.  Units are ns for time,
# effective drive amplitude for the two amplitudes, and GHz for detunings.
INITIAL = np.array([828.759495, 0.013563, 0.034964, -0.003029, -0.003182])
PARAM_NAMES = ["tg", "drive_amp_1", "drive_amp_2", "detune_1", "detune_2"]


def _bool(text):
    return str(text).lower() in {"1", "true", "yes", "y"}


class GateEvaluator:
    """Evaluate fidelity and (optionally) transient intermediate population."""

    def __init__(self, H, w1, w2, logical, intermediate, options,
                 population_points, use_qt_fidelity):
        self.H = H
        self.w1, self.w2 = w1, w2
        self.logical = logical
        self.intermediate = intermediate
        self.options = options
        self.population_points = population_points
        self.use_qt_fidelity = use_qt_fidelity
        self.cache = {}

    def _key(self, x, need_population):
        return (tuple(np.asarray(x, dtype=float)), bool(need_population))

    def evaluate(self, x, need_population=False):
        key = self._key(x, need_population)
        if key in self.cache:
            return self.cache[key]
        tg, amp1, amp2, det1, det2 = map(float, x)
        pulse = {
            "drive_amp_A": amp1,
            "drive_freq_A": self.w1 + 2 * np.pi * det1,
            "drive_amp_B": amp2,
            "drive_freq_B": self.w2 + 2 * np.pi * det2,
            "gate_time": tg,
        }
        # Only the final state is needed in the direct task.  The constrained
        # task samples the whole trajectory; QuTiP still controls its internal
        # integration accuracy through max_step.
        tlist = (np.linspace(0.0, tg, self.population_points)
                 if need_population else np.array([0.0, tg]))
        columns = np.zeros((self.H[0].shape[0], len(self.logical)), complex)
        max_population = 0.0
        for col, state_index in enumerate(self.logical):
            result = qt.sesolve(
                self.H, qt.basis(self.H[0].shape[0], state_index), tlist,
                args=pulse, options=self.options,
            )
            columns[:, col] = result.states[-1].full().ravel()
            if need_population:
                max_population = max(
                    max_population,
                    max(abs(state.full().ravel()[self.intermediate]) ** 2
                        for state in result.states),
                )
        logical_map = qt.Qobj(columns[np.asarray(self.logical), :])
        fidelity = ut.get_fidelity_super_operator(
            logical_map, self.logical, qt.sigmax(), [],
            use_qt_fidelity=self.use_qt_fidelity,
        )
        infidelity = max(1.0 - float(np.real(fidelity)), np.finfo(float).tiny)
        answer = (float(np.log10(infidelity)), float(max_population))
        self.cache[key] = answer
        # Keep memory bounded during long differential-evolution runs.
        if len(self.cache) > 8:
            self.cache.pop(next(iter(self.cache)))
        return answer

    def objective(self, x):
        return self.evaluate(x, False)[0]

    def objective_with_population(self, x):
        return self.evaluate(x, True)[0]

    def population(self, x):
        return self.evaluate(x, True)[1]


def save_result(path, task, result, evaluator, population_limit):
    log_error, max_population = evaluator.evaluate(result.x, True)
    row = dict(zip(PARAM_NAMES, map(float, result.x)))
    row.update({
        "task": task,
        "log10_infidelity": log_error,
        "fidelity": 1.0 - 10.0 ** log_error,
        "max_intermediate_population": max_population,
        "population_limit": population_limit,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "finished": datetime.now().isoformat(),
    })
    pd.DataFrame([row]).to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    print("FINAL_RESULT", json.dumps(row), flush=True)


def optimization_828(args):
    args.workers = 32 if args.workers is None else args.workers
    args.popsize = 10 if args.popsize is None else args.popsize
    args.maxiter = 200 if args.maxiter is None else args.maxiter
    args.tol = 1e-3 if args.tol is None else args.tol
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_base = out / args.task
    print("task =", args.task, "initial =", INITIAL.tolist(), flush=True)

    evals, n_theta, n_phi, logical_states = ut.load_qubit_data_xgate(qubit_0=True)
    w1, w2, drive, _ = ut.compute_drive_xgate(
        evals, n_theta, n_phi, True, True, gamma_t1=0,
    )
    hspace = ut.get_truncated_subspace_xgate(drive, args.n_full)
    if args.intermediate_state not in hspace:
        raise ValueError(f"intermediate state {args.intermediate_state} absent from hspace {hspace}")
    H, _, logical = ut.build_hamiltonian_xgate(evals, drive, hspace, logical_states)
    intermediate = hspace.index(args.intermediate_state)
    options = qt.Options(
        max_step=args.max_step, nsteps=10_000_000, store_states=True, num_cpus=1,
    )
    evaluator = GateEvaluator(H, w1, w2, logical, intermediate, options,
                              args.population_points, args.use_qt_fidelity)

    # Local bounds centered on the validated 828.759495 ns pulse.  They allow
    # ample movement while keeping DE on the requested pulse family.
    bounds = [
        (INITIAL[0] - args.tg_window, INITIAL[0] + args.tg_window),
        (0.002, 0.03), (0.015, 0.055), (-0.008, 0.002), (-0.008, 0.002),
    ]
    run_config = {
        "started": datetime.now().isoformat(),
        "pid": os.getpid(),
        "task": args.task,
        "output_dir": str(out),
        "initial": dict(zip(PARAM_NAMES, map(float, INITIAL))),
        "bounds": dict(zip(PARAM_NAMES, bounds)),
        "n_full": args.n_full,
        "hamiltonian_dim": H[0].shape[0],
        "state_range": [hspace[0], hspace[-1]],
        "logical_indices": logical,
        "intermediate_state": args.intermediate_state,
        "population_limit": args.population_limit,
        "population_points": args.population_points,
        "max_step": args.max_step,
        "solver_nsteps": 10_000_000,
        "workers": args.workers,
        "popsize": args.popsize,
        "maxiter": args.maxiter,
        "tol": args.tol,
        "seed": args.seed,
        "init_method": "sobol",
        "mutation": [0.5, 1.0],
        "recombination": 0.7,
        "polish": False,
        "use_qt_fidelity": args.use_qt_fidelity,
        "tg_window": args.tg_window,
        "check_only": args.check_only,
        "thread_environment": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS")
        },
    }
    config_text = json.dumps(run_config, indent=2)
    result_base.with_suffix(".config.json").write_text(config_text + "\n")
    print("RUN_CONFIG", config_text, flush=True)

    # Evaluate x0 before DE so the log makes it explicit whether the supplied
    # high-fidelity pulse is feasible under the population constraint.
    initial_log_error, initial_population = evaluator.evaluate(INITIAL, True)
    constraint_active = args.task == "population-limited"
    initial_evaluation = {
        "log10_infidelity": initial_log_error,
        "infidelity": 10.0 ** initial_log_error,
        "fidelity": 1.0 - 10.0 ** initial_log_error,
        "max_intermediate_population": initial_population,
        "intermediate_state": args.intermediate_state,
        "constraint_active": constraint_active,
        "population_limit": args.population_limit if constraint_active else None,
        "constraint_margin": (
            args.population_limit - initial_population
            if constraint_active else None
        ),
        "constraint_feasible": (
            initial_population <= args.population_limit
            if constraint_active else None
        ),
    }
    result_base.with_suffix(".initial.json").write_text(
        json.dumps(initial_evaluation, indent=2) + "\n"
    )
    print("INITIAL_EVALUATION", json.dumps(initial_evaluation, indent=2),
          flush=True)

    if args.check_only:
        print("CHECK_ONLY complete", flush=True)
        return

    constraints = ()
    objective = evaluator.objective
    if args.task == "population-limited":
        constraints = (opt.NonlinearConstraint(
            evaluator.population, -np.inf, args.population_limit,
        ),)
        objective = evaluator.objective_with_population

    # Write discoverable status before the expensive call begins.
    (result_base.with_suffix(".status")).write_text(
        f"RUNNING pid={os.getpid()} started={datetime.now().isoformat()}\n"
    )

    def callback(xk, convergence):
        f, pop = evaluator.evaluate(xk, args.task == "population-limited")
        checkpoint = dict(zip(PARAM_NAMES, map(float, xk)))
        checkpoint.update(log10_infidelity=f, max_intermediate_population=pop,
                          convergence=float(convergence), updated=datetime.now().isoformat())
        result_base.with_suffix(".checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2) + "\n"
        )
        print("CHECKPOINT", json.dumps(checkpoint), flush=True)
        return False

    try:
        result = opt.differential_evolution(
            objective, bounds, constraints=constraints, x0=INITIAL, init="sobol",
            popsize=args.popsize, maxiter=args.maxiter, tol=args.tol,
            mutation=(0.5, 1.0), recombination=0.7, seed=args.seed,
            workers=args.workers, updating="immediate" if args.workers == 1 else "deferred",
            callback=callback, disp=True, polish=False,
        )
        save_result(result_base, args.task, result, evaluator,
                    args.population_limit if constraints else None)
        result_base.with_suffix(".status").write_text(
            f"FINISHED pid={os.getpid()} at={datetime.now().isoformat()}\n"
        )
    except BaseException as exc:
        result_base.with_suffix(".status").write_text(
            f"FAILED pid={os.getpid()} at={datetime.now().isoformat()} error={exc!r}\n"
        )
        raise


def legacy_optimization(args):
    """Run the original pulse-table phi or theta optimization."""
    import scipy as sp
    from tqdm import tqdm

    workers = 100 if args.workers is None else args.workers
    popsize = 10 if args.popsize is None else args.popsize
    maxiter = 1000 if args.maxiter is None else args.maxiter
    tol = 0.01 if args.tol is None else args.tol

    drive_phi, drive_theta = args.drive == "phi", args.drive == "theta"
    pulse_file = ("../figure/data/data_xgate_phi_mstep_1e3_npz.txt"
                  if drive_phi else "../figure/data/data_xgate_theta_mstep_3e4_npz.txt")
    fidelity_column = "f_charge_160_npz" if drive_phi else "f_157_charge"
    pulse_table = pd.read_csv(pulse_file)
    initial_params = pulse_table[PARAM_NAMES].to_numpy()[args.start_index:]
    initial_fidelity = pulse_table[fidelity_column].to_numpy()[args.start_index:]
    if drive_theta:
        amp1_bounds, amp2_bounds = (0, 0.45), (0, 0.1)
        detune1_bounds, detune2_bounds = (-0.1, 0.05), (-0.05, 0.05)
    else:
        amp1_bounds, amp2_bounds = (0.15, 0.3), (0, 0.3)
        detune1_bounds, detune2_bounds = (0.3, 0.5), (0.3, 0.5)

    max_step = 3e-4 if drive_theta else 1e-3
    options = qt.Options(max_step=max_step, nsteps=10 / max_step,
                         store_states=True, num_cpus=1)
    evals, n_theta, n_phi, logical_states = ut.load_qubit_data_xgate()
    w1, w2, drive = ut.compute_drive_xgate(evals, n_theta, n_phi,
                                            drive_phi, drive_theta)
    hspace = ut.get_truncated_subspace_xgate(drive, args.legacy_n_truc)
    H, _, logical = ut.build_hamiltonian_xgate(evals, drive, hspace,
                                                logical_states)
    objective_args = [H, w1, w2, 1, [], logical, options, None,
                      args.use_qt_fidelity]
    print(Path(__file__).name)
    print("drive_phi=", drive_phi, ", drive_theta=", drive_theta)
    print("pulse_file=", pulse_file, "start_index=", args.start_index)
    ut.print_fidelity("initial_fidelity", initial_fidelity)
    print("n_truc=", args.legacy_n_truc, ", n_optimize=", len(hspace))
    for initial in tqdm(initial_params, total=len(initial_params)):
        bounds = ((initial[0] - 0.01, initial[0] + 0.01), amp1_bounds,
                  amp2_bounds, detune1_bounds, detune2_bounds)
        result = sp.optimize.differential_evolution(
            ut.xgate_fidelity_log_noise, bounds, args=objective_args,
            disp=True, callback=ut.print_soln, init="sobol", x0=initial,
            workers=workers, popsize=popsize, mutation=(0.5, 1.0),
            recombination=0.7, tol=tol, polish=False, maxiter=maxiter)
        print(f"\nOptimal result for tg={initial[0]}:")
        print(result)
        print("Drive parameters:", np.round(result.x, 6).tolist())
        ut.print_fidelity("f_optimize", [result.fun])
        ut.print_time()


def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize legacy phi/theta X gates or the 828 ns simultaneous gate.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--drive", choices=("phi", "theta"))
    mode.add_argument("--task", choices=("direct", "population-limited"))
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--output-dir", default=str(HERE / "xgate_828_optimization"))
    p.add_argument("--n-full", type=int, default=50)
    p.add_argument("--legacy-n-truc", type=int, default=300)
    p.add_argument("--max-step", type=float, default=3e-4)
    p.add_argument("--population-points", type=int, default=1001)
    p.add_argument("--population-limit", type=float, default=0.12,
                   help="maximum transient population of the intermediate state (default: 0.12)")
    p.add_argument("--intermediate-state", type=int, default=9)
    p.add_argument("--workers", type=int,
                   help="differential-evolution workers (default: 30 for 828 ns tasks)")
    p.add_argument("--popsize", type=int)
    p.add_argument("--maxiter", type=int)
    p.add_argument("--tol", type=float)
    p.add_argument("--seed", type=int, default=828)
    p.add_argument("--use-qt-fidelity", type=_bool, default=False,
                   help="use QuTiP fidelity instead of the default leakage-inclusive trace-decreasing formula")
    p.add_argument("--tg-window", type=float, default=2.0)
    p.add_argument("--check-only", action="store_true",
                   help="evaluate the initial 828 ns pulse once, without optimizing")
    return p.parse_args()


def main():
    args = parse_args()
    optimization_828(args) if args.task else legacy_optimization(args)


if __name__ == "__main__":
    main()
