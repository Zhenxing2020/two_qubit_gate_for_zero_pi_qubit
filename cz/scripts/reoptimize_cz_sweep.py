#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Re-optimize every point of a completed CZ sweep, seeding each point from its OWN
previously-optimized pulse.

Use this to refresh a finished sweep -- e.g. to re-score it under the corrected
cz_fidelity_log_noise evaluator (the optimized-large fidelity now matches the DE
objective / fixed-pulse path exactly), or to polish points off a good warm start.
Because each point's DE keeps its prior optimum in the population, f_trunc never
regresses. The original result is preserved by default (a *_reopt copy is made);
pass --in-place to update it directly.

Usage
-----
    conda activate zp2q
    cd cz/scripts
    # refresh a finished run into cz_sweep_<...>_reopt/, warm-started per point:
    python reoptimize_cz_sweep.py --result-dir data/results/cz_sweep_tg91_20260703_144814
    # override parallelism / bounds for the re-opt:
    python reoptimize_cz_sweep.py --result-dir <dir> --workers 10 --n-cpu-inner 1
"""

import os
import argparse

_ORIG_CWD = os.getcwd()

import optimize_cz_sweep as o   # importing this chdir's to the gate dir (cz/)
import utils_2Q_gate_zp as ut


def _resolve(p):
    """Resolve a possibly-relative path against the cwd the user launched from
    (optimize_cz_sweep's import chdir'd us into cz/)."""
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(_ORIG_CWD, p))


def main():
    parser = argparse.ArgumentParser(
        description="Re-optimize every point of a CZ sweep from its previous optimum.")
    parser.add_argument("--result-dir", required=True,
                        help="Existing cz_sweep_* result folder (or its npz).")
    parser.add_argument("--out-dir", default=None,
                        help="Destination for the refreshed result "
                             "(default: <result-dir>_reopt). Ignored with --in-place.")
    parser.add_argument("--folder-load", default=None,
                        help="Sweep-data folder (default: the folder_load saved in "
                             "the result's config).")
    parser.add_argument("--in-place", action="store_true",
                        help="Update the original result instead of copying to *_reopt.")
    parser.add_argument("--no-fixed-eval", action="store_true",
                        help="Skip recomputing the fixed-pulse drift curve after "
                             "re-optimizing (not recommended -- it would be left blank).")
    parser.add_argument("--workers", type=int, default=None,
                        help="DE population workers for the re-opt (default: saved config).")
    parser.add_argument("--n-cpu-inner", type=int, default=None,
                        help="Inner sesolve parallelism per objective (default: saved config).")
    args = parser.parse_args()

    custom = {}
    if args.workers is not None:
        custom["workers"] = args.workers
    if args.n_cpu_inner is not None:
        custom["inner_num_cpus"] = args.n_cpu_inner

    print(os.path.basename(__file__))
    ut.print_time()
    out = o.reoptimize_sweep(
        _resolve(args.result_dir),
        out_dir=_resolve(args.out_dir),
        folder_load=_resolve(args.folder_load),
        custom_config=(custom or None),
        redo_fixed_eval=not args.no_fixed_eval,
        in_place=args.in_place,
    )
    print(f"refreshed result -> {out}")
    ut.print_time()


if __name__ == "__main__":
    main()
