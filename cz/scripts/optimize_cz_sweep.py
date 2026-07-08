#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Optimize a CZ gate at every point of a parameter sweep produced by
ham_data.save_two_qubit_sweep (EC2, EJ2, ECJ2, EL2, Ecc, ...).

Design
------
* Parameter-agnostic. The sweep npz is self-describing (it stores
  ``sweep_param_name``/``sweep_param_values``), and the physics enters only
  through each point's ``eval_tot`` and dressed drive operator. So the same
  script optimizes a CZ gate for an EC2 sweep, an EJ2 sweep, etc., unchanged.

* Coherent only. The sweep drops ``evecs_tot``, so the loader returns
  ``eket_tot=None``. That is exactly what the ideal (Schrodinger) CZ fidelity
  needs; noisy collapse operators are not available and are not used here.

* Serial across sweep points, warm-started outward from the center.
  Differential evolution already parallelizes internally (``workers``), so a
  serial outer loop keeps every optimization fully parallel while letting each
  point reuse its neighbor's solution:
    - even number of points -> DUAL anchor at the two central points, both
      seeded from the pre-optimized pulse file;
    - odd number of points  -> SINGLE central anchor seeded from the pulse file;
  every non-anchor point is seeded from the neighbor one step toward the center.

* The anchor(s) are re-optimized too. Because the seed pulse is already
  optimized near the sweep center, the anchor result should barely move; a large
  change there is a red flag that something is off.

Usage
-----
    conda activate zp2q
    cd cz/scripts        # scripts expect the gate dir as cwd
    python optimize_cz_sweep.py
"""

import os
import sys
import json
import shutil
import multiprocessing.pool
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy as sp
from tqdm import tqdm
from matplotlib import pyplot as plt
import scqubits.settings as settings

settings.OVERLAP_THRESHOLD = 0.3


# --- Non-daemonic pool so inner parallelism can nest inside DE's workers ------
# scipy.differential_evolution(workers=int) parallelizes the population with a
# multiprocessing.Pool whose workers are daemonic, and daemonic processes may
# not spawn children -- which blocks the inner qutip parallel_map over the 4
# logical-state sesolves. Passing workers=NestablePool(...).map instead runs the
# population on NON-daemonic workers, so the inner 4-way pool can nest, giving
# workers x inner_num_cpus processes total.
class _NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonContext(type(multiprocessing.get_context())):
    Process = _NoDaemonProcess


class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = _NoDaemonContext()
        super().__init__(*args, **kwargs)

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)

import utils_2Q_gate_zp as ut
import ham_data as hd
from optimize_2q_fidelity import (
    get_optimization_config,
    build_hamiltonians_cz,
    evaluate_fidelity_large,
)


# === globals read by the DE objective (populated per sweep point before each
#     differential_evolution call; workers inherit them via fork on Linux) ===
GLOBAL_system_data = None
GLOBAL_hamiltonians = None


def cz_objective(x, num_cpus=1):
    """Truncated ideal CZ log-error objective for differential evolution."""
    sd, hm = GLOBAL_system_data, GLOBAL_hamiltonians
    args = [
        hm["H_drive_select"],
        sd["W_20_50"],
        num_cpus,          # num_cpus (inner); DE parallelism is at the population level
        [],         # no collapse ops -> ideal evolution
        hm["logi_idx_select"],
        sd["option_ideal"],
        sd["option_noisy"],
    ]
    return ut.cz_fidelity_log_noise(list(x), *args)


# ==============================================================
# CONFIG
# ==============================================================
def get_sweep_config(custom_config=None):
    """CZ optimization config plus sweep-specific keys."""
    config = get_optimization_config(gate_type="CZ")

    config.update(dict(
        # Sweep data produced by save_two_qubit_sweep (folder holding
        # two_qubit_sweep_data.npz).
        folder_load="/data/zp_sweeps/test/EC2_sweep",

        # Seed pulse for the center anchor(s): a pre-optimized CZ pulse
        # [tg, amp, detune] for the nominal parameter value. Row 71 of the full
        # file is tg=90.9375 ns (the tg~=91 ns default gate).
        folder_pulse="../figure/data/data_cz_fidelity_npz_full.txt",
        seed_tg_idx=71,         # which row of the pulse file to seed from

        # Hilbert-space truncations: optimize on truc_optimize states (cheap,
        # drives DE), then re-evaluate the optimum on truc_large states for an
        # accurate per-point fidelity. (truc=1000 matches the converged file
        # values; truc=50 -- the get_optimization_config default -- does not.)
        truc_optimize=300,
        truc_large=1000,

        # Gate-time search window (ns) around the seed tg. Wider than the
        # near-pinned get_optimization_config default so tg can drift as the
        # swept parameter changes; the bound recenters on each point's seed tg.
        tg_bound=(-2.0, 2.0),

        # Absolute drive-amplitude search range. Lowered from the
        # get_optimization_config default (0.01, 0.05) so longer/weaker-drive
        # gates fit (e.g. the tg~182 ns seed amp is ~0.0079). optimize_cz_point
        # also auto-widens any bound that would exclude the seed.
        amp_bound=(0.001, 0.1),

        # Also evaluate each optimum on the larger Hilbert space.
        do_large=True,

        # Final validation pass: hold the pulse calibrated at the sweep midpoint
        # fixed and evaluate it (at truc_large) across the sweep -- the drift
        # sensitivity, overlaid on the summary gate-error panel.
        do_fixed_eval=True,

        # Dry run: optimize ONLY the central anchor(s) and report their shift
        # from the seed, so the red-flag check can be eyeballed before
        # committing to the full multi-hour sweep.
        dry_run=False,

        # Inner (per-objective) parallelism over the 4 logical-state sesolves,
        # nested inside DE's population workers via NestablePool (see
        # optimize_cz_point). Total processes ~= min(workers, population) *
        # inner_num_cpus; keep that <= core count to avoid oversubscription.
        inner_num_cpus=2,

        # Resume: skip sweep indices already saved in result_file.
        resume=False,
    ))

    # Apply overrides before building the label so e.g. seed_tg_idx is reflected.
    if custom_config:
        config.update(custom_config)

    # get_optimization_config already created an empty cz_<ts> result dir that
    # we are about to replace; drop it if still empty so it does not litter.
    old_dir = config.get("result_dir")
    if old_dir and os.path.isdir(old_dir) and not os.listdir(old_dir):
        os.rmdir(old_dir)

    # Label the output dir with the swept parameter AND the seed gate time so
    # runs over different sweeps (EC2 vs Ecc vs ...) and different gate durations
    # (tg~91 vs tg~182) are self-documenting and never collide -- important when
    # launching many sweeps at once. The parameter tag comes from the folder_load
    # basename (e.g. "/data/.../EC2_sweep" -> "EC2").
    seed_tg = ut.load_drive_params_2q(True, folder=config["folder_pulse"])[config["seed_tg_idx"], 0]
    param_tag = os.path.basename(os.path.normpath(config["folder_load"])).replace("_sweep", "")
    label = f"cz_sweep_{param_tag}_tg{round(seed_tg)}_{config['timestamp']}"
    base_dir = f"data/results/{label}"
    os.makedirs(base_dir, exist_ok=True)
    config.update(dict(
        result_dir=base_dir,
        result_file=os.path.join(base_dir, f"{label}.npz"),
        csv_file=os.path.join(base_dir, f"{label}.csv"),
        plot_dir=base_dir,
        # Fixed seed-pulse gate time; tg_absolute anchors the tg_bound window on
        # this (constant across the sweep) instead of each point's drifting seed.
        seed_tg=float(seed_tg),
    ))
    return config


# ==============================================================
# PER-POINT SYSTEM DATA
# ==============================================================
def load_cz_sweep_point(config, index):
    """Build the CZ system_data dict for a single sweep index."""
    (hspace_full, eket_tot, eval_tot, _, _, _, n_theta1_dress, _, _,
     _, _, logi_state) = hd.load_two_qubit_sweep(config["folder_load"], index=index)

    drive_term = n_theta1_dress
    W_20_50 = eval_tot[hspace_full.index("5-0")] - eval_tot[hspace_full.index("2-0")]

    if config["use_truc_model"]:
        hspace_select = ut.truc_model[config["truc_model_name"]][:config["truc_optimize"]]
    else:
        hspace_select = hspace_full[:config["truc_optimize"]]

    option_ideal, option_noisy = ut.get_qutip_options(
        config["max_step_ideal"], config["max_step_noisy"]
    )

    return dict(
        gate_type="CZ",
        hspace_full=hspace_full,
        eket_tot=eket_tot,          # None for sweep data
        eval_tot=eval_tot,
        drive_term=drive_term,
        logi_state=logi_state,
        hspace_select=hspace_select,
        W_20_50=W_20_50,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
    )


# ==============================================================
# OPTIMIZATION
# ==============================================================
def optimize_cz_point(config, index, x0):
    """Load one sweep point, build Hamiltonians, and run DE seeded at x0.

    Returns (f_trunc, f_large, optimized_params, W_20_50).
    """
    global GLOBAL_system_data, GLOBAL_hamiltonians

    system_data = load_cz_sweep_point(config, index)
    hamiltonians = build_hamiltonians_cz(system_data, config)

    GLOBAL_system_data = system_data
    GLOBAL_hamiltonians = hamiltonians

    x0 = list(x0)
    tg0 = x0[0]
    # tg_bound is a +/-window around a center gate time. By default the center is
    # this point's own (warm-started, drifting) seed tg0, so the window walks
    # with the sweep. With tg_absolute the center is instead the fixed seed-pulse
    # tg -- the SAME window for every point -- so gate time stays pinned across
    # the sweep rather than drifting cumulatively.
    tg_center = config.get("seed_tg", tg0) if config.get("tg_absolute", False) else tg0
    tg_bounds = (tg_center + config["tg_bound"][0], tg_center + config["tg_bound"][1])
    bounds = [list(tg_bounds), list(config["amp_bound"]), list(config["detune_bound"])]

    # DE requires x0 within bounds. A seed pulse for a different gate regime
    # (e.g. the weaker-drive tg~182 ns gate) can fall outside a configured
    # bound; widen that bound to include the seed plus a 10%-of-span margin so
    # DE still has room to move, rather than crashing on "x0 outside bounds".
    for k, label in enumerate(("tg", "amp", "detune")):
        lo, hi = bounds[k]
        margin = 0.1 * (hi - lo)
        if x0[k] < lo:
            bounds[k][0] = x0[k] - margin
            print(f"[WARN] seed {label}={x0[k]:.6g} < bound; widened low to {bounds[k][0]:.6g}")
        elif x0[k] > hi:
            bounds[k][1] = x0[k] + margin
            print(f"[WARN] seed {label}={x0[k]:.6g} > bound; widened high to {bounds[k][1]:.6g}")
    bounds = tuple(tuple(b) for b in bounds)

    inner_cpus = config.get("inner_num_cpus", 1)

    print(f"\n--- CZ optimize sweep index {index}, W_20_50={system_data['W_20_50']:.4f}, x0={np.round(x0, 6)} ---")
    ut.print_time()

    de_kwargs = dict(
        func=cz_objective,
        bounds=bounds,
        args=(inner_cpus,),
        x0=x0,
        init="sobol",
        disp=True,
        callback=ut.print_soln,
        popsize=config["popsize"],
        mutation=config["mutation"],
        recombination=config["recombination"],
        tol=config["tol"],
        polish=False,
    )

    if config["workers"] == 1:
        # Serial DE; inner parallelism (if any) runs in this process, no nesting.
        result = sp.optimize.differential_evolution(workers=1, **de_kwargs)
    else:
        # Population on a non-daemonic pool so inner_cpus>1 can nest.
        # NestablePool is forked here (after the globals above are set) so its
        # workers inherit THIS point's system_data/hamiltonians.
        with NestablePool(processes=config["workers"]) as pool:
            result = sp.optimize.differential_evolution(workers=pool.map, **de_kwargs)

    params = result.x.tolist()
    f_trunc = float(result.fun)
    f_large = float(evaluate_fidelity_large(params, system_data, hamiltonians, config)) \
        if config["do_large"] else np.nan

    print(f"index {index}: f_trunc={f_trunc:.8f}, f_large={f_large:.8f}, params={np.round(params, 6)}")
    ut.print_time()
    return f_trunc, f_large, params, float(system_data["W_20_50"])


def build_anchor_and_passes(n):
    """Return (anchors, down_range, up_range).

    anchors : sorted list of central indices seeded from the pulse file
              (two for even n, one for odd n).
    down_range : indices below the lowest anchor, expanding outward (descending).
    up_range   : indices above the highest anchor, expanding outward (ascending).
    """
    if n % 2 == 0:
        anchors = [n // 2 - 1, n // 2]
    else:
        anchors = [n // 2]
    down_range = list(range(min(anchors) - 1, -1, -1))
    up_range = list(range(max(anchors) + 1, n))
    return anchors, down_range, up_range


# ==============================================================
# SAVE / RESUME / PLOT
# ==============================================================
def save_results(config, name, values, f_trunc, f_large, params, W, done,
                 f_fixed=None, ref_index=-1):
    # f_fixed: drift curve of the fixed (calibrated-at-ref) pulse; NaN until the
    # end-of-sweep pass fills it. ref_index: the calibration sweep index (-1 = none).
    if f_fixed is None:
        f_fixed = np.full(len(values), np.nan)
    np.savez(
        config["result_file"],
        config=json.dumps(config, default=str),
        sweep_param_name=name,
        sweep_param_values=values,
        f_trunc=f_trunc,
        f_large=f_large,
        f_fixed=f_fixed,
        ref_index=ref_index,
        drive_params=params,
        W_20_50=W,
        done=done,
    )
    # csv (one row per sweep point)
    import csv
    with open(config["csv_file"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([name, "done", "f_trunc", "f_large", "f_fixed", "tg", "amp", "detune", "W_20_50"])
        for i in range(len(values)):
            p = params[i] if done[i] else [np.nan, np.nan, np.nan]
            w.writerow([values[i], bool(done[i]), f_trunc[i], f_large[i], f_fixed[i],
                        p[0], p[1], p[2], W[i]])
    n_done = int(np.sum(done))
    print(f"[SAVE] {n_done}/{len(values)} points -> {config['result_file']} + {config['csv_file']}")


def load_resume(config, n):
    """Return (f_trunc, f_large, params, W, done) arrays, restored if resuming."""
    f_trunc = np.full(n, np.nan)
    f_large = np.full(n, np.nan)
    params = [[np.nan, np.nan, np.nan] for _ in range(n)]
    W = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)

    if config.get("resume") and os.path.exists(config["result_file"]):
        d = np.load(config["result_file"], allow_pickle=True)
        f_trunc = d["f_trunc"]
        f_large = d["f_large"]
        params = d["drive_params"].tolist()
        W = d["W_20_50"]
        done = d["done"]
        print(f"[RESUME] {int(done.sum())}/{n} points already done.")
    return f_trunc, f_large, params, W, done


# CZ-relevant dressed-state transitions (same pairs printed in the two-qubit
# summary "-- CZ gate --" block). Each is |E(s2) - E(s1)| in GHz.
CZ_TRANSITION_PAIRS = [
    ("2-2", "2-5"), ("0-2", "0-5"),
    ("2-0", "2-1"), ("0-0", "0-1"),
    ("2-2", "5-2"), ("2-0", "5-0"),
    ("0-2", "1-2"), ("0-0", "1-0"),
]


def compute_cz_transitions(folder_load):
    """
    CZ transition frequencies vs the swept parameter, from the sweep npz spectrum.

    evals_tot is stored without the 2*pi factor (GHz), matching the values in
    the per-point summary text files.

    Returns
    -------
    (values, {("s1","s2"): freqs_array_over_sweep})  or  None if unavailable.
    """
    try:
        d = np.load(Path(folder_load, "two_qubit_sweep_data.npz"), allow_pickle=True)
    except FileNotFoundError:
        print(f"[WARN] no two_qubit_sweep_data.npz in {folder_load}; skipping transition panel.")
        return None
    values = np.asarray(d["sweep_param_values"], dtype=float)
    evals = d["evals_tot"]
    hspace = d["hspace_full"]
    freqs = {}
    for s1, s2 in CZ_TRANSITION_PAIRS:
        arr = np.full(len(values), np.nan)
        for i in range(len(values)):
            hs = list(hspace[i])
            ev = np.real(evals[i])
            ev = ev - ev[0]
            if s1 in hs and s2 in hs:
                arr[i] = abs(ev[hs.index(s1)] - ev[hs.index(s2)])
        freqs[(s1, s2)] = arr
    return values, freqs


def _apply_symlog_x(axes, values):
    """Put the x-axis on a symmetric-log scale when the swept values straddle zero.

    Sweeps like ng2 and Φ2 are log-spaced symmetrically about 0 (negative and
    positive points plus 0), so a plain linear axis crams the small-magnitude
    points against 0. symlog gives a log axis on BOTH sides with a small linear
    bridge across 0 (width = smallest nonzero |value|). No-op for the all-positive
    (or all-negative) sweeps, so it is safe to call unconditionally. `axes` may be
    a single Axes or any iterable of Axes.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not (v.min() < 0 < v.max()):
        return
    nz = np.abs(v[v != 0])
    linthresh = float(nz.min()) if nz.size else 1e-12
    try:
        ax_list = list(axes)
    except TypeError:
        ax_list = [axes]
    for a in ax_list:
        a.set_xscale("symlog", linthresh=linthresh)


def _plot_summary_core(name, values, f_trunc, f_large, params, done,
                       truc_optimize, truc_large, do_large, save_path,
                       transitions=None, f_fixed=None, ref_value=None):
    """Two-column summary.

    Top row: gate error, OPTIMIZED per point (left) vs FIXED pulse / drift
    sensitivity (right), sharing y-limits so the drift cliff is directly
    comparable. Rows below: CZ transitions, gate time, drive amplitude, detuning.
    """
    values = np.asarray(values, dtype=float)
    done = np.asarray(done, dtype=bool)
    f_trunc = np.asarray(f_trunc, dtype=float)
    f_large = np.asarray(f_large, dtype=float)
    # drive_params rows are [tg, amp, detune]; blank out not-done points.
    arr = np.array([p if d else [np.nan] * 3 for p, d in zip(params, done)], dtype=float)
    ff = np.asarray(f_fixed, dtype=float) if f_fixed is not None else None
    has_fixed = ff is not None and np.any(np.isfinite(ff))

    # Panels below the top row (filled 2-per-row, in order).
    secondary = (["transitions"] if transitions is not None else []) + ["tg", "amp", "detune"]
    n_rows = 1 + int(np.ceil(len(secondary) / 2))

    fig, ax = plt.subplots(n_rows, 2, figsize=(11, 2.7 * n_rows), sharex=True, squeeze=False)
    fig.suptitle(f"CZ optimization vs {name}", y=0.995)
    used = {(0, 0), (0, 1)}

    # --- top-left: optimized per point ---
    a = ax[0][0]
    a.plot(values, 10.0 ** f_trunc, ".-", label=f"trunc {truc_optimize}")
    if do_large:
        a.plot(values, 10.0 ** f_large, "o", alpha=0.7, label=f"large {truc_large}")
    a.set_title("Optimized per point")
    a.set_ylabel("Gate error (1-F)")
    a.set_yscale("log")
    a.legend(fontsize=7)
    a.grid(True, which="both", alpha=0.3)

    # --- top-right: fixed pulse (un-optimized / drift) ---
    a = ax[0][1]
    if has_fixed:
        a.plot(values, 10.0 ** ff, "x--", color="crimson", label=f"fixed pulse (truc {truc_large})")
        if ref_value is not None:
            a.axvline(ref_value, color="k", ls=":", lw=1, alpha=0.6, label=f"calib @ {ref_value:.4f}")
        a.legend(fontsize=7)
    else:
        a.text(0.5, 0.5, "no fixed-pulse eval\n(do_fixed_eval=False)",
               ha="center", va="center", transform=a.transAxes, color="gray")
    a.set_title("Fixed pulse (calibrated at nominal)")
    a.set_ylabel("Gate error (1-F)")
    a.set_yscale("log")
    a.grid(True, which="both", alpha=0.3)

    # Share y-limits across the two top panels for visual comparison.
    err = [10.0 ** f_trunc[np.isfinite(f_trunc)]]
    if do_large:
        err.append(10.0 ** f_large[np.isfinite(f_large)])
    if has_fixed:
        err.append(10.0 ** ff[np.isfinite(ff)])
    err = np.concatenate([e for e in err if e.size])
    if err.size:
        ylim = (err.min() * 0.6, err.max() * 1.6)
        ax[0][0].set_ylim(ylim)
        ax[0][1].set_ylim(ylim)

    # --- secondary panels ---
    slots = [(r, c) for r in range(1, n_rows) for c in range(2)]
    for panel, (r, c) in zip(secondary, slots):
        a = ax[r][c]
        used.add((r, c))
        if panel == "transitions":
            for (s1, s2), fr in transitions[1].items():
                i1, j1 = s1.split("-"); i2, j2 = s2.split("-")
                a.plot(transitions[0], fr, ".-", lw=1, label=f"|{i1},{j1}⟩→|{i2},{j2}⟩")
            a.set_ylabel("CZ transition (GHz)")
            a.legend(fontsize=5, ncol=2)
        elif panel == "tg":
            a.plot(values, arr[:, 0], ".-"); a.set_ylabel("Gate time (ns)")
        elif panel == "amp":
            a.plot(values, arr[:, 1], ".-"); a.set_ylabel("Drive amplitude")
        elif panel == "detune":
            a.plot(values, arr[:, 2], ".-"); a.set_ylabel("Detuning")
        a.grid(True, alpha=0.3)

    # Turn off any unused slots, and x-label the lowest used panel in each column.
    for (r, c) in slots:
        if (r, c) not in used:
            ax[r][c].axis("off")
    for c in range(2):
        rows_used = [r for r in range(n_rows) if (r, c) in used]
        bottom = max(rows_used)
        ax[bottom][c].set_xlabel(name)
        ax[bottom][c].tick_params(labelbottom=True)  # re-enable under sharex

    # Symmetric-log x for zero-straddling sweeps (ng2/Φ2); no-op otherwise.
    _apply_symlog_x(ax.flat, values)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {save_path}")


def plot_results(config, name, values, f_trunc, f_large, params, done,
                 f_fixed=None, ref_index=-1):
    values = np.asarray(values, dtype=float)
    ref_value = float(values[ref_index]) if ref_index is not None and ref_index >= 0 else None
    _plot_summary_core(
        name, values, f_trunc, f_large, params, done,
        config["truc_optimize"], config["truc_large"], config["do_large"],
        os.path.join(config["plot_dir"], "cz_sweep_summary.png"),
        transitions=compute_cz_transitions(config["folder_load"]),
        f_fixed=f_fixed, ref_value=ref_value,
    )


def plot_sweep_summary(result, save_path=None):
    """Regenerate the summary plot from a saved cz_sweep_*.npz (or its folder)."""
    result = Path(result)
    if result.is_dir():
        matches = sorted(result.glob("cz_sweep_*.npz"))
        if not matches:
            raise FileNotFoundError(f"No cz_sweep_*.npz found in {result}")
        npz_path = matches[0]
    else:
        npz_path = result

    d = np.load(npz_path, allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    if save_path is None:
        save_path = npz_path.with_name("cz_sweep_summary.png")
    values = np.asarray(d["sweep_param_values"], dtype=float)
    f_fixed = d["f_fixed"] if "f_fixed" in d.files else None
    ref_index = int(d["ref_index"]) if "ref_index" in d.files else -1
    ref_value = float(values[ref_index]) if ref_index >= 0 else None
    _plot_summary_core(
        str(d["sweep_param_name"]), values, d["f_trunc"], d["f_large"],
        d["drive_params"].tolist(), d["done"],
        cfg.get("truc_optimize", "?"), cfg.get("truc_large", "?"), cfg.get("do_large", True),
        save_path,
        transitions=compute_cz_transitions(cfg["folder_load"]),
        f_fixed=f_fixed, ref_value=ref_value,
    )


def plot_sweep_infidelity(result, save_path=None, show=False):
    """
    Plot log10 gate infidelity vs the swept parameter from a saved sweep result.

    f_trunc/f_large are already log10(1 - F), so they ARE the log infidelity and
    are plotted directly on the y axis.

    Parameters
    ----------
    result : str or Path
        Path to a cz_sweep_*.npz file, or the folder containing one.
    save_path : str or Path, optional
        Output PNG path. Defaults to <result_dir>/cz_sweep_infidelity.png.
    show : bool
        If True, display the figure instead of closing it.

    Returns
    -------
    (fig, ax)
    """
    result = Path(result)
    if result.is_dir():
        matches = sorted(result.glob("cz_sweep_*.npz"))
        if not matches:
            raise FileNotFoundError(f"No cz_sweep_*.npz found in {result}")
        npz_path = matches[0]
    else:
        npz_path = result

    d = np.load(npz_path, allow_pickle=True)
    name = str(d["sweep_param_name"])
    values = np.asarray(d["sweep_param_values"], dtype=float)
    done = np.asarray(d["done"], dtype=bool)
    f_trunc = np.asarray(d["f_trunc"], dtype=float)
    f_large = np.asarray(d["f_large"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(values[done], f_trunc[done], "o-", label="truncated (optimize space)")
    if np.any(np.isfinite(f_large[done])):
        ax.plot(values[done], f_large[done], "s--", alpha=0.8, label="large Hilbert space")
    ax.set_xlabel(name)
    ax.set_ylabel(r"$\log_{10}$ gate infidelity, $\log_{10}(1-F)$")
    ax.set_title(f"CZ infidelity vs {name}")
    _apply_symlog_x(ax, values)   # symlog x for zero-straddling sweeps (ng2/Φ2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is None:
        save_path = npz_path.with_name("cz_sweep_infidelity.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[PLOT] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


# ==============================================================
# MAIN
# ==============================================================
def run_sweep(config):
    name, values = hd.load_two_qubit_sweep_values(config["folder_load"])
    n = len(values)
    anchors, down_range, up_range = build_anchor_and_passes(n)

    seed = np.asarray(
        ut.load_drive_params_2q(True, folder=config["folder_pulse"])[config["seed_tg_idx"], :3],
        dtype=float,
    )

    print("=" * 60)
    print(f" CZ sweep optimization over '{name}' ({n} points)")
    print(f" anchors (from pulse file): {anchors}   seed={np.round(seed, 6)}")
    print(f" down pass: {down_range}")
    print(f" up pass:   {up_range}")
    print("=" * 60)

    f_trunc, f_large, params, W, done = load_resume(config, n)

    def do_point(i, x0):
        if done[i]:
            print(f"[SKIP] index {i} already done.")
            return params[i]
        ft, fl, p, w = optimize_cz_point(config, i, x0)
        f_trunc[i], f_large[i], params[i], W[i], done[i] = ft, fl, p, w, True
        save_results(config, name, values, f_trunc, f_large, params, W, done)
        return p

    # 1) anchors, each seeded from the pulse file
    for a in anchors:
        do_point(a, seed)

    def report_anchor_check():
        # Red-flag check: anchor optima should sit right on the seed.
        for a in anchors:
            shift = np.abs(np.array(params[a], dtype=float) - seed)
            print(f"[CHECK] anchor {a} ({name}={values[a]:.6f}): "
                  f"f_trunc={f_trunc[a]:.4f} f_large={f_large[a]:.4f} "
                  f"shift from seed [tg,amp,detune] = {np.round(shift, 6)}")

    if config.get("dry_run", False):
        report_anchor_check()
        print("[DRY RUN] anchors only; skipping outward passes. "
              "Set dry_run=False (or drop --dry-run) for the full sweep.")
        return

    # 2) expand outward; each point seeded from the neighbor toward center
    prev = params[min(anchors)]
    for i in tqdm(down_range, desc="down"):
        prev = do_point(i, prev)

    prev = params[max(anchors)]
    for i in tqdm(up_range, desc="up"):
        prev = do_point(i, prev)

    # 3) Final validation pass: hold the pulse calibrated at the reference point
    #    (nearest the sweep midpoint = nominal) FIXED and evaluate it at full
    #    truncation across the sweep -- the drift sensitivity, saved + plotted
    #    alongside the re-optimized curve.
    f_fixed = np.full(n, np.nan)
    ref_index = -1
    if config.get("do_fixed_eval", True):
        # Calibrate at the center anchor: index n//2 is the nominal parameter
        # value (exactly nominal even when the sweep is linspaced in capacitance,
        # where the EC2 value-midpoint would be slightly skewed), and it was the
        # point seeded from the nominal-optimized pulse.
        ref_index = n // 2
        fixed_pulse = params[ref_index]
        print(f"[fixed-eval] calibrating at {name}={values[ref_index]:.6f} "
              f"(pulse {np.round(fixed_pulse, 6)}), truc={config['truc_large']}")
        _, _, f_fixed = evaluate_fixed_pulse_sweep(
            config["folder_load"], fixed_pulse,
            truc=config["truc_large"], num_cpus=config.get("inner_num_cpus", 1))
        save_results(config, name, values, f_trunc, f_large, params, W, done,
                     f_fixed=f_fixed, ref_index=ref_index)

    plot_results(config, name, values, f_trunc, f_large, params, done,
                 f_fixed=f_fixed, ref_index=ref_index)
    report_anchor_check()
    print("CZ sweep optimization complete.")


def _find_result_npz(result_dir):
    result_dir = Path(result_dir)
    if result_dir.is_file():
        return result_dir
    matches = sorted(result_dir.glob("cz_sweep_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No cz_sweep_*.npz found in {result_dir}")
    return matches[0]


def reoptimize_points(folder_load, result_dir, indices, custom_config=None):
    """
    Re-optimize selected sweep points in an existing result, seeded from their
    previously-optimized pulse, and write the improved results back in place.

    Use this to escape a suspected local optimum by searching wider bounds
    and/or a larger population at specific points (e.g. widen detune to cover
    the full transition manifold). Each point's DE is seeded with x0 = its prior
    optimum, so with x0 in the population the result never regresses in f_trunc.

    Parameters
    ----------
    folder_load : str or Path
        Sweep-data folder holding two_qubit_sweep_data.npz.
    result_dir : str or Path
        Folder holding the cz_sweep_*.npz to update (or the npz path itself).
    indices : int or sequence of int
        Sweep index/indices to re-optimize.
    custom_config : dict, optional
        Config overrides for the re-optimization only (e.g. wider bounds, larger
        popsize). Common keys: popsize, tg_bound + tg_absolute=True, amp_bound,
        detune_bound, workers, inner_num_cpus. The saved file keeps the ORIGINAL
        config; only the results (params/fidelities) for `indices` are updated.

    Returns
    -------
    dict of {index: (old_f_trunc, new_f_trunc, old_f_large, new_f_large)}
    """
    if np.isscalar(indices):
        indices = [int(indices)]

    npz_path = _find_result_npz(result_dir)
    d = np.load(npz_path, allow_pickle=True)
    base_cfg = json.loads(str(d["config"]))
    base_cfg["result_file"] = str(npz_path)
    base_cfg["csv_file"] = str(npz_path.with_suffix(".csv"))
    base_cfg["plot_dir"] = str(npz_path.parent)

    name = str(d["sweep_param_name"])
    values = np.asarray(d["sweep_param_values"], dtype=float)
    f_trunc = np.asarray(d["f_trunc"], dtype=float).copy()
    f_large = np.asarray(d["f_large"], dtype=float).copy()
    params = [list(p) for p in d["drive_params"].tolist()]
    W = np.asarray(d["W_20_50"], dtype=float).copy()
    done = np.asarray(d["done"], dtype=bool).copy()

    # Config used for the DE search: base + this point's data folder + overrides.
    reopt_cfg = dict(base_cfg)
    reopt_cfg["folder_load"] = str(folder_load)
    if custom_config:
        reopt_cfg.update(custom_config)

    print(f"[reopt] {npz_path.name}: re-optimizing indices {list(indices)}")
    summary = {}
    for idx in indices:
        if not done[idx] or not np.all(np.isfinite(params[idx])):
            raise ValueError(f"index {idx} has no prior optimized pulse to seed from.")
        x0 = params[idx]
        old_ft, old_fl = float(f_trunc[idx]), float(f_large[idx])
        ft, fl, p, w = optimize_cz_point(reopt_cfg, idx, x0)
        summary[idx] = (old_ft, ft, old_fl, fl)
        # x0 is in the DE population, so f_trunc never regresses; keep the new run.
        f_trunc[idx], f_large[idx], params[idx], W[idx], done[idx] = ft, fl, p, w, True
        print(f"[reopt] index {idx} ({name}={values[idx]:.6f}): "
              f"f_trunc {old_ft:.4f} -> {ft:.4f} | f_large {old_fl:.4f} -> {fl:.4f}")
        print(f"[reopt]   params {np.round(x0, 6)} -> {np.round(p, 6)}")
        # Save after each point so a long re-opt is crash-safe (original config kept).
        save_results(base_cfg, name, values, f_trunc, f_large, params, W, done)

    plot_sweep_summary(npz_path)
    plot_sweep_infidelity(npz_path)
    return summary


def reoptimize_sweep(result_dir, out_dir=None, folder_load=None,
                     custom_config=None, redo_fixed_eval=True, in_place=False):
    """
    Re-optimize EVERY point of an existing sweep result, seeding each point from
    its OWN previously-optimized pulse (x0 = prior optimum for that index).

    This refreshes a completed sweep -- e.g. to re-score it under the corrected
    cz_fidelity_log_noise evaluator, or to polish points off a good warm start
    (better than the original neighbor-chain seed, so it can shake out the
    quantized-step artifacts). Because each point's DE keeps x0 in the population,
    f_trunc never regresses.

    By default the original result is preserved: the dir is copied to
    ``<result_dir>_reopt`` (or ``out_dir``) and the re-optimization runs there.
    Pass ``in_place=True`` to update the original instead.

    Parameters
    ----------
    result_dir : str or Path
        Existing cz_sweep_* result folder (or the npz path itself).
    out_dir : str or Path, optional
        Destination for the refreshed result. Default ``<result_dir>_reopt``.
        Ignored when ``in_place=True``.
    folder_load : str or Path, optional
        Sweep-data folder. Default: the ``folder_load`` saved in the result's
        config (so you usually need not pass it).
    custom_config : dict, optional
        Overrides for the re-optimization (e.g. workers, inner_num_cpus, wider
        bounds). The swept parameter, tg_absolute/tg_bound, truncations, etc. are
        otherwise inherited from the saved config.
    redo_fixed_eval : bool
        After re-optimizing, recompute the fixed-pulse drift curve calibrated at
        the (updated) reference point n//2. Recommended -- reoptimize_points drops
        the old f_fixed, so this restores a consistent fixed curve. Default True.
    in_place : bool
        Update ``result_dir`` directly instead of copying. Default False.

    Returns
    -------
    Path to the refreshed npz.
    """
    src_npz = _find_result_npz(result_dir)
    d = np.load(src_npz, allow_pickle=True)
    base_cfg = json.loads(str(d["config"]))
    name = str(d["sweep_param_name"])
    values = np.asarray(d["sweep_param_values"], dtype=float)
    n = len(values)
    folder_load = str(folder_load or base_cfg.get("folder_load"))

    if in_place:
        work_npz = src_npz
    else:
        if out_dir is None:
            out_dir = src_npz.parent.parent / (src_npz.parent.name + "_reopt")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        work_npz = out_dir / src_npz.name
        shutil.copy2(src_npz, work_npz)
        csv_src = src_npz.with_suffix(".csv")
        if csv_src.exists():
            shutil.copy2(csv_src, work_npz.with_suffix(".csv"))
        print(f"[reopt-sweep] copied {src_npz.parent.name} -> {out_dir.name}")

    # 1) Re-optimize all points, each warm-started from its own prior optimum.
    #    (reoptimize_points saves per-point and drops the stale f_fixed.)
    reoptimize_points(folder_load, work_npz, list(range(n)), custom_config=custom_config)

    # 2) Restore the fixed-pulse drift curve, calibrated at the refreshed ref pulse.
    if redo_fixed_eval:
        d2 = np.load(work_npz, allow_pickle=True)
        cfg = json.loads(str(d2["config"]))
        cfg.update(result_file=str(work_npz),
                   csv_file=str(work_npz.with_suffix(".csv")),
                   plot_dir=str(work_npz.parent),
                   folder_load=folder_load)
        if custom_config:
            cfg.update(custom_config)
        f_trunc = np.asarray(d2["f_trunc"], dtype=float)
        f_large = np.asarray(d2["f_large"], dtype=float)
        W = np.asarray(d2["W_20_50"], dtype=float)
        done = np.asarray(d2["done"], dtype=bool)
        params = [list(p) for p in d2["drive_params"].tolist()]
        ref_index = n // 2
        fixed_pulse = params[ref_index]
        print(f"[reopt-sweep] fixed-eval calibrated at {name}={values[ref_index]:.6f} "
              f"(pulse {np.round(fixed_pulse, 6)}), truc={cfg['truc_large']}")
        _, _, f_fixed = evaluate_fixed_pulse_sweep(
            folder_load, fixed_pulse,
            truc=cfg["truc_large"], num_cpus=cfg.get("inner_num_cpus", 1))
        save_results(cfg, name, values, f_trunc, f_large, params, W, done,
                     f_fixed=f_fixed, ref_index=ref_index)
        plot_sweep_summary(work_npz)
        plot_sweep_infidelity(work_npz)

    print(f"[reopt-sweep] done -> {work_npz}")
    return work_npz


def get_optimized_pulse(result_dir, value=None, index=None):
    """
    Pull the optimized [tg, amp, detune] pulse from a saved sweep result at a
    reference point (nearest sweep value, or explicit index).

    Returns (pulse_list, ref_value).
    """
    npz = _find_result_npz(result_dir)
    d = np.load(npz, allow_pickle=True)
    vals = np.asarray(d["sweep_param_values"], dtype=float)
    if (value is None) == (index is None):
        raise ValueError("Provide exactly one of value or index.")
    if index is None:
        index = int(np.argmin(np.abs(vals - value)))
    return list(d["drive_params"].tolist()[index]), float(vals[index])


def evaluate_fixed_pulse_sweep(folder_load, pulse, truc=1000, num_cpus=1):
    """
    Evaluate a FIXED CZ pulse at every point of a parameter sweep, without
    re-optimizing -- the drift/tolerance sensitivity: calibrate the gate once,
    then watch its fidelity degrade as the swept parameter moves.

    Parameters
    ----------
    folder_load : str or Path
        Sweep-data folder holding two_qubit_sweep_data.npz.
    pulse : sequence [tg, amp, detune]
        The fixed pulse to evaluate everywhere (e.g. the pulse optimized at the
        nominal value, from get_optimized_pulse).
    truc : int
        Hilbert-space truncation for the evaluation (1000 to match the large /
        converged fidelity).
    num_cpus : int
        Inner sesolve parallelism for a single (serial) evaluation. Safe to use
        >1 here since there is no DE worker pool to nest inside.

    Returns
    -------
    (name, values, f_log)  where f_log[i] = log10(1 - F) of the fixed pulse at
    sweep point i (NaN if the point's spectrum lacks a needed logical state).
    """
    pulse = list(pulse)
    # Minimal eval config; avoids get_sweep_config's dir/pulse-file side effects.
    cfg = dict(
        gate_type="CZ", folder_load=str(folder_load),
        use_truc_model=False, truc_model_name=None,
        truc_optimize=truc, max_step_ideal=1e-3, max_step_noisy=1e-3,
    )
    name, values = hd.load_two_qubit_sweep_values(folder_load)
    f_log = np.full(len(values), np.nan)

    global GLOBAL_system_data, GLOBAL_hamiltonians
    for i in tqdm(range(len(values)), desc=f"fixed-pulse eval (truc={truc})"):
        sd = load_cz_sweep_point(cfg, i)
        try:
            index_select = [sd["hspace_full"].index(x) for x in sd["hspace_select"]]
            logi_idx_select = [sd["hspace_select"].index(x) for x in sd["logi_state"]]
        except ValueError:
            continue  # a required logical/target state missing at this point
        H_drive_select, _ = ut.build_hamiltonian_2q(
            True, index_select, sd["eval_tot"], sd["eket_tot"], sd["drive_term"])
        GLOBAL_system_data = sd
        GLOBAL_hamiltonians = {"H_drive_select": H_drive_select,
                               "logi_idx_select": logi_idx_select}
        f_log[i] = cz_objective(pulse, num_cpus)
    return name, values, f_log


def plot_fixed_pulse_sweep(name, values, f_log, ref_value=None, reopt=None, save_path=None):
    """
    Plot log10 infidelity of a FIXED pulse vs the swept parameter (drift
    sensitivity). Optionally overlay a re-optimized curve reopt=(values, f_log)
    for comparison, and mark the calibration point ref_value.
    """
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(values, np.asarray(f_log, dtype=float), "o-", label="fixed pulse (calibrated once)")
    if reopt is not None:
        rv, rf = reopt
        ax.plot(np.asarray(rv, float), np.asarray(rf, float), "s--", alpha=0.7,
                label="re-optimized per point")
    if ref_value is not None:
        ax.axvline(ref_value, color="k", ls=":", lw=1, alpha=0.6,
                   label=f"calibration point ({name}={ref_value:.4f})")
    ax.set_xlabel(name)
    ax.set_ylabel(r"$\log_{10}$ gate infidelity, $\log_{10}(1-F)$")
    ax.set_title(f"CZ drift sensitivity vs {name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[PLOT] {save_path}")
    return fig, ax


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize a CZ gate over a parameter sweep.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Optimize only the central anchor(s) and report the red-flag check.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sweep indices already saved in the result file.")
    parser.add_argument("--seed-tg-idx", type=int, default=None,
                        help="Row of the pulse file to seed the anchor(s) from "
                             "(e.g. 71 for tg~91 ns, 162 for tg~182 ns).")
    parser.add_argument("--folder-load", type=str, default=None,
                        help="Folder holding two_qubit_sweep_data.npz to optimize over.")
    parser.add_argument("--n-cpu-inner", type=int, default=None,
                        help="Inner sesolve parallelism per objective (config inner_num_cpus). "
                             "Total procs ~= min(workers, population) * n_cpu_inner; keep <= cores.")
    parser.add_argument("--workers", type=int, default=None,
                        help="DE population workers (config workers). Total procs ~= "
                             "min(workers, population) * n_cpu_inner; keep <= cores. Use a "
                             "small value (e.g. 10) to cap footprint when sharing the box.")
    parser.add_argument("--tg-absolute", action="store_true",
                        help="Anchor the existing tg_bound window on the fixed seed-pulse "
                             "gate time for EVERY sweep point, instead of recentering on "
                             "each point's drifting warm-started seed. Keeps gate time "
                             "essentially pinned across the sweep (window width unchanged).")
    parser.add_argument("--plot", type=str, default=None,
                        help="Regenerate the infidelity + summary plots from an existing "
                             "result (cz_sweep_*.npz file or its folder) and exit; no optimization.")
    args = parser.parse_args()

    if args.plot is not None:
        plot_sweep_infidelity(args.plot)
        plot_sweep_summary(args.plot)
        return

    print(os.path.basename(__file__))
    ut.print_time()
    custom = {}
    if args.dry_run:
        custom["dry_run"] = True
    if args.resume:
        custom["resume"] = True
    if args.seed_tg_idx is not None:
        custom["seed_tg_idx"] = args.seed_tg_idx
    if args.folder_load is not None:
        custom["folder_load"] = args.folder_load
    if args.n_cpu_inner is not None:
        custom["inner_num_cpus"] = args.n_cpu_inner
    if args.workers is not None:
        custom["workers"] = args.workers
    if args.tg_absolute:
        custom["tg_absolute"] = True
    config = get_sweep_config(custom)
    print(json.dumps({k: str(v) for k, v in config.items() if k != "zp_yml"}, indent=1)[:2000])
    run_sweep(config)
    ut.print_time()


if __name__ == "__main__":
    main()
