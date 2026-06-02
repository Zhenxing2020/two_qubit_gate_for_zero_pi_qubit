"""Run CZ/CNOT population-transfer simulations with the current 2Q data loader."""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import qutip as qt
import scqubits.settings as settings
from scipy.ndimage import gaussian_filter

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)

import utils_2Q_gate_zp as ut
from run_2q_fidelity import (
    build_collapse_ops,
    load_system_data,
    print_config,
    select_gate_quantities,
)

settings.OVERLAP_THRESHOLD = 0.3


def get_config():
    """
    Return editable run-time configurations.

    Unit convention:
    - gate time and solver max_step are in ns
    - detunings are in GHz and converted to angular frequencies by 2*pi
    - loaded spectra and drive operators are angular frequencies in rad/ns
    """
    cfg = {}

    # ---------- Gate ----------
    cfg["cz_run"] = True  # True for CZ, False for CNOT
    cfg["calculate_ideal"] = True
    cfg["calculate_noise"] = False

    # ---------- Truncation ----------
    cfg["n_truc_list"] = [55]
    cfg["reduced_model"] = "charge_pick"  # "charge_pick", "lowest_state", or "graph_pick"
    cfg["graph_model_name"] = "n_theta_dress_charge_truc"

    # ---------- Solver ----------
    cfg["max_step_ideal"] = 1e-3
    cfg["max_step_noisy"] = 1e-3
    cfg["num_cpus"] = 4
    cfg["samples_per_ns"] = 1

    # ---------- Plot / save ----------
    cfg["save_txt"] = True
    cfg["save_plot"] = True
    cfg["show_plot"] = False
    cfg["plot_smooth_sigma"] = 3
    cfg["plot_ylim"] = (-0.02, 1.02)

    # ---------- Noise options ----------
    cfg["apply_decay"] = True
    cfg["apply_dephase"] = True
    cfg["decay_enlarge"] = 1
    cfg["filter_ratio"] = 0.3
    cfg["t1_tphi_other"] = 3  # us
    cfg["noise_dephase_path"] = "../data/flux_derivative_truc500/gamma_phi_2Q_truc500.npz"

    # ---------- Data ----------
    cfg["folder_load"] = "../data/Two_qubit_data_Sorted_Truc"

    # ---------- Pulse parameters ----------
    if cfg["cz_run"]:
        folder = "../figure/data/data_cz_fidelity_npz_select.txt"
        cfg["tg_list"] = [0] # [12]
        cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], :]
    else:
        folder = "../figure/data/data_cnot_fidelity_npz.txt"
        cfg["tg_list"] = [0] # [15]
        cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], :]

    cfg["output_dir"] = "data/population"
    cfg["run_datetime"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    return cfg


def select_truncation(n_truc, cfg, data):
    """Select the reduced Hilbert space using the same models as run_2q_fidelity.py."""
    if cfg["reduced_model"] == "lowest_state":
        hspace_select = data["hspace_full"][:n_truc]
        index_select = list(range(n_truc))
    elif cfg["reduced_model"] == "charge_pick":
        index_select = data["hspace_n_theta1"][:n_truc]
        hspace_select = [data["hspace_full"][i] for i in index_select]
    elif cfg["reduced_model"] == "graph_pick":
        index_select = ut.truc_model[cfg["graph_model_name"]][:n_truc]
        hspace_select = [data["hspace_full"][i] for i in index_select]
    else:
        raise ValueError("Unknown reduced_model type. Please choose from 'graph_pick', 'lowest_state', 'charge_pick'.")

    missing_logical = [state for state in data["logi_state"] if state not in hspace_select]
    if missing_logical:
        raise ValueError(f"Logical states are missing from selected hspace: {missing_logical}")

    return hspace_select, index_select


def build_pulse_args(cz_run, params, gate):
    """Convert one row of optimized pulse parameters into QuTiP time-dependent args."""
    if cz_run:
        tg, drive_amp, detune = params
        return {
            "drive_amp_A": drive_amp,
            "drive_freq_A": gate["W_20_50"] + 2 * np.pi * detune,
            "gate_time": tg,
        }

    tg, drive_amp_a, drive_amp_b, detune_a, detune_b = params
    return {
        "drive_amp_A": drive_amp_a,
        "drive_freq_A": gate["W_0_2"] + 2 * np.pi * detune_a,
        "drive_amp_B": drive_amp_b,
        "drive_freq_B": gate["W_1_2"] + 2 * np.pi * detune_b,
        "gate_time": tg,
    }


def population_states_by_initial(cz_run):
    """States saved in CSV for each logical initial state."""
    if cz_run:
        return {
            "0-0": ["0-0", "0-1", "1-1", "1-0"],
            "0-2": ["0-2", "1-2", "0-5"],
            "2-0": ["2-0", "5-0", "2-1", "5-1"],
            "2-2": ["2-2", "5-2", "2-5"],
        }

    return {
        "0-0": ["0-0", "2-0", "8-0", "1-0"],
        "0-2": ["0-2", "2-2", "8-2", "1-2"],
        "2-0": ["2-0", "0-0", "8-0", "5-0"],
        "2-2": ["2-2", "0-2", "8-2", "5-2"],
    }


def plot_states_by_initial(cz_run):
    """States plotted for each logical initial state."""
    logical_states = ["0-0", "0-2", "2-0", "2-2"]
    plot_states = {}
    for initial_state, states in population_states_by_initial(cz_run).items():
        plot_states[initial_state] = list(dict.fromkeys(logical_states + states))
    return plot_states


def make_output_dir(cfg, n_truc, params):
    """Create one folder per truncation, gate time, and run timestamp."""
    gate_name = "cz" if cfg["cz_run"] else "cnot"
    tg = float(params[0])
    tg_tag = f"{tg:.0f}" if tg.is_integer() else f"{tg:.3f}".replace(".", "p")
    output_dir = Path(cfg["output_dir"]) / (
        f"{gate_name}_truc={n_truc}_tg={tg_tag}_{cfg['run_datetime']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def qutip_options(cfg, noisy):
    """Population-mode options need store_states for later inspection."""
    max_step = cfg["max_step_noisy"] if noisy else cfg["max_step_ideal"]
    nsteps = int(1e5) # int(1 / max_step) if max_step else int(1e5)
    return qt.Options(
        max_step=max_step,
        nsteps=nsteps,
        store_states=True,
        num_cpus=cfg["num_cpus"],
    )


def run_population_for_params(H_drive_select, hspace_select, params, gate, cfg, c_op_list):
    """Run mesolve for all logical initial states for one pulse row."""
    tg = float(params[0])
    num_points = max(2, int(cfg["samples_per_ns"] * tg))
    tlist = np.linspace(0, tg, num=num_points)
    pulse_args = build_pulse_args(cfg["cz_run"], params, gate)

    n_hspace = len(hspace_select)
    basis_states = [qt.basis(n_hspace, i) for i in range(n_hspace)]
    projectors = [state * state.dag() for state in basis_states]
    options = qutip_options(cfg, noisy=bool(c_op_list))

    result = {}
    for initial_state in population_states_by_initial(cfg["cz_run"]).keys():
        initial_idx = hspace_select.index(initial_state)
        result[initial_state] = qt.mesolve(
            H_drive_select,
            basis_states[initial_idx],
            tlist,
            c_ops=c_op_list,
            e_ops=projectors,
            args=pulse_args,
            options=options,
        )

    return tlist, result


def save_population(result, tlist, hspace_select, params, n_truc, cfg, label, output_dir):
    """Save one CSV per logical initial state."""
    gate_name = "cz" if cfg["cz_run"] else "cnot"
    tg = float(params[0])
    saved_files = []

    if not cfg["save_txt"]:
        return saved_files

    for initial_state, state_interest in population_states_by_initial(cfg["cz_run"]).items():
        present_states = [state for state in state_interest if state in hspace_select]
        missing_states = [state for state in state_interest if state not in hspace_select]
        if missing_states:
            print(f"[warn] skip missing states for {initial_state}: {missing_states}")

        state_interest_idx = [hspace_select.index(state) for state in present_states]
        pop = np.array(result[initial_state].expect)
        pop_interest = pop[state_interest_idx, :]
        pop_other = np.delete(pop, state_interest_idx, axis=0).sum(axis=0)
        pop_save = np.vstack((tlist, pop_interest, pop_other)).T

        columns = ["tg"] + present_states + ["other"]
        initial_tag = initial_state.replace("-", "")
        output_file = output_dir / (
            f"{gate_name}_population_{label}_tg={tg:.0f}_{initial_tag}_n={n_truc}.txt"
        )
        pd.DataFrame(pop_save, columns=columns).to_csv(output_file, sep=",", index=False, header=True)
        saved_files.append(output_file)
        print(f"data saved in {output_file}")

    return saved_files


def save_population_plot(result, tlist, hspace_select, params, n_truc, cfg, label, output_dir):
    """Save a 2x2 population-transfer plot like the analysis notebook."""
    if not cfg["save_plot"] and not cfg["show_plot"]:
        return None

    gate_name = "cz" if cfg["cz_run"] else "cnot"
    tg = float(params[0])
    logi_state_one = [0, 2]
    states_to_plot = plot_states_by_initial(cfg["cz_run"])
    fig, ax = plt.subplots(figsize=(9, 9), nrows=2, ncols=2, sharex=True, sharey=True)

    for idx, state_i in enumerate(logi_state_one):
        for jdx, state_j in enumerate(logi_state_one):
            initial_state = f"{state_i}-{state_j}"
            for state_label in states_to_plot[initial_state]:
                if state_label not in hspace_select:
                    continue
                state_idx = hspace_select.index(state_label)
                pop = np.array(result[initial_state].expect[state_idx]).flatten()
                if cfg["plot_smooth_sigma"] and cfg["plot_smooth_sigma"] > 0:
                    pop = gaussian_filter(pop, cfg["plot_smooth_sigma"])
                ax[idx][jdx].plot(tlist, pop, label=state_label)

            ax[idx][jdx].set_title(f"initial |{initial_state}>")
            ax[idx][jdx].set_ylim(*cfg["plot_ylim"])
            ax[idx][jdx].grid()

        ax[1][idx].set_xlabel("Time (ns)")
        ax[idx][0].set_ylabel("Population")

    handles, labels = ax[1][1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
        ).get_frame().set_alpha(0.1)
    fig.suptitle(f"{gate_name.upper()} population, {label}, n={n_truc}, tg={tg:.3f} ns")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    output_file = output_dir / f"{gate_name}_population_{label}_tg={tg:.0f}_n={n_truc}.png"
    if cfg["save_plot"]:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"plot saved in {output_file}")
    if cfg["show_plot"]:
        plt.show()
    plt.close(fig)
    return output_file if cfg["save_plot"] else None


def run_one_truncation(n_truc, cfg, data, gate):
    """Build the reduced Hamiltonian and run population simulations."""
    hspace_select, index_select = select_truncation(n_truc, cfg, data)
    ut.print_fidelity(f"hspace_select (len={len(hspace_select)})", hspace_select, num_each_row=10)
    ut.print_fidelity(f"index_select (len={n_truc})", index_select, num_each_row=10)

    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        cfg["cz_run"],
        index_select,
        data["eval_tot"],
        data["eket_tot"],
        gate["drive_term"],
    )

    collapse_ops_by_label = {}
    if cfg["calculate_ideal"]:
        collapse_ops_by_label["ideal"] = []
    if cfg["calculate_noise"]:
        c_op_list = build_collapse_ops(cfg, data, eket_truc)
        print(f"np.shape(c_op_list) = {np.shape(c_op_list)}")
        collapse_ops_by_label[f"{cfg['t1_tphi_other']}us"] = c_op_list

    saved_files = []
    for label, c_op_list in collapse_ops_by_label.items():
        for params in np.atleast_2d(cfg["params"]):
            output_dir = make_output_dir(cfg, n_truc, params)
            print(f"\n----- {label}, tg = {params[0]:.6f} ns -----")
            print(f"output folder = {output_dir}")
            tlist, result = run_population_for_params(
                H_drive_select,
                hspace_select,
                params,
                gate,
                cfg,
                c_op_list,
            )
            for initial_state, res in result.items():
                print(f"\nTop population for initial {initial_state}:")
                ut.top_population(np.array(res.expect).sum(axis=1), hspace_select)
            saved_files.extend(save_population(result, tlist, hspace_select, params, n_truc, cfg, label, output_dir))
            plot_file = save_population_plot(result, tlist, hspace_select, params, n_truc, cfg, label, output_dir)
            if plot_file is not None:
                saved_files.append(plot_file)

    return saved_files


def main():
    print(os.path.basename(__file__))
    ut.print_time()

    cfg = get_config()

    cfg["cz_run"] = True  # True for CZ, False for CNOT
    cfg["calculate_ideal"] = True
    cfg["calculate_noise"] = False
    cfg["n_truc_list"] = [50]# [1000]
    cfg["reduced_model"] = "lowest_state"  # "charge_pick", "lowest_state", or "graph_pick"
    cfg["graph_model_name"] = "n_theta_dress_charge_truc"
    cfg["max_step_ideal"] = 1e-3
    cfg["max_step_noisy"] = 1e-3
    cfg["num_cpus"] = 4
    cfg["samples_per_ns"] = 1
    # ---------- Pulse parameters ----------
    if cfg["cz_run"]:
        folder = "../figure/data/data_cz_fidelity_npz_select.txt"
        cfg["tg_list"] = [0] # [12] # tg need to add another 20ns
        cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], :]
    else:
        folder = "../cnot/data/data_cnot_fidelity_npz.txt"
        cfg["tg_list"] = [0] # [14]
        cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], :]

    ut.print_fidelity("params", np.atleast_2d(cfg["params"]).tolist(), num_each_row=1)
    print_config(cfg)

    data = load_system_data(cfg)
    gate = select_gate_quantities(cfg, data)

    saved_files = []
    for n_truc in cfg["n_truc_list"]:
        print(f"\n===== n_truc = {n_truc} =====")
        saved_files.extend(run_one_truncation(n_truc, cfg, data, gate))
        ut.print_time()

    print("\n===== Saved population files =====")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()
