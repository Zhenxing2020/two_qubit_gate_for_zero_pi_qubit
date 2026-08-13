import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)

import scqubits.settings as settings
import utils_2Q_gate_zp as ut
from run_2q_fidelity import (
    build_collapse_ops,
    get_config,
    load_system_data,
    run_gate_fidelity,
    select_gate_quantities,
)

settings.OVERLAP_THRESHOLD = 0.3


DETUNE_GHZ_PER_MHZ = 1e-3


def add_datestamp(path):
    stamp = datetime.now().strftime("%m%d-%H%M")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def select_reduced_space(cfg, data, n_truc):
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
        raise ValueError(
            "Unknown reduced_model type. Please choose from "
            "'graph_pick', 'lowest_state', 'charge_pick'."
        )

    return hspace_select, index_select


def build_fidelity_context(cfg, data, gate, n_truc, option_ideal, option_noisy):
    hspace_select, index_select = select_reduced_space(cfg, data, n_truc)

    ut.print_fidelity(
        f"hspace_select (len={len(hspace_select)})",
        hspace_select,
        num_each_row=10,
    )
    ut.print_fidelity(
        f"index_select (len={n_truc})",
        index_select,
        num_each_row=10,
    )

    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        cfg["cz_run"],
        index_select,
        data["eval_tot"],
        data["eket_tot"],
        gate["drive_term"],
    )
    logi_idx_select = [hspace_select.index(i) for i in data["logi_state"]]
    c_op_list = build_collapse_ops(cfg, data, eket_truc) if cfg["calculate_noise"] else []

    return H_drive_select, c_op_list, logi_idx_select, option_ideal, option_noisy


def cnot_target_cols(target, col_1, col_2):
    if target == "all":
        return [col_1, col_2]
    if target == "1":
        return [col_1]
    if target == "2":
        return [col_2]
    raise ValueError("target must be 'all', '1', or '2'.")


def make_sweep_params(cfg):
    base_params = np.array(cfg["base_params"], dtype=float)
    n_points = cfg["sweep_points"]
    tg_offsets = np.linspace(-cfg["tg_span"], cfg["tg_span"], n_points)
    amp_offsets = np.linspace(-cfg["amp_span"], cfg["amp_span"], n_points)
    detune_offsets_mhz = np.linspace(
        -cfg["detune_span_mhz"],
        cfg["detune_span_mhz"],
        n_points,
    )

    if cfg["cz_run"]:
        amp_cols = [1]
        detune_cols = [2]
    else:
        amp_cols = cnot_target_cols(cfg["amp_target"], 1, 2)
        detune_cols = cnot_target_cols(cfg["detune_target"], 3, 4)

    tg_params = np.repeat(base_params[None, :], n_points, axis=0)
    tg_params[:, 0] += tg_offsets

    amp_params = np.repeat(base_params[None, :], n_points, axis=0)
    for col in amp_cols:
        amp_params[:, col] += amp_offsets

    detune_params = np.repeat(base_params[None, :], n_points, axis=0)
    for col in detune_cols:
        detune_params[:, col] += detune_offsets_mhz * DETUNE_GHZ_PER_MHZ

    return (
        base_params,
        tg_offsets,
        tg_params,
        amp_offsets,
        amp_params,
        detune_offsets_mhz,
        detune_params,
    )


def print_sweep_values(cfg, tg_params, amp_params, detune_params):
    if cfg["cz_run"]:
        amp_cols = [1]
        detune_cols = [2]
    else:
        amp_cols = cnot_target_cols(cfg["amp_target"], 1, 2)
        detune_cols = cnot_target_cols(cfg["detune_target"], 3, 4)

    print("\n" + "=" * 60)
    print(" Sweep Values ")
    print("=" * 60)
    ut.print_fidelity("tg_sweep_ns", tg_params[:, 0], num_digits=8)

    for col in amp_cols:
        ut.print_fidelity(f"amp_sweep_col{col}", amp_params[:, col], num_digits=8)

    for col in detune_cols:
        ut.print_fidelity(
            f"detune_sweep_col{col}_GHz",
            detune_params[:, col],
            num_digits=8,
        )

    print("=" * 60 + "\n")


def evaluate_sweep(params, cfg, H_drive_select, gate, c_op_list,
                   logi_idx_select, option_ideal, option_noisy):
    cfg_run = cfg.copy()
    cfg_run["params"] = np.atleast_2d(params)
    cfg_run["n_job"] = min(cfg["n_job"], len(cfg_run["params"]))
    log_errors = run_gate_fidelity(
        cfg_run,
        H_drive_select,
        gate,
        c_op_list,
        logi_idx_select,
        option_ideal,
        option_noisy,
    )
    return np.array(log_errors, dtype=float)


def save_sensitivity_plot(cfg, base_params, tg_offsets, tg_log_errors,
                          amp_offsets, amp_log_errors, detune_offsets_mhz,
                          detune_log_errors):
    output_png = add_datestamp(Path(cfg["output_png"]))
    output_png.parent.mkdir(parents=True, exist_ok=True)

    tg_errors = np.power(10.0, tg_log_errors)
    amp_errors = np.power(10.0, amp_log_errors)
    detune_errors = np.power(10.0, detune_log_errors)

    fig, axes = plt.subplots(ncols=3, figsize=(14, 4), constrained_layout=True)

    axes[0].plot(tg_offsets, tg_errors, marker="o")
    axes[0].set_xlabel("Gate-time offset (ns)")
    axes[0].set_ylabel("Gate error (1 - F)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(detune_offsets_mhz, detune_errors, marker="o")
    axes[1].set_xlabel("Detuning offset (MHz)")
    axes[1].set_ylabel("Gate error (1 - F)")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].plot(amp_offsets, amp_errors, marker="o")
    axes[2].set_xlabel("Drive-amplitude offset")
    axes[2].set_ylabel("Gate error (1 - F)")
    axes[2].set_yscale("log")
    axes[2].grid(True, which="both", alpha=0.3)

    gate_name = "CZ" if cfg["cz_run"] else "CNOT"
    run_mode = "noise" if cfg["calculate_noise"] else "ideal"
    noise_label = (
        f", T1/Tphi={cfg['t1_tphi_other']:g} us" if cfg["calculate_noise"] else ""
    )
    fig.suptitle(
        f"{gate_name} sensitivity ({run_mode}), "
        f"tg={base_params[0]:.0f} ns, n_truc={cfg['n_truc']}{noise_label}"
    )
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    # data_path = output_png.with_suffix(".npz")
    # np.savez(
    #     data_path,
    #     cz_run=cfg["cz_run"],
    #     base_params=base_params,
    #     tg_offsets=tg_offsets,
    #     tg_log_errors=tg_log_errors,
    #     tg_errors=tg_errors,
    #     amp_offsets=amp_offsets,
    #     amp_log_errors=amp_log_errors,
    #     amp_errors=amp_errors,
    #     detune_offsets_mhz=detune_offsets_mhz,
    #     detune_log_errors=detune_log_errors,
    #     detune_errors=detune_errors,
    # )
    # print(f"Saved data: {data_path}")

    print(f"Saved PNG: {output_png}")


def print_config(cfg):
    print("\n" + "=" * 60)
    print(" Sensitivity Configuration ")
    print("=" * 60)
    for key, value in cfg.items():
        print(f"{key} = {value}")
    print("=" * 60 + "\n")


def main():
    print(os.path.basename(__file__))
    ut.print_time()

    ################################################################
    cfg = get_config()
    cfg["cz_run"] = True  # True for CZ, False for CNOT
    cfg.pop("n_truc_list", None)

    cfg["n_truc"] = 300
    cfg["reduced_model"] = "charge_pick"  # 'graph_pick', 'lowest_state', 'charge_pick'
    cfg["calculate_ideal"] = True
    cfg["calculate_noise"] = False
    cfg["apply_decay"] = True
    cfg["apply_dephase"] = True
    cfg["decay_enlarge"] = 1
    cfg["filter_ratio"] = 0.3
    cfg["t1_tphi_other"] = 30  # us
    cfg["noise_dephase_path"] = "../data/flux_derivative_truc500/gamma_phi_2Q_truc500.npz"
    cfg["max_step_ideal"] = 1e-3
    cfg["max_step_noisy"] = 1e-3
    cfg["num_cpus_ideal"] = 4
    cfg["num_cpus_noisy"] = 16
    cfg["n_job"] = 1

    cfg["sweep_points"] = 91
    cfg["tg_span"] = 10.0  # ns
    cfg["amp_span"] = 0.003  # dimensionless drive-amplitude offset
    cfg["detune_span_mhz"] = 3.0  # sweep +/- 2 MHz; fidelity code applies 2*pi

    # For CNOT, choose which column to sweep: 'all', '1', or '2'.
    cfg["amp_target"] = "all"
    cfg["detune_target"] = "all"

    if cfg["cz_run"]:
        folder = "../figure/data/data_cz_fidelity_npz_select.txt"
        tg_idx = 27 # 12-> tg=92ns, 27-> tg=182ns # 0
        cfg["base_params"] = ut.load_drive_params_2q(
            cfg["cz_run"],
            folder=folder,
        )[tg_idx]
        cfg["output_png"] = f"data/sensitivity/cz_sensitivity_tg{tg_idx}_amp{cfg['amp_span']}_det{cfg['detune_span_mhz']}.png"
    else:
        folder = "../figure/data/data_cnot_fidelity_npz.txt"
        tg_idx = 0
        cfg["base_params"] = ut.load_drive_params_2q(
            cfg["cz_run"],
            folder=folder,
        )[tg_idx]
        cfg["output_png"] = "data/sensitivity/cnot_sensitivity.png"

    # To use explicit pulse parameters instead of loading a row, uncomment one:
    # cfg["base_params"] = np.array([20.0396, 0.042986, 0.024999])  # CZ
    # cfg["base_params"] = np.array([29.994376, 0.099205, 0.057109, -0.028523, -0.027362])  # CNOT
    ################################################################

    if cfg["calculate_ideal"] == cfg["calculate_noise"]:
        raise ValueError("Set exactly one of calculate_ideal/calculate_noise to True.")
    if cfg["sweep_points"] < 2:
        raise ValueError("sweep_points must be at least 2.")

    print_config(cfg)
    ut.print_fidelity("base_params", np.atleast_2d(cfg["base_params"]).tolist(), num_each_row=1)

    data = load_system_data(cfg)
    gate = select_gate_quantities(cfg, data)
    option_ideal, option_noisy = ut.get_qutip_options(
        cfg["max_step_ideal"],
        cfg["max_step_noisy"],
    )

    (
        H_drive_select,
        c_op_list,
        logi_idx_select,
        option_ideal,
        option_noisy,
    ) = build_fidelity_context(
        cfg,
        data,
        gate,
        cfg["n_truc"],
        option_ideal,
        option_noisy,
    )

    (
        base_params,
        tg_offsets,
        tg_params,
        amp_offsets,
        amp_params,
        detune_offsets_mhz,
        detune_params,
    ) = make_sweep_params(cfg)
    print_sweep_values(cfg, tg_params, amp_params, detune_params)

    print("Calculating gate-time sweep...")
    tg_log_errors = evaluate_sweep(
        tg_params,
        cfg,
        H_drive_select,
        gate,
        c_op_list,
        logi_idx_select,
        option_ideal,
        option_noisy,
    )
    ut.print_fidelity("tg_log_errors", tg_log_errors, num_digits=8)

    print("Calculating amplitude sweep...")
    amp_log_errors = evaluate_sweep(
        amp_params,
        cfg,
        H_drive_select,
        gate,
        c_op_list,
        logi_idx_select,
        option_ideal,
        option_noisy,
    )
    ut.print_fidelity("amp_log_errors", amp_log_errors, num_digits=8)

    print("Calculating detuning sweep...")
    detune_log_errors = evaluate_sweep(
        detune_params,
        cfg,
        H_drive_select,
        gate,
        c_op_list,
        logi_idx_select,
        option_ideal,
        option_noisy,
    )
    ut.print_fidelity("detune_log_errors", detune_log_errors, num_digits=8)

    save_sensitivity_plot(
        cfg,
        base_params,
        tg_offsets,
        tg_log_errors,
        amp_offsets,
        amp_log_errors,
        detune_offsets_mhz,
        detune_log_errors,
    )
    ut.print_time()


if __name__ == "__main__":
    main()
