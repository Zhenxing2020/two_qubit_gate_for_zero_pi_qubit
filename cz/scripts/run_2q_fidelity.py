import os
import sys
import argparse
import re
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)

import scqubits.settings as settings
import utils_2Q_gate_zp as ut
import ham_data as hd

settings.OVERLAP_THRESHOLD = 0.3


class MemoryMonitor:
    """
    Track peak memory for this Python process and all child processes.

    tracemalloc only sees Python allocations in the current process, while this
    script spends most memory in NumPy/QuTiP native allocations and joblib
    workers. RSS/PSS from psutil is closer to what the OS reports.
    """
    def __init__(self, interval=0.2):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.running = False
        self.thread = None
        self.peak_rss = 0
        self.peak_pss = 0

    def _get_process_tree_memory(self):
        procs = [self.proc] + self.proc.children(recursive=True)
        total_rss = 0
        total_pss = 0

        for proc in procs:
            try:
                mem = proc.memory_full_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            total_rss += mem.rss
            total_pss += getattr(mem, "pss", mem.rss)

        return total_rss, total_pss

    def _watch(self):
        while self.running:
            rss, pss = self._get_process_tree_memory()
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_pss = max(self.peak_pss, pss)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

        rss, pss = self._get_process_tree_memory()
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_pss = max(self.peak_pss, pss)

    def peak_gb(self):
        return {
            "rss": self.peak_rss / 1024**3,
            "pss": self.peak_pss / 1024**3,
        }

    def print_summary(self, label):
        peak = self.peak_gb()
        print(f"[{label}] Peak RSS: {peak['rss']:.2f} GB")
        print(f"[{label}] Peak PSS: {peak['pss']:.2f} GB")


# ============================================================
# Configuration
# ============================================================
def get_config():
    """
    Return a dictionary containing all run-time configurations.
    """
    cfg = {}

    # ---------- General ----------
    cfg["cz_run"] = False # True for CZ, False for CNOT
    
    # ---------- Truncation ----------
    # cfg["n_truc_list"] = np.arange(50, 201, step=10) # [200] #    
    cfg["n_truc_list"] = np.array([55, 240]) # np.arange(300, 901, step=100) # [200] #    
    # cfg["use_truc_model"] = True
    cfg["reduced_model"] = 'charge_pick' # 'graph_pick', 'lowest_state', 'charge_pick'

    # cfg["n_truc_list"] = [200, 400, 600, 800, 1000, 2000, 3000] # np.arange(500, 3001, 500) # [200] #    
    # cfg["use_truc_model"] = False
    
    # cfg["graph_model_name"] = "n_theta_dress_charge_truc"
    cfg["calculate_ideal"] = True
    cfg["calculate_noise"] = False
    cfg["max_step_ideal"] = 1e-3
    cfg["max_step_noisy"] = 1e-3
    
    # ---------- Noise options ----------
    cfg["apply_decay"] = True
    cfg["apply_dephase"] = True
    cfg["decay_enlarge"] = 1
    cfg["filter_ratio"] = 0.3
    cfg["t1_tphi_other"] = 170  # us
    cfg["noise_dephase_path"] = "../data/flux_derivative_truc500/gamma_phi_2Q_truc500.npz"

    # ---------- Parallel ----------
    cfg["tg_para"] = False
    cfg["num_cpus_ideal"] = 4
    cfg["num_cpus_noisy"] = 16

    # ---------- Data ----------
    cfg["folder_load"] = '../data/Two_qubit_data_Sorted_Truc'
    # cfg["folder_load"] = "../../data/_truc_3000"

    # ---------- Pulse parameters ----------
    # If Load pulse parameters from CSV, uncomment 8 lines below
    
    # if cfg["cz_run"]:
    #     folder = 'data/npz/cz_pulse_neighbor.txt'
    #     cfg["tg_list"] = np.arange(180)[0::6]
    #     cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], ]  # [1::4,] 
    # else:
    #     folder = '../cnot/data/cnot_fidelity_npz.txt'
    #     cfg["tg_list"] = np.arange(33)#[:1] # [0::6]
    #     cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], ]  # [1::4,]
    


    #     cfg["params"] = np.array([
    # ### CNOT From charge-truc=400        
    # [29.994376, 0.099205, 0.057109, -0.028523, -0.027362] ,
    # [40.003004, 0.096609, 0.049579, -0.028325, -0.026694] ,
    # [50.009627, 0.086208, 0.045334, -0.026182, -0.024551] ,
    # [59.995626, 0.077544, 0.041816, -0.024491, -0.02316] ,
    # [70.001584, 0.070142, 0.037217, -0.020717, -0.019962] ,
    # [79.999318, 0.064654, 0.035147, -0.019493, -0.019465] ,
    # [90.004152, 0.05837, 0.033131, -0.016679, -0.016419] ,
    # [99.993121, 0.054087, 0.03088, -0.01407, -0.013906] ,
    # [110.006914, 0.04942, 0.029523, -0.012467, -0.012274] ,
    # [120.002847, 0.046972, 0.028541, -0.012215, -0.011723] ,
    # [130.008905, 0.044967, 0.027755, -0.011844, -0.011358] ,
    # [140.001622, 0.043144, 0.027082, -0.011653, -0.011332] ,
    # [150.007858, 0.039792, 0.025436, -0.010327, -0.00999] ,
    # [159.993478, 0.037136, 0.023904, -0.00912, -0.008799] ,
    # [169.994539, 0.035013, 0.022727, -0.008229, -0.007889] ,
    # [179.994327, 0.032879, 0.021406, -0.007326, -0.006974] ,
    # [190.005655, 0.031012, 0.020307, -0.006613, -0.006325] ,
    # [199.99683, 0.029348, 0.019253, -0.005939, -0.005642] ,
    # [209.999963, 0.027811, 0.018316, -0.005361, -0.005105] ,
    # [220.001258, 0.026491, 0.017488, -0.004918, -0.004691] ,
    # [229.990815, 0.025142, 0.016545, -0.00441, -0.004176] ,
    # [240.00977, 0.024099, 0.015795, -0.003997, -0.003807] ,
    # [249.998675, 0.02298, 0.015082, -0.003623, -0.003456] ,
    # [260.001744, 0.021969, 0.014451, -0.003319, -0.003154] ,
    # [269.992401, 0.021155, 0.013872, -0.003069, -0.002914] ,
    # [279.998122, 0.020361, 0.013365, -0.00284, -0.002704] ,

    # [289.999552, 0.019624, 0.012863, -0.002621, -0.0025] ,
    # [300.007623, 0.018925, 0.012408, -0.002449, -0.002338] ,
    # [309.995594, 0.018283, 0.011979, -0.002273, -0.002166] ,
    # [320.004107, 0.017647, 0.01159, -0.002118, -0.002025] ,
    # [330.006586, 0.017119, 0.011225, -0.002, -0.001905] ,
    # [340.000951, 0.016589, 0.010887, -0.001872, -0.001781] ,
    # [350.00857, 0.016105, 0.010552, -0.001759, -0.001678] ,
    #     ])

    cfg["n_job"] = 10 # len(cfg["tg_list"])    
    return cfg

# ============================================================
# Load system data
# ============================================================
def load_system_data(cfg):
    """
    Load two-qubit Hilbert space and operators.
    """
    data = {}
    (
        data["hspace_full"],
        data["eket_tot"],
        data["eval_tot"],
        data["n_theta0"],
        data["n_theta1"],
        data["n_theta0_dress"],
        data["n_theta1_dress"],
        data["hspace_0"],
        data["hspace_1"],
        data["hspace_n_theta1"],
        data["hspace_n_theta2"],
        data["logi_state"],
    ) = hd.load_two_qubit_data(cfg["folder_load"], return_full=False)

    data["dim_0"] = len(data["hspace_0"])
    data["dim_1"] = len(data["hspace_1"])
    return data


# ============================================================
# Gate-dependent quantities
# ============================================================
def select_gate_quantities(cfg, data):
    """
    Return gate-dependent quantities.
    """
    gate = {}

    if cfg["cz_run"]:
        gate["drive_term"] = data["n_theta1_dress"]
        gate["W_20_50"] = (
            data["eval_tot"][data["hspace_full"].index("5-0")]
            - data["eval_tot"][data["hspace_full"].index("2-0")]
        )
    else:
        gate["drive_term"] = data["n_theta0_dress"]
        gate["mid_state"] = "8-2"

        idx0 = data["hspace_full"].index("0-2")
        idx1 = data["hspace_full"].index("2-2")
        idxm = data["hspace_full"].index(gate["mid_state"])

        gate["W_0_2"] = data["eval_tot"][idxm] - data["eval_tot"][idx0]
        gate["W_1_2"] = data["eval_tot"][idxm] - data["eval_tot"][idx1]

    return gate


def run_gate_fidelity(
    cfg,
    H_drive_select,
    gate,
    c_op_list,
    logi_idx_select,
    option_ideal,
    option_noisy,
):
    """
    Run CZ or CNOT fidelity using the selected collapse operators.
    """
    num_cpus = cfg["num_cpus_noisy"] if len(c_op_list) else cfg["num_cpus_ideal"]

    if cfg["cz_run"]:
        args_common = [
            H_drive_select,
            gate["W_20_50"],
            num_cpus,
            c_op_list,
            logi_idx_select,
            option_ideal,
            option_noisy,
            cfg["use_qt_fidelity"],
        ]
        return Parallel(n_jobs=cfg["n_job"])(
            delayed(ut.cz_fidelity_log_noise)(p, *args_common)
            for p in cfg["params"]
        )

    args_common = [
        H_drive_select,
        gate["W_0_2"],
        gate["W_1_2"],
        num_cpus,
        c_op_list,
        logi_idx_select,
        gate["mid_state"],
        option_ideal,
        option_noisy,
        cfg["use_qt_fidelity"],
    ]
    return Parallel(n_jobs=cfg["n_job"])(
        delayed(ut.cnot_fidelity_log_noise)(p, *args_common)
        for p in cfg["params"]
    )


def build_collapse_ops(cfg, data, eket_truc):
    """
    Build two-qubit collapse operators for noisy CZ/CNOT simulations.
    """
    noise_dephase_path = Path(cfg["noise_dephase_path"])
    noise_data = np.load(noise_dephase_path)
    gamma_dephase_02_q0 = np.abs(noise_data["q0"] / cfg["t1_tphi_other"])
    gamma_dephase_02_q1 = np.abs(noise_data["q1"] / cfg["t1_tphi_other"])
    print(f"noise source = {noise_dephase_path}")

    n_theta0 = data["n_theta0"] / (2 * np.pi)
    n_theta1 = data["n_theta1"] / (2 * np.pi)

    Gamma = 1 / 1e3 / cfg["t1_tphi_other"]
    Gamma_decay_q0 = Gamma / (n_theta0[4, 8] ** 2)
    Gamma_decay_q1 = Gamma / (n_theta1[4, 8] ** 2)

    transition_a, n_theta0_trunc = ut.get_transitions_for_collapse(
        data["hspace_0"],
        n_theta0,
        filter_ratio=cfg["filter_ratio"],
    )
    transition_b, n_theta1_trunc = ut.get_transitions_for_collapse(
        data["hspace_1"],
        n_theta1,
        filter_ratio=cfg["filter_ratio"],
    )

    c_op_list = ut.construct_c_ops_2q(
        data["dim_0"],
        data["dim_1"],
        n_theta0_trunc,
        n_theta1_trunc,
        gamma_dephase_02_q0,
        gamma_dephase_02_q1,
        eket_truc,
        Gamma_decay_q0,
        Gamma_decay_q1,
        transition_a,
        transition_b,
        cfg["apply_decay"],
        cfg["apply_dephase"],
        cfg["decay_enlarge"],
    )
    # print(f"len(c_op_list) = {len(c_op_list)}")
    return c_op_list


# ============================================================
# One truncation run
# ============================================================
def run_one_truncation(n_truc, cfg, data, gate, option_ideal, option_noisy):
    """
    Run fidelity calculation for a given truncation size.
    """
    ### utils.py stores states
    # if cfg["use_truc_model"]:
    #     hspace_select = ut.truc_model[cfg["truc_model_name"]][:n_truc]
    # else:
    #     hspace_select = data["hspace_full"][:n_truc]
    # index_select = [data["hspace_full"].index(i) for i in hspace_select]
    
    ### utils.py stores state indexes

    if cfg["reduced_model"] == 'lowest_state':
        hspace_select = data["hspace_full"][:n_truc]
        index_select = list(range(n_truc))
    elif cfg["reduced_model"] == 'charge_pick':
        index_select = data["hspace_n_theta1"][:n_truc]
        hspace_select = [data["hspace_full"][i] for i in index_select]
    elif cfg["reduced_model"] == 'graph_pick':
        index_select = ut.truc_model[cfg["graph_model_name"]][:n_truc]
        hspace_select = [data["hspace_full"][i] for i in index_select]        
    else:
        raise ValueError("Unknown reduced_model type. Please choose from 'graph_pick', 'lowest_state', 'charge_pick'.")
    
    ut.print_fidelity(f'hspace_select (len={len(hspace_select)})', 
                    hspace_select, num_each_row=10)
    ut.print_fidelity(f'index_select (len={n_truc})', index_select, num_each_row=10)

    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        cfg["cz_run"],
        index_select,
        data["eval_tot"],
        data["eket_tot"],
        gate["drive_term"],
    )

    logi_idx_select = [hspace_select.index(i) for i in data["logi_state"]]

    results = {}

    if cfg["calculate_ideal"]:
        results["ideal"] = run_gate_fidelity(
            cfg,
            H_drive_select,
            gate,
            [],
            logi_idx_select,
            option_ideal,
            option_noisy,
        )
        ut.print_fidelity(f"f_ideal_{n_truc}", results["ideal"], num_digits=8)

    if cfg["calculate_noise"]:
        c_op_list = build_collapse_ops(cfg, data, eket_truc)
        print(f'np.shape(c_op_list) = {np.shape(c_op_list)}')     
        results["noise"] = run_gate_fidelity(
            cfg,
            H_drive_select,
            gate,
            c_op_list,
            logi_idx_select,
            option_ideal,
            option_noisy,
        )

    return results


def print_config(cfg, max_array_rows=50):
    """
    Pretty-print configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    max_array_rows : int
        Max number of rows to print for numpy arrays.
    """
    print("\n" + "=" * 60)
    print(" Configuration ")
    print("=" * 60)

    def _print_value(v):
        if isinstance(v, np.ndarray):
            print(f"    type: np.ndarray, shape={v.shape}")
            if v.ndim == 1:
                print(f"    preview: {v[:max_array_rows].tolist()}")
            elif v.ndim == 2:
                preview = v[:max_array_rows]
                for row in preview:
                    print(f"      {row.tolist()}")
                if v.shape[0] > max_array_rows:
                    print("      ...")
        else:
            print(f"    {v}")

    for key in cfg.keys():
    # for key in sorted(cfg.keys()):
        print(f"\n[{key}]")
        _print_value(cfg[key])

    print("\n" + "=" * 60 + "\n")


# ============================================================
# Main
# ============================================================
def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_bool(text):
    value = text.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _load_cz_pulse_params(folder):
    """Load either a CSV pulse table or a printed ``np.array`` table."""
    try:
        table = pd.read_csv(folder)
        return table[["tg", "drive_amp", "detune"]].to_numpy()
    except (pd.errors.ParserError, KeyError):
        rows = []
        with open(folder, encoding="utf-8") as stream:
            for line in stream:
                match = re.match(
                    r"\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,"
                    r"\s*([-+0-9.eE]+)\s*\]",
                    line,
                )
                if match:
                    rows.append([float(value) for value in match.groups()])
                elif rows:
                    break
        if not rows:
            raise ValueError(f"No [tg, drive_amp, detune] rows found in {folder}")
        return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate", choices=("cz", "cnot"), default="cz",
        help="two-qubit gate to simulate",
    )
    parser.add_argument("--n-truc-list", type=_parse_int_list, default=None)
    parser.add_argument("--tg-list", type=_parse_int_list, default=None)
    parser.add_argument("--calculate-ideal", type=_parse_bool, default=None)
    parser.add_argument("--calculate-noise", type=_parse_bool, default=None)
    parser.add_argument("--use-qt-fidelity", type=_parse_bool, default=None)
    parser.add_argument("--t1", type=float, default=None, help="T1=Tphi in us")
    parser.add_argument("--pulse-file", default=None)
    parser.add_argument("--parallel-jobs", type=int, default=None)
    parser.add_argument("--tg-parallel", type=_parse_bool, default=None)
    cli_args = parser.parse_args()

    print(os.path.basename(__file__))
    ut.print_time()

    ################################################################
    cfg = get_config()
    cfg["cz_run"] = cli_args.gate == "cz"
    cfg["t1_tphi_other"] = 170  # us
    cfg["n_truc_list"] = [200] # np.arange(200, 401, step=10) # [100] # [60, 90, 120] # [70, 100, 130] # [80, 110, 140] #
    #  [60, 90, 120] # np.arange(60, 241, step=20).tolist() + [500,1000]   
    cfg["reduced_model"] = 'charge_pick' # 'graph_pick', 'lowest_state', 'charge_pick'
    cfg["calculate_ideal"] = True
    cfg["calculate_noise"] = False    
    cfg["apply_decay"] = True
    cfg["apply_dephase"] = True
    cfg["decay_enlarge"] = 1
    cfg["filter_ratio"] = 0.3 # default is 0.3    
    cfg["tg_para"] = False
    # Trace-decreasing fidelity is the project default because it includes
    # leakage/survival loss. Pass --use-qt-fidelity true for QuTiP's formula.
    cfg["use_qt_fidelity"] = False

    if cli_args.n_truc_list is not None:
        cfg["n_truc_list"] = cli_args.n_truc_list
    if cli_args.calculate_ideal is not None:
        cfg["calculate_ideal"] = cli_args.calculate_ideal
    if cli_args.calculate_noise is not None:
        cfg["calculate_noise"] = cli_args.calculate_noise
    if cli_args.use_qt_fidelity is not None:
        cfg["use_qt_fidelity"] = cli_args.use_qt_fidelity
    if cli_args.t1 is not None:
        cfg["t1_tphi_other"] = cli_args.t1
    if cli_args.parallel_jobs is not None:
        cfg["n_job"] = cli_args.parallel_jobs
    if cli_args.tg_parallel is not None:
        cfg["tg_para"] = cli_args.tg_parallel

    if cfg["cz_run"]:
        folder = cli_args.pulse_file or '../figure/data/data_cz_fidelity_npz_select.txt'
        # folder = 'data/npz/cz_pulse_neighbor.txt'
        pulse_params_all = _load_cz_pulse_params(folder)
        cfg["tg_list"] = np.arange(len(pulse_params_all))
        if cli_args.tg_list is not None:
            cfg["tg_list"] = cli_args.tg_list
        # [0, 36, 72, 108, 144] # [72] # [72,179] # [0, 45, 90, 135, 179] 
        # [0,  30,  60,  90, 120, 150] [0, 45, 72, 90, 135, 179]
        # np.arange(180)[0::6].tolist() +[179] #np.array([135, 179]) #np.arange(180)[0::6]
        cfg["params"] = pulse_params_all[cfg["tg_list"], ]
    else:
        folder = cli_args.pulse_file or '../figure/data/data_cnot_fidelity_npz.txt'
        pulse_params_all = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)
        cfg["tg_list"] = np.arange(len(pulse_params_all))
        if cli_args.tg_list is not None:
            cfg["tg_list"] = cli_args.tg_list
        cfg["params"] = pulse_params_all[cfg["tg_list"], ]
    ################################################################

    ut.print_fidelity("params", cfg["params"].tolist(), num_each_row=1)
    print_config(cfg)

    data = load_system_data(cfg)
    gate = select_gate_quantities(cfg, data)

    option_ideal, option_noisy = ut.get_qutip_options(
        cfg["max_step_ideal"], cfg["max_step_noisy"]
    )

    f_ideal_all = []
    f_noise_all = []
    memory_stats = []
    tg_list_all = list(np.atleast_1d(cfg["tg_list"]))
    params_all = np.atleast_2d(cfg["params"])

    if len(tg_list_all) != len(params_all):
        raise ValueError(
            f"tg_list length ({len(tg_list_all)}) does not match "
            f"params length ({len(params_all)})."
        )

    for n_truc in cfg["n_truc_list"]:
        print(f"\n===== n_truc = {n_truc} =====")
        if cfg["tg_para"]:
            label = f"n_truc={n_truc}, all tg"
            mem = MemoryMonitor(interval=0.2)
            mem.start()
            try:
                results = run_one_truncation(
                    n_truc, cfg, data, gate, option_ideal, option_noisy
                )
            finally:
                mem.stop()
                print("\n===== Memory usage =====")
                mem.print_summary(label)
                peak = mem.peak_gb()
                memory_stats.append(
                    {
                        "n_truc": n_truc,
                        "tg": "all",
                        "peak_rss_gb": peak["rss"],
                        "peak_pss_gb": peak["pss"],
                    }
                )

            if "ideal" in results:
                # ut.print_fidelity(f"f_ideal_{n_truc}", results["ideal"], num_digits=8)
                f_ideal_all.append(results["ideal"])
            if "noise" in results:
                ut.print_fidelity(
                    f"f_{cfg['t1_tphi_other']}us_{n_truc}",
                    results["noise"],
                    num_digits=8,
                )
                f_noise_all.append(results["noise"])
            ut.print_time()
            print("\n" + "=" * 60 + "\n")
        else:
            for tg, params in zip(tg_list_all, params_all):
                cfg_run = cfg.copy()
                cfg_run["tg_list"] = [tg.item() if isinstance(tg, np.generic) else tg]
                cfg_run["params"] = np.atleast_2d(params)
                cfg_run["n_job"] = 1

                tg_label = cfg_run["tg_list"][0]
                label = f"n_truc={n_truc}, tg={tg_label}"
                print(f"\n----- {label} -----")

                mem = MemoryMonitor(interval=0.2)
                mem.start()
                try:
                    results = run_one_truncation(
                        n_truc, cfg_run, data, gate, option_ideal, option_noisy
                    )
                finally:
                    mem.stop()
                    print("\n===== Memory usage =====")
                    mem.print_summary(label)
                    peak = mem.peak_gb()
                    memory_stats.append(
                        {
                            "n_truc": n_truc,
                            "tg": tg_label,
                            "peak_rss_gb": peak["rss"],
                            "peak_pss_gb": peak["pss"],
                        }
                    )

                if "ideal" in results:
                    # ut.print_fidelity(f"f_ideal_{n_truc}", results["ideal"], num_digits=8)
                    f_ideal_all.append(results["ideal"])
                if "noise" in results:
                    ut.print_fidelity(
                        f"f_{cfg['t1_tphi_other']}us_{n_truc}_tg{tg_label}",
                        results["noise"],
                        num_digits=8,
                    )
                    f_noise_all.append(results["noise"])
                ut.print_time()
                print("\n" + "=" * 60 + "\n")

    if memory_stats:
        print("\n===== Memory usage by n_truc and tg =====")
        for stat in memory_stats:
            print(
                "n_truc={n_truc}, tg={tg}: "
                "Peak RSS={peak_rss_gb:.2f} GB, "
                "Peak PSS={peak_pss_gb:.2f} GB".format(**stat)
            )

    if f_ideal_all:
        ut.print_fidelity("fidelity_ideal_list", f_ideal_all, num_each_row=1, num_digits=8)
    if f_noise_all:
        ut.print_fidelity("fidelity_noise_list", f_noise_all, num_each_row=1, num_digits=8)

if __name__ == "__main__":
    main()
