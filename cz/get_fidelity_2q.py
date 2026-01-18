import os
import sys
from pathlib import Path
import numpy as np
import qutip as qt
from joblib import Parallel, delayed

sys.path.append('../')

import scqubits.settings as settings
import utils_2Q_gate_zp as ut
import ham_data as hd

settings.OVERLAP_THRESHOLD = 0.3


# ============================================================
# Configuration
# ============================================================
def get_config():
    """
    Return a dictionary containing all run-time configurations.
    """
    cfg = {}

    # ---------- General ----------
    cfg["cz_run"] = True # True for CZ, False for CNOT
    
    # ---------- Truncation ----------
    # cfg["n_truc_list"] = np.arange(50, 201, step=10) # [200] #    
    cfg["n_truc_list"] = np.arange(300, 901, step=100) # [200] #    
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
    if cfg["calculate_noise"]:
        cfg["apply_decay"] = True
        cfg["apply_dephase"] = True
        cfg["decay_enlarge"] = 1
        cfg["filter_ratio"] = 0.3

        # ---------- Time / noise ----------
        cfg["t1_tphi_other"] = 3  # us

    # ---------- Parallel ----------
    cfg["num_cpus"] = 4 # 16

    # ---------- Data ----------
    cfg["folder_load"] = '../data/December_17_2025_Sorted_Untruc'
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
    
    if cfg["cz_run"]:
        folder = 'data/npz/cz_pulse_neighbor.txt'
        cfg["tg_list"] = np.arange(180)[0::6]
        cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], ]  # [1::4,] 
    else:
        folder = '../cnot/data/cnot_fidelity_npz.txt'
        cfg["tg_list"] = np.arange(26)#[:1] # [0::6]
        cfg["params"] = ut.load_drive_params_2q(cfg["cz_run"], folder=folder)[cfg["tg_list"], ]  # [1::4,]

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
    
    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        cfg["cz_run"],
        index_select,
        data["eval_tot"],
        data["eket_tot"],
        gate["drive_term"],
    )

    logi_idx_select = [hspace_select.index(i) for i in data["logi_state"]]

    # ---------- Ideal ----------
    if cfg["calculate_ideal"]:
        c_op_list = []

        if cfg["cz_run"]:
            args_common = [
                H_drive_select,
                gate["W_20_50"],
                cfg["num_cpus"],
                c_op_list,
                logi_idx_select,
                option_ideal,
                option_noisy,
            ]
            f_list = Parallel(n_jobs=cfg["n_job"])(
                delayed(ut.cz_fidelity_log_noise)(p, *args_common)
                for p in cfg["params"]
            )
        else:
            args_common = [
                H_drive_select,
                gate["W_0_2"],
                gate["W_1_2"],
                cfg["num_cpus"],
                c_op_list,
                logi_idx_select,
                gate["mid_state"],
                option_ideal,
                option_noisy,
            ]
            f_list = Parallel(n_jobs=cfg["n_job"])(
                delayed(ut.cnot_fidelity_log_noise)(p, *args_common)
                for p in cfg["params"]
            )

        return f_list

    return None


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
def main():
    print(os.path.basename(__file__))
    ut.print_time()

    cfg = get_config()
    print_config(cfg)
    data = load_system_data(cfg)
    gate = select_gate_quantities(cfg, data)

    option_ideal, option_noisy = ut.get_qutip_options(
        cfg["max_step_ideal"], cfg["max_step_noisy"]
    )

    ut.print_fidelity("params", cfg["params"].tolist(), num_each_row=1)

    f_all = []
    for n_truc in cfg["n_truc_list"]:
        print(f"\n===== n_truc = {n_truc} =====")
        f = run_one_truncation(
            n_truc, cfg, data, gate, option_ideal, option_noisy
        )
        ut.print_fidelity(f"f_ideal_{n_truc}", f, num_digits=8)
        f_all.append(f)
        ut.print_time()

    ut.print_fidelity("fidelity_list", f_all, num_each_row=1, num_digits=8)


if __name__ == "__main__":
    main()

