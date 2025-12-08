#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Unified 2Q Gate Optimizer for Zero-Pi System
Supports:
    - CZ gate optimization
    - CNOT gate optimization

Features:
    - Modular, function-based structure
    - Differential evolution optimizer
    - Automatic saving (npz + csv)
    - Resume from previous runs
    - Simple plotting (fidelity & parameters)

Author: ChatGPT + Zhenxing's original scripts
Date: 2025
"""

# ===== Standard Library Imports =====
import os
import sys
from datetime import datetime
import pytz
import json

# ===== Third-Party Imports =====
import numpy as np
import scipy as sp
from tqdm import tqdm
import qutip as qt
import pandas as pd
from matplotlib import pyplot as plt
import scqubits.settings as settings

settings.OVERLAP_THRESHOLD = 0.3

# ===== Local Imports =====
sys.path.append('../')
import utils_2Q_gate_zp as ut
import ham_data as hd
from functools import partial


# === global variables for multiprocessing ===
GLOBAL_system_data = None
GLOBAL_hamiltonians = None
GLOBAL_config = None


# ==============================================================
# SYSTEM LOADING
# ==============================================================

def load_system_data(config):
    """根据 gate_type 分发到 CZ / CNOT 的系统加载函数。"""
    gate_type = config["gate_type"]
    if gate_type == "CZ":
        return load_system_data_cz(config)
    elif gate_type == "CNOT":
        return load_system_data_cnot(config)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


def load_system_data_cz(config):
    """
    基于你原来的 CZ 版本 load_system_data 改造。
    """

    print("Loading system data for CZ...")

    (hspace_full, eket_tot, eval_tot, _,
     n_theta1_dress, _, _, logi_state) = hd.load_two_qubit_data(
        config['folder_load'], return_full=False
    )

    drive_term = n_theta1_dress

    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]

    if config['use_truc_model']:
        hspace_select = ut.truc_model[config['truc_model_name']][:config['truc_optimize']]
    else:
        hspace_select = hspace_full[:config['truc_optimize']]

    option_ideal, option_noisy = ut.get_qutip_options(
        config['max_step_ideal'], config['max_step_noisy']
    )

    pulse_param = ut.load_drive_params_2q(
        config['gate_type'] == 'CZ', folder=config['folder_pulse']
    )[config['gate_time_indices'], :]

    print(f"Loaded CZ data: total states = {len(hspace_full)}")
    print(f"Optimization truncation = {len(hspace_select)}")
    print(f"W_20_50 = {np.round(W_20_50, 3)}")

    return dict(
        gate_type='CZ',
        hspace_full=hspace_full,
        eket_tot=eket_tot,
        eval_tot=eval_tot,
        drive_term=drive_term,
        logi_state=logi_state,
        hspace_select=hspace_select,
        W_20_50=W_20_50,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        pulse_param=pulse_param,
    )


def load_system_data_cnot(config):
    """
    CNOT 使用你旧脚本的那套数据结构：
        - eval_tot, hspace_full, n_theta0_dress, n_theta1_dress
        - eval_False, hspace_False, n_theta0_False, n_theta1_False
    """

    print("Loading system data for CNOT...")

    (hspace_full, eket_tot, eval_tot, n_theta0_dress,
     n_theta1_dress, hspace_0, hspace_1, logi_state) = hd.load_two_qubit_data(
        config['folder_load'], return_full=False
    )

    # 逻辑态索引
    if config['mid_state'] in ['8-2', '4-5']:
        idx_0 = hspace_full.index('0-2')
        idx_1 = hspace_full.index('2-2')
    elif config['mid_state'] in ['1-4', '8-0', '4-1']:
        idx_0 = hspace_full.index('0-0')
        idx_1 = hspace_full.index('2-0')
    else:
        raise ValueError(f"Unsupported mid_state for CNOT: {config['mid_state']}")

    if config['mid_state'] in ['8-2', '4-5', '1-4', '8-0', '4-1']:
        drive_term = n_theta0_dress
    else:
        drive_term = n_theta1_dress

    idx_mid = hspace_full.index(config['mid_state'])
    W_0_2 = eval_tot[idx_mid] - eval_tot[idx_0]
    W_1_2 = eval_tot[idx_mid] - eval_tot[idx_1]

    # 选择优化空间的 Hilbert space（沿用你原来的逻辑）
    if config['use_truc_model']:
        hspace_select = ut.truc_model['cnot_' + config['mid_state'][0] + config['mid_state'][2]][:config['truc_optimize']]
    else:
        hspace_select = hspace_full[:config['truc_optimize']]

    option_ideal, option_noisy = ut.get_qutip_options(config['max_step_ideal'], config['max_step_noisy'])

    # pulse 参数：优先使用 pulse 文件, 其次可以用 x0_array，
    if config['folder_pulse'] is not None:
        pulse_param = ut.load_drive_params_2q(
            config['gate_type']=='CZ', folder=config['folder_pulse']
        )[config['gate_time_indices'], :]
    else:
        pulse_param = config['x0_array'][config['gate_time_indices'], :]

    print(f"Loaded CNOT data: total states = {len(hspace_full)}")

    return dict(
        gate_type='CNOT',
        hspace_full=hspace_full,
        eket_tot=eket_tot,
        eval_tot=eval_tot,
        drive_term=drive_term,
        hspace_select=hspace_select,
        logi_state=logi_state,
        W_0_2=W_0_2,
        W_1_2=W_1_2,
        option_ideal=option_ideal,
        option_noisy=option_noisy,
        pulse_param=pulse_param,
    )


# ==============================================================
# HAMILTONIAN BUILDING
# ==============================================================

def build_hamiltonians(system_data, config):
    gate_type = config["gate_type"]
    if gate_type == "CZ":
        return build_hamiltonians_cz(system_data, config)
    elif gate_type == "CNOT":
        return build_hamiltonians_cnot(system_data, config)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


def build_hamiltonians_cz(sys, config):
    """
    和你原来的 CZ 版本 build_hamiltonians 类似。
    """
    print("Building Hamiltonians for CZ...")

    logi_idx_select = [sys['hspace_select'].index(i) for i in sys['logi_state']]
    index_select = [sys['hspace_full'].index(i) for i in sys['hspace_select']]

    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', index_select,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    hspace_large = sys['hspace_full'][:config['truc_large']]
    logi_idx_large = [hspace_large.index(i) for i in sys['logi_state']]
    idx_large = np.arange(config['truc_large']).tolist()

    H_drive_large, _ = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', idx_large,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    return dict(
        H_drive_select=H_drive_select,
        H_drive_large=H_drive_large,
        logi_idx_select=logi_idx_select,
        logi_idx_large=logi_idx_large,
    )


def build_hamiltonians_cnot(sys, config):
    """
    按照你旧 CNOT 脚本构建：
        - H_drive_part: 用于优化
        - H_drive_False: 大空间检查
    """
    print("Building Hamiltonians for CNOT...")

    logi_idx_select = [sys['hspace_select'].index(i) for i in sys['logi_state']]
    index_select = [sys['hspace_full'].index(i) for i in sys['hspace_select']]

    H_drive_select, eket_truc = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', index_select,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    hspace_large = sys['hspace_full'][:config['truc_large']]
    logi_idx_large = [hspace_large.index(i) for i in sys['logi_state']]
    idx_large = np.arange(config['truc_large']).tolist()

    H_drive_large, _ = ut.build_hamiltonian_2q(
        config['gate_type']=='CZ', idx_large,
        sys['eval_tot'], sys['eket_tot'], sys['drive_term']
    )

    return dict(
        H_drive_select=H_drive_select,
        H_drive_large=H_drive_large,
        logi_idx_select=logi_idx_select,
        logi_idx_large=logi_idx_large,
    )


# ==============================================================
# FIDELITY EVALUATION HELPERS
# ==============================================================

def evaluate_fidelity_truncated(x, system_data, hamiltonians, config):
    """给 DE 用的目标函数 wrapper。"""
    gate_type = config['gate_type']

    if gate_type == 'CZ':
        # x = [tg, amp, detune]
        args = [
            hamiltonians['H_drive_select'],
            system_data['W_20_50'],
            1,
            [],
            hamiltonians['logi_idx_select'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cz_fidelity_log_noise(x, *args)

    elif gate_type == 'CNOT':
        # x = [tg, A1, A2, d1, d2]
        args = [
            hamiltonians['H_drive_select'],
            system_data['W_0_2'],
            system_data['W_1_2'],
            1,
            [],
            hamiltonians['logi_idx_select'],
            config['mid_state'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cnot_fidelity_log_noise(x, *args)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


def evaluate_fidelity_large(drive_params, system_data, hamiltonians, config):
    """在大 Hilbert 空间上检查 fidelity。"""
    gate_type = config['gate_type']

    if gate_type == 'CZ':
        tg, amp, detune = drive_params
        n_cpu_parallel = 16
        arg_all = [
            tg, amp, detune,
            n_cpu_parallel,
            np.arange(config['truc_large']),
            system_data['W_20_50'],
            hamiltonians['H_drive_large'],
            hamiltonians['logi_idx_large'],
        ]
        return ut.cz_fidelity_log_old(arg_all)

    elif gate_type == 'CNOT':
        tg, A1, A2, d1, d2 = drive_params
        num_cpus = 1
        c_op_list = []
        arg_all = [
            tg, A1, A2, d1, d2,
            hamiltonians['H_drive_large'],
            system_data['W_0_2'],
            system_data['W_1_2'],
            num_cpus,
            c_op_list,
            hamiltonians['logi_idx_large'],
            config['mid_state'],
            system_data['option_ideal'],
            system_data['option_noisy'],
        ]
        return ut.cnot_fidelity_log(arg_all)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")


# ==============================================================
# OPTIMIZATION CORE
# ==============================================================

def fid_func_global(x):
    # x 是 differential_evolution 给的参数
    # 复杂对象从全局读，不用被 pickle
    return evaluate_fidelity_truncated(
        x,
        system_data=GLOBAL_system_data,
        hamiltonians=GLOBAL_hamiltonians,
        config=GLOBAL_config,
    )

from pathos.multiprocessing import ProcessingPool as Pool
pool = Pool(nodes=50)
def vectorized_fid(X):
    return pool.map(fid_func_global, X)


def optimize_single_gate_time(
    gate_time_idx, system_data, hamiltonians, config, drive_param_list
):
    gate_type = config['gate_type']
    x0_full = system_data['pulse_param'][gate_time_idx]

    # ===== bounds =====
    if gate_type == 'CZ':
        tg_initial = x0_full[0]
        tg_bounds = (tg_initial + config['tg_bound'][0],
                     tg_initial + config['tg_bound'][1])
        bounds = (tg_bounds, config['amp_bound'], config['detune_bound'])
    else:
        tg_initial = x0_full[0]
        tg_bounds = (tg_initial + config['tg_bound'][0],
                     tg_initial + config['tg_bound'][1])
        bounds = (
            tg_bounds,
            config['A1_bound'],
            config['A2_bound'],
            config['detune1_bound'],
            config['detune2_bound'],
        )

    print(f"\n--- Optimizing {gate_type} at index {gate_time_idx}, tg_init={tg_initial:.6f} ---")
    ut.print_time()

    # differential evolution settings
    # 用 partial 包装成可 pickle 的函数
    # fid_func = partial(
    #     evaluate_fidelity_truncated,
    #     system_data=system_data,
    #     hamiltonians=hamiltonians,
    #     config=config
    # )

    global GLOBAL_system_data, GLOBAL_hamiltonians, GLOBAL_config

    # === 把复杂对象放入全局变量 ===
    GLOBAL_system_data = system_data
    GLOBAL_hamiltonians = hamiltonians
    GLOBAL_config = config

    params = dict(
        func=fid_func_global,
        bounds=bounds,
        disp=True,
        callback=ut.print_soln,
        init="sobol",
        workers=config['workers'],
        popsize=config['popsize'],
        mutation=config['mutation'],
        recombination=config['recombination'],
        tol=config['tol'],
        polish=False,
    )

    # x0 逻辑
    if config['use_x0'] == 'from_neighbor':
        if gate_time_idx == 0:
            if config['first_x0_from_input']:
                print("Using first x0 from input for initialization (first gate time)")
                params['x0'] = x0_full[:len(bounds)]
            else:
                print("No x0 for first gate time")
        else:
            print("Using x0 from previous optimized param")
            params['x0'] = [tg_initial] + drive_param_list[-1][1:]
    elif config['use_x0'] == 'from_input':
        print("Using x0 from input for initialization")
        params['x0'] = x0_full[:len(bounds)]
    else:
        print("No x0 used")

    print(f'params["x0"] = {params.get("x0", "None")}')
    result = sp.optimize.differential_evolution(**params)

    ut.print_time()
    print(f"Optimization completed, fidelity = {result.fun:.8f}")
    print(f"Optimized params = {result.x}")

    return result.fun, result.x.tolist()


# ==============================================================
# SAVE / RESUME / PLOT
# ==============================================================

def save_results(config, system_data, fidelity_list, fidelity_large_list,
                 drive_param_list, last_index):
    """保存当前优化结果到 npz + csv。"""
    result_file = config['result_file']
    csv_file = config['csv_file']

    np.savez(
        result_file,
        config=json.dumps(config, default=str),
        fidelity=np.array(fidelity_list),
        fidelity_large=np.array(fidelity_large_list),
        drive_params=np.array(drive_param_list, dtype=object),
        last_index=last_index,
    )
    print(f"[SAVE] npz saved to {result_file}")

    # 保存 csv（简单表格式）
    if len(drive_param_list) > 0:
        max_len = max(len(p) for p in drive_param_list)
        data = {}
        data['idx'] = np.arange(len(drive_param_list))
        data['f_trunc'] = fidelity_list
        if len(fidelity_large_list) == len(drive_param_list):
            data['f_large'] = fidelity_large_list
        for j in range(max_len):
            data[f'p{j}'] = [p[j] if j < len(p) else np.nan for p in drive_param_list]

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"[SAVE] csv saved to {csv_file}")


def load_resume_if_any(config):
    """断点续跑: 如果配置要求 resume 且文件存在，则加载之前的结果。"""
    if not config.get('resume', False):
        return 0, [], [], []

    result_file = config['result_file']
    if not os.path.exists(result_file):
        print(f"[RESUME] No existing file {result_file}, starting from scratch.")
        return 0, [], [], []

    data = np.load(result_file, allow_pickle=True)
    fidelity_list = data.get('fidelity', np.array([])).tolist()
    fidelity_large_list = data.get('fidelity_large', np.array([])).tolist()
    drive_param_list = data.get('drive_params', np.array([], dtype=object)).tolist()
    last_index = int(data.get('last_index', 0))

    start_idx = last_index + 1
    print(f"[RESUME] Loaded from {result_file}, resume from index {start_idx}.")

    return start_idx, fidelity_list, fidelity_large_list, drive_param_list


# def plot_results(config, fidelity_list, fidelity_large_list, drive_param_list):
#     if not config.get('do_plot', False):
#         return

#     plot_dir = config['plot_dir']
#     os.makedirs(plot_dir, exist_ok=True)

#     # fidelity vs index
#     x = np.arange(len(fidelity_list))

#     plt.figure()
#     plt.plot(x, fidelity_list, 'o-', label='log(error) truncated')
#     if len(fidelity_large_list) == len(fidelity_list):
#         plt.plot(x, fidelity_large_list, 's--', label='log(error) large')
#     plt.xlabel("Gate index")
#     plt.ylabel("log(error)")
#     plt.legend()
#     plt.title(f"{config['gate_type']} fidelity")
#     fname = os.path.join(plot_dir, f"{config['gate_type']}_fidelity.png")
#     plt.savefig(fname, dpi=200, bbox_inches='tight')
#     plt.close()
#     print(f"[PLOT] Saved {fname}")

#     # parameters vs index
#     if len(drive_param_list) == 0:
#         return
#     arr = np.array(drive_param_list)
#     for j in range(arr.shape[1]):
#         plt.figure()
#         plt.plot(x, arr[:, j], 'o-')
#         plt.xlabel("Gate index")
#         plt.ylabel(f"param {j}")
#         plt.title(f"{config['gate_type']} parameter {j}")
#         fname = os.path.join(plot_dir, f"{config['gate_type']}_param{j}.png")
#         plt.savefig(fname, dpi=200, bbox_inches='tight')
#         plt.close()
#         print(f"[PLOT] Saved {fname}")


def plot_results(config, fidelity_list, fidelity_large_list, drive_param_list):
    """
    绘制 3×1 图：
    1. fidelity vs tg
    2. drive_amp vs tg
    3. detuning vs tg

    CNOT 参数：
        0: tg
        1: drive_amp1
        2: drive_amp2
        3: detuning1
        4: detuning2
    CZ 参数：
        0: tg
        1: drive_amp1
        2: detuning1
    """
    if not config.get('do_plot', False):
        return

    plot_dir = config['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)

    if len(drive_param_list) == 0:
        print("[PLOT] No parameters to plot.")
        return

    arr = np.array(drive_param_list)   # shape: (N, num_params)
    tg = arr[:, 0]

    # -----------------------------
    # Fidelity subplot
    # -----------------------------
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))

    ax[0].set_title(f"{config['gate_type']} optimization results")

    # truncated fidelity
    fidelity = 10 ** np.array(fidelity_list)
    ax[0].plot(tg, fidelity, ".-", label="truncated")

    # large Hilbert fidelity
    if len(fidelity_large_list) == len(fidelity_list):
        fidelity_large = 10 ** np.array(fidelity_large_list)
        ax[0].plot(tg, fidelity_large, "o", label="large", alpha=0.8)

    ax[0].set_ylabel("Fidelity")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid()

    # -----------------------------
    # Drive amplitude subplot
    # -----------------------------
    gate_type = config["gate_type"].upper()

    if gate_type == "CNOT":
        # param1 = amp1; param2 = amp2
        ax[1].plot(tg, arr[:, 1], ".-", label="drive_amp1")
        ax[1].plot(tg, arr[:, 2], ".-", label="drive_amp2")
    else:  # CZ
        ax[1].plot(tg, arr[:, 1], ".-", label="drive_amp1")

    ax[1].set_ylabel("Drive amplitude")
    ax[1].legend()
    ax[1].grid()

    # -----------------------------
    # Detuning subplot
    # -----------------------------
    if gate_type == "CNOT":
        ax[2].plot(tg, arr[:, 3], ".-", label="detuning1")
        ax[2].plot(tg, arr[:, 4], ".-", label="detuning2")
    else:  # CZ
        ax[2].plot(tg, arr[:, 2], ".-", label="detuning1")

    ax[2].set_ylabel("Detuning")
    ax[2].set_xlabel("Gate time tg")
    ax[2].legend()
    ax[2].grid()

    # -----------------------------
    # Save figure
    # -----------------------------
    fname = os.path.join(plot_dir, f"{config['gate_type']}_summary.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[PLOT] Saved {fname}")


# ==============================================================
# PRINTING
# ==============================================================

def print_configuration_summary(sys, config):
    print("\n" + "=" * 60)
    print(f"{config['gate_type']} GATE FIDELITY OPTIMIZATION CONFIGURATION")
    print("=" * 60)
    print(f"Start time: {datetime.now(pytz.timezone('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"gate_type = {config['gate_type']}")
    
    print(f"truc_large      = {config['truc_large']}")
    print(f"truc_optimize   = {config['truc_optimize']}")    
    print(f"use_truc_model   = {config['use_truc_model']}")
    print(f"truc_model_name  = {config['truc_model_name']}")
    
    print(f"max_step_ideal  = {config['max_step_ideal']}")
    print(f"max_step_noisy  = {config['max_step_noisy']}")

    print(f"workers         = {config['workers']}")
    print(f"popsize         = {config['popsize']}")
    print(f"recombination   = {config['recombination']}")
    print(f"tol             = {config['tol']}")
    print(f"mutation        = {config['mutation']}")
    print(f"resume          = {config['resume']}")
    print(f"do_plot         = {config['do_plot']}")
    
    print(f"use_x0         = {config['use_x0']}")
    print(f"first_x0_from_input = {config['first_x0_from_input']}")
    
    print(f"result_file     = {config['result_file']}")
    
    print(f"folder_pulse   = {config['folder_pulse']}")
    print(f"gate_time_indices = {config['gate_time_indices']}") 

    if config['gate_type'] == 'CZ':
        print(f"W_20_50         = {np.round(sys['W_20_50'], 3)}")
        print(f"amp_bound       = {config['amp_bound']}")
        print(f"detune_bound    = {config['detune_bound']}")
    else:
        print(f"mid_state       = {config['mid_state']}")
        print(f"W_0_2, W_1_2    = {np.round(sys['W_0_2'], 3)}, {np.round(sys['W_1_2'], 3)}")
        print(f"A1_bound        = {config['A1_bound']}")
        print(f"A2_bound        = {config['A2_bound']}")
        print(f"detune1_bound   = {config['detune1_bound']}")
        print(f"detune2_bound   = {config['detune2_bound']}")
        
        print(f"x0_array       = {config['x0_array']}")

    print(f"pulse_param = {sys['pulse_param']}")

    print("=" * 60)


# ==============================================================
# RUN SWEEP
# ==============================================================

def run_fidelity_sweep(system_data, hamiltonians, config):
    print("Starting fidelity sweep...")

    # resume
    start_idx, fidelity_list, fidelity_large_list, drive_param_list = load_resume_if_any(config)

    n_total = len(system_data['pulse_param'])
    # print(f"pulse_param = {system_data['pulse_param']}")
    # print(f"Total gate times to optimize: {n_total}, starting from index {start_idx}")
    
    for jdx in tqdm(range(start_idx, n_total), desc=f"{config['gate_type']} optimize"):
        f_trunc, params = optimize_single_gate_time(
            jdx, system_data, hamiltonians, config, drive_param_list
        )
        fidelity_list.append(f_trunc)
        drive_param_list.append(params)

        ut.print_pulse_params("param_optimized", drive_param_list)
        ut.print_fidelity(f"f_{config['truc_optimize']}", fidelity_list, num_digits=8)

        f_large = evaluate_fidelity_large(params, system_data, hamiltonians, config)
        fidelity_large_list.append(f_large)
        ut.print_fidelity(f"f_{config['truc_large']}", fidelity_large_list, num_digits=8)

        # 每步都保存一次，方便断点恢复
        save_results(config, system_data, fidelity_list, fidelity_large_list,
                     drive_param_list, jdx)

    print("Fidelity sweep completed!")

    # 最后画图
    plot_results(config, fidelity_list, fidelity_large_list, drive_param_list)


# ==============================================================
# MAIN
# ==============================================================

def main():
    print(f"Starting {os.path.basename(__file__)}")
    ut.print_time()

    # ===== 这里切换门类型：'CZ' 或 'CNOT' =====
    # gate_type = "CZ"
    gate_type = "CNOT"

    # 如果要覆盖某些配置，在这里传 custom_config
    custom_config = {
        # "resume": True,
        # "do_plot": False,
    }
    config = get_optimization_config(gate_type=gate_type, custom_config=custom_config)

    # Step 1: Load system data
    system_data = load_system_data(config)

    # Step 2: Build Hamiltonians
    hamiltonians = build_hamiltonians(system_data, config)

    # Step 3: Print configuration summary
    print_configuration_summary(system_data, config)

    # Step 4: Run sweep
    run_fidelity_sweep(system_data, hamiltonians, config)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)

# ==============================================================
# CONFIGURATION
# ==============================================================

def get_optimization_config(gate_type="CZ", custom_config=None):
    """
    返回统一配置字典。
    gate_type: "CZ" 或 "CNOT"
    custom_config 若提供，则覆盖默认配置。
    """

    # ===== 通用部分 =====
    config = dict(
        gate_type=gate_type,

        # truncation
        truc_large=1000,
        truc_optimize=200,
        use_truc_model=False,
        truc_model_name="cz_short_500_detune1",

        # qutip options
        max_step_ideal=1e-3,
        max_step_noisy=1e-3,

        # differential evolution parameters
        workers=-1,
        popsize=10,
        recombination=0.7,
        tol=0.01,
        mutation=(0.5, 1.0),

        # resume & saving
        resume=False,
        do_plot=True,

        # x0 使用方式：None / 'from_neighbor' / 'from_input'
        use_x0='from_neighbor',
        first_x0_from_input=True,
        
        folder_load='../../data/_truc_3000',
        logi_state=['0-0', '0-2', '2-0', '2-2'],    
    )

    # =====================================================
    # 🔥 正确添加时间戳目录（唯一生效的路径）
    # =====================================================    
    
    beijing_tz = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")

    label_str = f"{gate_type.lower()}_{timestamp}"
    base_dir = f"data/results/{label_str}"
    result_file = os.path.join(base_dir, f"{label_str}.npz")
    csv_file = os.path.join(base_dir, f"{label_str}.csv")
    os.makedirs(base_dir, exist_ok=True)

    # 放入 config
    config.update(dict(
        timestamp=timestamp,
        result_dir=base_dir,
        result_file=result_file,
        csv_file=csv_file,
        plot_dir=base_dir,
    ))
    # =====================================================

    # ===== CZ 特定参数 =====
    if gate_type == "CZ":
        config.update(dict(
            # data path
            folder_pulse='data/npz/cz_pulse_neighbor.txt',
            gate_time_indices=(np.arange(20, 50) - 20).tolist(),

            tg_bound=(-0.01, 0.01),
            amp_bound=(0.01, 0.05),
            detune_bound=(0.001, 0.04),
        ))

    # ===== CNOT 特定参数 =====
    elif gate_type == "CNOT":
        # 这里沿用你旧脚本中对 CNOT 的数据读取方式（truc1/truc_tot/folder）
        config.update(dict(
            
            mid_state='8-2',
            
            # 数据相关
            truc1=300,
            truc_tot=1000,
            truc_full=1000,
            charge_pick=True,

            # 如果有 pulse 文件，也可以设置为 True + 给路径
            folder_pulse='../figure/data/data_cnot_fidelity_3ncut.txt',
            gate_time_indices= [0, 15, -1], # np.arange(3).tolist(),  #[0],

            # CNOT 参数 bounds （你可以按需要改）
            tg_bound=(-0.001, 0.001),
            A1_bound=(0.01, 0.15),
            A2_bound=(0.01, 0.15),
            detune1_bound=(-0.1, 0.1),
            detune2_bound=(-0.1, 0.1),

            # 初始 x0 向量（可扩展多行对应多个 gate time）
            x0_array=np.array([
                [2.009075, 0.432514, 0.042229, -0.181457, -0.236707],
            ]),
        ))
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    # 用户自定义覆盖
    if custom_config:
        config.update(custom_config)

    return config

if __name__ == '__main__':
    main()
