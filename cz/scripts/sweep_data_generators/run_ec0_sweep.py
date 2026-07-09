"""
Run a 21-point sweep of Ec0 over +/- 5% of its nominal value and save all
sweep points into a single npz via save_two_qubit_sweep.

Usage
-----
    conda activate zp2q
    python run_ec0_sweep.py

Output
------
    data/Ec0_sweep/two_qubit_sweep_data.npz   (combined, evecs dropped)
    data/Ec0_sweep/summary_Ec0=<value>.txt    (one per sweep point)
"""

import numpy as np
from pathlib import Path
import sys

# from paths import DATA_FOLDER
DATA_FOLDER = "/data/zp_sweeps/test2"
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from ham_data import save_two_qubit_sweep, load_params



def ghz_to_henry(El):
    e = 1.60217663*(10**(-19))
    hbar = 1.05457182*(10**(-34))
    return 1e-09*(1/El)*((hbar/(2*np.pi))/e**2)*(1/4)

def henry_to_ghz(L):
    e = 1.60217663*(10**(-19))
    hbar = 1.05457182*(10**(-34))
    return 1e-09*(1/L)*((hbar/(2*np.pi))/e**2)*(1/4)


def ghz_to_farad(Ec):
    e = 1.60217663*(10**(-19))
    hbar = 1.05457182*(10**(-34))
    return 1e-09*((e**2)/(2*Ec))*(1/(2*np.pi*hbar))

def farad_to_ghz(C):
    e = 1.60217663*(10**(-19))
    hbar = 1.05457182*(10**(-34))
    return 1e-09*((e**2)/(2*C))*(1/(2*np.pi*hbar))

# Base parameters (truc1/truc2 = 300 come from the yaml)
params = load_params("../../data/Two_qubit_data_Sorted_Truc/params_truc.yaml", convert_lists=True)
params["truc_total"] = 1000

n_pts = 21  # number of sweep points
max_entent = (0.95, 1.05)  # sweep range as fraction of nominal value
# 21 points spanning +/- 5% of nominal Ec0, swept linearly in the underlying
# ground capacitance C0 (Ec0 ~ 1/C0, and capacitance is the additive physical
# quantity), same convention as the EC2 sweep.
nominal_Ec0 = params["Ec0"]
nominal_C0 = ghz_to_farad(nominal_Ec0)
sweep_values = farad_to_ghz(np.linspace(max_entent[0] * nominal_C0, max_entent[1] * nominal_C0, n_pts))
print(f"Nominal Ec0 = {nominal_Ec0}")
print(f"Sweeping Ec0 over {sweep_values[0]:.5f} .. {sweep_values[-1]:.5f} ({len(sweep_values)} points)")

# One worker per point (21 points). Your .bashrc pins OMP/MKL/OpenBLAS to 1
# thread, so 21 single-threaded workers use ~21 of the 128 cores and finish in
# roughly one point's wall time. To spend the idle cores on making each point
# faster instead, set inner_max_num_threads (e.g. 6 -> 21*6=126 cores); it
# overrides the env inside each worker.
folder_save = Path(DATA_FOLDER, "Ec0_sweep")
save_two_qubit_sweep(params, "Ec0", sweep_values, folder_save,
                     n_jobs=n_pts, inner_max_num_threads=None)
print(f"Saved sweep to {folder_save}")
