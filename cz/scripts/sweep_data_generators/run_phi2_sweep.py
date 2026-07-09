"""
Run a 21-point sweep of Φ2 over a symmetric log-spaced range and save all
sweep points into a single npz via save_two_qubit_sweep.

Usage
-----
    conda activate zp2q
    python run_phi2_sweep.py

Output
------
    data/Phi2_sweep/two_qubit_sweep_data.npz   (combined, evecs dropped)
    data/Phi2_sweep/summary_Φ2=<value>.txt    (one per sweep point)
"""

import numpy as np
from pathlib import Path
import sys

# from paths import DATA_FOLDER
DATA_FOLDER = "/data/zp_sweeps/test2"
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from ham_data import save_two_qubit_sweep, load_params


# Base parameters (truc1/truc2 = 300 come from the yaml)
params = load_params("../../data/Two_qubit_data_Sorted_Truc/params_truc.yaml", convert_lists=True)
params["truc_total"] = 1000

n_pts = 21  # number of sweep points
bound = 1e-04  # sweep range: -bound .. +bound
# Symmetric log spacing about 0: 10 negative log-spaced points, 0, and 10
# positive log-spaced points (21 total). The smallest nonzero magnitude is
# set 4 decades below `bound` since log spacing can't include 0 directly.
n_half = (n_pts - 1) // 2
floor = bound * 1e-04
pos_values = np.logspace(np.log10(floor), np.log10(bound), n_half)
sweep_values = np.concatenate([-pos_values[::-1], [0.0], pos_values])
print(f"Nominal Φ2 = {params['Φ2']}")
print(f"Sweeping Φ2 over {sweep_values[0]:.2e} .. {sweep_values[-1]:.2e} ({len(sweep_values)} points, log-spaced about 0)")

# One worker per point (21 points). Your .bashrc pins OMP/MKL/OpenBLAS to 1
# thread, so 21 single-threaded workers use ~21 of the 128 cores and finish in
# roughly one point's wall time. To spend the idle cores on making each point
# faster instead, set inner_max_num_threads (e.g. 6 -> 21*6=126 cores); it
# overrides the env inside each worker.
folder_save = Path(DATA_FOLDER, "Phi2_sweep")
save_two_qubit_sweep(params, "Φ2", sweep_values, folder_save,
                     n_jobs=n_pts, inner_max_num_threads=None)
print(f"Saved sweep to {folder_save}")
