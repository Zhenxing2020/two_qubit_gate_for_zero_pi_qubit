import numpy as np

from ham_data import *


def build_spec_data(
    params_path: str,
    evals_count: int,
    num_cpus: int,
    subtract_ground: bool,
):
    params = load_params(params_path)
    # Define system parameters (in GHz)
    EL = params["EL"]  # Inductive energy
    EJ = params["EJ"] * 0.9  # Josephson energy
    EC_phi = params["EC_phi"]  # Phi mode charging energy
    EC_theta = params["EC_theta"]  # Theta mode charging energy

    # Compute derived parameters
    E_CJ = 2 * EC_phi
    E_C = 2 / (1 / EC_theta - 1 / EC_phi)

    # Create the grid for the phi coordinate
    phi_grid = scq.Grid1d(
        params["phi_range"][0],
        params["phi_range"][1],
        params["phi_cut"],
    )

    # Initialize the Zero-Pi qubit system
    zero_pi = scq.ZeroPi(
        grid=phi_grid,
        EJ=EJ,
        EL=EL,
        ECJ=E_CJ,
        EC=E_C,
        dEJ=params["dEJ"],
        ng=params["ng"],
        flux=params["flux"],
        ncut=params["n_cut"],
        truncated_dim=params["truc1"],
    )

    spec_data = zero_pi.get_spectrum_vs_paramvals(
        "flux",
        [0] + list(np.logspace(-6, -1, 101)),
        evals_count=evals_count,
        num_cpus=num_cpus,
        subtract_ground=subtract_ground,
    )

    meta = {
        "EJ": EJ,
        "EL": EL,
        "ECJ": E_CJ,
        "EC": E_C,
        "dEJ": params["dEJ"],
        "ng": params["ng"],
        "flux": params["flux"],
        "ncut": params["n_cut"],
        "truncated_dim": params["truc1"],
        "phi_range": np.array(params["phi_range"]),
        "phi_cut": params["phi_cut"],
        "evals_count": evals_count,
        "num_cpus": num_cpus,
        "subtract_ground": subtract_ground,
        "param_name": "flux",
    }

    return spec_data, meta


def save_spec_data(spec_data, output_path: str, meta: dict):
    np.savez(
        output_path,
        param_vals=np.array(spec_data.param_vals),
        energy_table=np.array(spec_data.energy_table),
        **meta,
    )

# nohup time python zp_spectrum_vs_flux_2nd_derivative_run.py > 'data/gen_2nd_deri.txt' &

def main():
    # Modify these by hand as needed.
    params_path = "params.yaml"
    subtract_ground = True

    output_path = "data/data_flux_derivative_q1_eval_300_cpu_100.npz"
    evals_count = 300
    num_cpus = 100

    spec_data, meta = build_spec_data(
        params_path=params_path,
        evals_count=evals_count,
        num_cpus=num_cpus,
        subtract_ground=subtract_ground,
    )
    save_spec_data(spec_data, output_path, meta)


if __name__ == "__main__":
    main()
