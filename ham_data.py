
import scqubits as scq
import numpy as np
import qutip as qt
import scipy.sparse as ssp
import sympy as sym
import yaml
from pathlib import Path

from paths import DATA_FOLDER, RESULTS_FOLDER
from truncation_estimate import trunc_by_thresh, trunc_by_graph_estimate
import utils_2Q_gate_zp as ut


def find_overlap_complex(eket, hspace_bare, num=10):

    # flat_array = overlaps.flatten() # Flatten the 2D array
    if isinstance(eket, qt.Qobj):
        eket = eket.data
    flat_array = eket.flatten()
    top_indices_flat = np.argsort(-np.abs(flat_array))[:num]
    hspace_select = [hspace_bare[i] for i in top_indices_flat]
    
    return hspace_select, flat_array[top_indices_flat]


def dressed_state_decomp(top_states, top_values, thresh=0.001, float_round=3, dressed_state_label=None,
                         latex=True):
    """
    Returns a latex string that shows the given dressed state in terms of the bare eigenstates.
    
    Args:
        sorted_top_indices: List of tuples (i, j) representing bare state indices, sorted by overlap magnitude
        sorted_top_values: Array of overlap values corresponding to the indices, sorted by magnitude
        thresh (float): Threshold of overlap (abs value) to be displayed
        float_round (int): Number of digits to round floats in the string to
        dressed_state_label (str): Optional label for the dressed state. If None, uses "state"
    
    Returns:
        str: latex string
    """
    
    # Start building the LaTeX string
    if dressed_state_label is None:
        if latex:
            str_repr = r"$|\text{dressed}\rangle = "
        else:
            str_repr = r"dressed = "
    else:
        if latex:
            str_repr = r"$\widetilde{|\text{" + dressed_state_label + r"}\rangle} = "
        else:
            str_repr = f"|{dressed_state_label}⟩ = "

    # Add significant overlaps to the string
    for idx, overlap_val in zip(top_states, top_values):
        if np.abs(overlap_val)**2 >= thresh:
            i, j = idx
            # Add the coefficient and bare state
            if latex:
                str_repr += str(np.round(overlap_val, float_round)) + \
                        r"\left|" + f"{i},{j}" + r"\right\rangle + "
            else:
                str_repr += f"({np.round(overlap_val, float_round)})|{i},{j}⟩ + "
    
    # Remove the trailing " + " and close the equation
    if str_repr.endswith(" + "):
        str_repr = str_repr[:-3]
    
    if latex:
        str_repr += r"$"
    return str_repr

def get_dressed_states_index(top_index, hspace_0, hspace_1):
    """
    Get the dressed states index for each eigenstate.

    Args:
        top_index (list): List of index pairs for each eigenstate.
        hspace_0 (list or np.ndarray): Indices for qubit 0.
        hspace_1 (list or np.ndarray): Indices for qubit 1.

    Returns:
        hspace_full (list): List of string indices for dressed states.
    """
    index_array = []
    for i, index in enumerate(top_index):
        j = 0
        while j < len(index):
            if index[j] not in index_array:
                index_array.append(index[j])
                break
            else:
                j += 1
            if j == 10:
                index_array.append((0, 0))
                print(i, 'need to further compare overlap')
    hspace_full = [f"{hspace_0[idx[0]]}-{hspace_1[idx[1]]}" for idx in index_array]
    return hspace_full


def normalize_eigenvector_phases(eigenvectors):
    """
    Normalize the phases of a list of eigenvectors to ensure consistent phase convention.
    
    Parameters:
        eigenvectors (list): List of eigenvectors (can be numpy arrays or qutip Qobj)
    
    Returns:
        list: List of phase-normalized eigenvectors in the same format as input
    """
    normalized_evecs = []
    
    for evec in eigenvectors:
        # Handle both qutip Qobj and numpy array inputs
        if isinstance(evec, qt.Qobj):
            data = evec.data.toarray().flatten()
            is_qobj = True
            original_shape = evec.shape
        else:
            data = np.array(evec).flatten()
            is_qobj = False
            original_shape = np.array(evec).shape
        
        # Find the phase correction based on the chosen method
        # Find element with largest magnitude
        max_idx = np.argmax(np.abs(data))
        phase_element = data[max_idx]
        
        # Calculate phase correction
        phase_correction = np.exp(-1j * np.angle(phase_element))
        
        # Apply phase correction
        normalized_data = data * phase_correction
        
        # Reshape back to original format
        if is_qobj:
            if len(original_shape) == 2:
                normalized_data = normalized_data.reshape(original_shape)
                normalized_evec = qt.Qobj(normalized_data)
            else:
                normalized_evec = qt.Qobj(normalized_data)
        else:
            normalized_evec = normalized_data.reshape(original_shape)
        
        normalized_evecs.append(normalized_evec)
    
    return normalized_evecs


def save_two_qubit_data(params, folder_save):

    # Replace parameters in the YAML template and make Circuit
    yml_with_params = params["zp_yml"]
    for cir_param in ["EJ1", "EJ2", "ECJ1", "ECJ2", "EL1", "EL2", "EC1", "EC2", "Ec0", "Ecc"]:
        yml_with_params = yml_with_params.replace(f"{cir_param}=x", f"{cir_param}={params[cir_param]}")
    zp = scq.Circuit(yml_with_params, from_file=False)
    zp.configure(transformation_matrix=params["Ztransform_2zeropi"],)
    zp.Φ1 = params["Φ1"]
    zp.Φ2 = params["Φ2"]
    setattr(zp, f"ng{params['theta_mode1']+1}", params["ng1"])
    setattr(zp, f"ng{params['theta_mode2']+1}", params["ng2"])

    # Set cutoffs and discretized phi range
    for i in zp.var_categories["periodic"]:
        setattr(zp, f"cutoff_n_{i}", params["n_cut"])
    for i in zp.var_categories["extended"]:
        setattr(zp, f"cutoff_ext_{i}", params["phi_cut"])
    zp.set_discretized_phi_range(var_indices=zp.var_categories["extended"], phi_range=params["phi_range"])


    summary_file = str(Path(folder_save, 'two_qubit_data_summary.txt'))
    with open(summary_file, 'w') as f:
        print("Two Qubit Data Summary:", file=f)
        print("scqubits version:", scq.__version__, file=f)
        print(f"circuit modes: {zp.var_categories}", file=f)
        print(f"circuit params: {zp.symbolic_params}", file=f)
        print("-", file=f)
        print(f"lagrangian (node vars): {sym.nsimplify(zp.sym_lagrangian(return_expr=True))}", file=f)
        print("-", file=f)
        print(f"hamiltonian (transformed vars): {sym.nsimplify(zp.sym_hamiltonian(return_expr=True))}", file=f)
        print("-", file=f)
        print(str(zp), file=f)
    
    # Define subsystems
    system_hierarchy = [[1,3],  [5,7]]  # theta and phi modes for each qubit
    subsystem_trunc_dims = [10, 10]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)
    
    print("Circuit and parameters set. Beginning calculations...")
    
    # Extract g_theta1theta2 coupling strength
    i_for_inv = []
    for i in range(params["Ztransform_2zeropi"].shape[0]):
        if i not in zp.var_categories["sigma"]:
            i_for_inv.append(i)
    Z = params["Ztransform_2zeropi"]
    cMat = zp.symbolic_circuit._capacitance_matrix(substitute_params=True)
    cTransInv = np.linalg.inv(ut.truncate_2(Z.T @ cMat @ Z, i_for_inv))
    g = cTransInv[params["theta_mode1"], params["theta_mode2"]]

    # Calculate subsystem eigenvalues/vectors
    H_zp1 = zp.subsystems[0].hamiltonian()
    eval1, evecs1 = ssp.linalg.eigsh(H_zp1, k=params["truc1"],  which='SA')
    evecs1 = np.array(normalize_eigenvector_phases(evecs1.T))
    H_zp2 = zp.subsystems[1].hamiltonian()
    eval2, evecs2 = ssp.linalg.eigsh(H_zp2, k=params["truc2"],  which='SA')
    evecs2 = np.array(normalize_eigenvector_phases(evecs2.T))

    # ntheta and nphi operators in single qubit bare basis
    n_theta1 = (evecs1 @ getattr(zp.subsystems[0], f"n{params['theta_mode1']+1}_operator")() @ evecs1.conj().T)
    n_theta2 = (evecs2 @ getattr(zp.subsystems[1], f"n{params['theta_mode2']+1}_operator")() @ evecs2.conj().T)

    # Reduced models for individual qubits
    hspace_1 = trunc_by_thresh([0, 2], n_theta1, params["charge_thresh"])
    hspace_2 = trunc_by_thresh([0, 2], n_theta2, params["charge_thresh"])
    # truncated n_theta and n_phi operators
    n_theta1_trunc = ut.truncate_2(n_theta1, hspace_1)
    n_theta2_trunc = ut.truncate_2(n_theta2, hspace_2)
    eval1_trunc = eval1[hspace_1]
    eval2_trunc = eval2[hspace_2]

    # Full System Hamiltonian
    H1 = qt.Qobj(np.diag(eval1_trunc))
    H2 = qt.Qobj(np.diag(eval2_trunc))
    n_theta1_qobj = qt.Qobj(n_theta1_trunc)
    n_theta2_qobj = qt.Qobj(n_theta2_trunc)
    H_tot = qt.tensor(H1, qt.qeye(len(hspace_2))) + qt.tensor(qt.qeye(len(hspace_1)), H2) + \
        g * qt.tensor(n_theta1_qobj, qt.qeye(len(hspace_2))) * qt.tensor(qt.qeye(len(hspace_1)),n_theta2_qobj)
    evals_tot, evecs_tot = ssp.linalg.eigsh(H_tot.data, k=params["truc_total"],  which='SA')
    evecs_tot = np.array(normalize_eigenvector_phases(evecs_tot.T))

    # Construct bare/dressed basis for full system
    hspace_idx_bare = np.array(np.unravel_index(np.arange(len(hspace_1)*len(hspace_2)), (len(hspace_1), len(hspace_2)))).T
    result = [find_overlap_complex(ket, hspace_idx_bare) for ket in evecs_tot.T]
    top_overlap = [res[1] for res in result]
    top_idx = []
    top_idx_mapped = []
    for res in result:
        mapped_pairs = []
        pairs = []
        for idx_pair in res[0]:
            mapped_pairs.append((hspace_1[idx_pair[0]], hspace_2[idx_pair[1]]))
            pairs.append(idx_pair)
        top_idx.append(pairs)
        top_idx_mapped.append(mapped_pairs)
    hspace_full = ut.get_dressed_states_index(top_idx, hspace_1, hspace_2)
    
    # Save top overlaps and indices (i.e., dressed state decomposition)
    with open(summary_file, 'w') as f:
        print("Dressed states (bottom 10):", file=f)
        for idx, state in enumerate(hspace_full[:10]):
            dressed_str = dressed_state_decomp(top_idx_mapped[idx], top_overlap[idx],
                                              thresh=0.01, float_round=3)

        print("scqubits version:", scq.__version__, file=f)
        print(f"circuit modes: {zp.var_categories}", file=f)
        print(f"circuit params: {zp.symbolic_params}", file=f)
        print("-", file=f)
        print(f"lagrangian (node vars): {sym.nsimplify(zp.sym_lagrangian(return_expr=True))}", file=f)
        print("-", file=f)
        print(f"hamiltonian (transformed vars): {sym.nsimplify(zp.sym_hamiltonian(return_expr=True))}", file=f)
        print("-", file=f)
        print(str(zp), file=f)
    
    # Save n_theta1 and n_theta2 operators in full dressed basis
    n_theta1_dressed = evecs_tot @ qt.tensor(n_theta1_qobj, qt.qeye(len(hspace_2))).data @ evecs_tot.conj().T
    n_theta2_dressed = evecs_tot @ qt.tensor(qt.qeye(len(hspace_1)), n_theta2_qobj).data @ evecs_tot.conj().T

    # # Save data
    # np.savez(str(Path(folder_save, 'two_qubit_data.npz')),
    #             eval1=eval1, eval2=eval2,
    #             n_theta1=n_theta1, n_theta2=n_theta2,
    #             hspace_1=hspace_1, hspace_2=hspace_2,
    #             g_theta1theta2=g,
    #             evals_tot=evals_tot,
    #             evecs_tot=evecs_tot,
    #             n_theta1=
    #             n_theta1_dressed=n_theta1_dressed,
    #             n_theta2_dressed=n_theta2_dressed,
    #             params=params
    #             )


    


def add_2qbt_graph_estimate(params, two_qubit_data):
    """
    Add graph-based truncation estimates to the two_qubit_data dictionary.

    Parameters:
        params (dict): Dictionary containing system parameters.
        two_qubit_data (dict): Dictionary containing two-qubit data, including eigenvalues and operators.

    Returns:
        dict: Updated two_qubit_data dictionary with added graph-based truncation estimates.
    """
    raise NotImplementedError("Function not yet implemented.")


def save_single_qubit_data(params, folder_save):
    """
    Initialize the parameters and operators for the Zero-Pi qubit system.

    Parameters:
        drive_phi (bool): Whether to consider the phi drive term.
        drive_theta (bool): Whether to consider the theta drive term.
        truncation (int): The truncation level for the system eigenstates. Default is 10.
        ncut (int): The number cutoff for charge states. Default is 60.
        phi_cut (int): The number of points in the phi coordinate grid. Default is 200.

    Returns:
        tuple: Contains the following elements:
            - H0 (qutip.Qobj): The Hamiltonian of the system.
            - drive_term (qutip.Qobj): The drive term operator.
            - w_trans_1 (float): Transition frequency between states 0 and 9 or 0 and 7 (based on drive type).
            - w_trans_2 (float): Transition frequency between states 2 and 9 or 2 and 7 (based on drive type).
            - hspace_charge (list): List of indices representing the significant Hilbert space for the charge basis.
    """
    # Define system parameters (in GHz)
    EL = params["EL"]  # Inductive energy
    EJ = params["EJ"]  # Josephson energy
    EC_phi = params["EC_phi"]  # Phi mode charging energy
    EC_theta = params["EC_theta"]  # Theta mode charging energy

    # Compute derived parameters
    E_CJ = 2 * EC_phi
    E_C = 2 / (1 / EC_theta - 1 / EC_phi)

    # Create the grid for the phi coordinate
    phi_grid = scq.Grid1d(params["phi_range"][0], params["phi_range"][1], params["phi_cut"])

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


    summary_file = str(Path(folder_save, 'single_qubit_data_summary.txt'))
    with open(summary_file, 'w') as f:
        print("Single Qubit Data Summary:", file=f)
        print("scqubits version:", scq.__version__, file=f)
        print(f"circuit modes: {zero_pi.var_categories}", file=f)
        print(f"circuit params: {zero_pi.symbolic_params}", file=f)
        print(f"lagrangian: {zero_pi.sym_lagrangian(return_expr=True)}", file=f)
        print(f"hamiltonian: {zero_pi.sym_hamiltonian(return_expr=True)}", file=f)

    # Compute matrix elements for the theta and phi operators
    n_theta =  zero_pi.matrixelement_table(operator="n_theta_operator", evals_count=params["truc"])
    n_phi =  zero_pi.matrixelement_table(operator="i_d_dphi_operator", evals_count=params["truc"])

    # Compute the eigenvalues and construct the Hamiltonian
    # By default, don't shift eigenvalues until loading
    evals, evecs = zero_pi.eigensys(evals_count=params["truc"])
    evals = evals  # Convert to angular frequency (rad/s)
    evecs = evecs.T  # Transpose to have eigenvectors as rows

    # Initialize drive term and transition frequencies based on drive type
    phi_drive = {}
    phi_drive["w_trans_1"] = evals[9] - evals[0]
    phi_drive["w_trans_2"] = evals[9] - evals[2]
    phi_drive["drive_term"] = n_phi
    theta_drive = {}
    theta_drive["w_trans_1"] = evals[7] - evals[0]
    theta_drive["w_trans_2"] = evals[7] - evals[2]
    theta_drive["drive_term"] = n_theta
    mixed_drive = {}
    mixed_drive["w_trans_1"] = evals[9] - evals[0]
    mixed_drive["w_trans_2"] = evals[9] - evals[2]
    mixed_drive["drive_term"] = 0.976 * n_phi + 0.024 * n_theta

    # Determine the significant Hilbert space for the charge basis
    logical_states = [0, 2]  # Start with the ground and first excited states
    for drive in [phi_drive, theta_drive, mixed_drive]:
        # hspace based on threshold
        drive["hspace_charge"] = trunc_by_thresh(logical_states, drive["drive_term"], params["charge_thresh"])
        # hspace based on graph estimate
        drive["hspace_graph"] = trunc_by_graph_estimate(logical_states, drive["drive_term"], evals,
                                                        [drive["w_trans_1"], drive["w_trans_2"]], A=[params["A"], params["A"]])
          
    # Save to npz file
    np.savez(str(Path(folder_save, 'single_qubit_data.npz')),
                evals=evals,
                n_theta=n_theta,
                n_phi=n_phi,
                phi_drive=phi_drive,
                theta_drive=theta_drive,
                mixed_drive=mixed_drive,
                params=params,
                evecs=evecs
                )
    
    # Print single qubit data summary
    summary_file = str(Path(folder_save, 'single_qubit_data_summary.txt'))
    with open(summary_file, 'w') as f:
        print(f"First 10 Eigenvalues (GHz): {evals[:10]}", file=f)
        print(f"n_theta matrix elements for bottom 10 states (GHz):\n {np.round(n_theta[:10, :10], 3)}", file=f)


def load_single_qubit_data(folder_load, drive_type='theta', return_full=False,
                           mult_by_2pi=True):
    """
    Load the single qubit data from the specified folder.

    Parameters:
        folder_load (str): The folder path where the single qubit data is stored.
        drive_type (str): The type of drive to consider ('phi', 'theta', or 'mixed'). Default is 'theta'.

    Returns:
        dict: A dictionary containing the loaded single qubit data.
    """
    data = np.load(Path(folder_load, 'single_qubit_data.npz'), allow_pickle=True)
    single_qubit_data = {
        'evals': data['evals'],
        'n_theta': data['n_theta'],
        'n_phi': data['n_phi'],
        'phi_drive': data['phi_drive'].item(),
        'theta_drive': data['theta_drive'].item(),
        'mixed_drive': data['mixed_drive'].item()
    }

    if return_full:
        return single_qubit_data
    elif drive_type == 'phi':
        drive = single_qubit_data['phi_drive']
    elif drive_type == 'theta':
        drive = single_qubit_data['theta_drive']
    elif drive_type == 'mixed':
        drive = single_qubit_data['mixed_drive']
    else:
        raise ValueError("Invalid drive_type. Choose from 'phi', 'theta', or 'mixed'.")
    
    # Shift the eigenvalues so the ground state energy is zero
    evals = single_qubit_data['evals'].copy()
    evals -= evals[0]
    H0 = qt.Qobj(np.diag(evals))

    if mult_by_2pi:
        return (2*np.pi*H0, 2*np.pi*drive["drive_term"],
                2*np.pi*drive["w_trans_1"], 2*np.pi*drive["w_trans_2"],
                drive["hspace_charge"])
    else:
        return (H0, drive["drive_term"],
                drive["w_trans_1"], drive["w_trans_2"],
                drive["hspace_charge"])


def load_params(file_path='params.yaml'):
    """
    Load parameters from the YAML configuration file.
    
    Parameters:
        file_path (str): Path to the YAML parameter file
        
    Returns:
        dict: Dictionary containing all parameters
    """
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        
        # Convert the transform_2zeropi list to numpy array
        params['Ztransform_2zeropi'] = np.array(params['Ztransform_2zeropi'])
            
        # Convert phi_range list to numpy array if it exists
        params['phi_range'] = np.array(params['phi_range'])
            
        return params
        
    except FileNotFoundError:
        print(f"Error: Could not find parameter file '{file_path}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None



if __name__ == "__main__":
    #### Time of run
    from time import time
    from datetime import datetime

    now = datetime.now()
    formatted_time = now.strftime(r"%B_%d_%Y_%H_%M_%S")


    #### Load parameters from YAML file
    import argparse
    parser = argparse.ArgumentParser(description="Generate and save Hamiltonian data for the Zero-Pi two qubit system.")
    parser.add_argument('-yaml', type=str, default="params.yaml", help="Path to the YAML parameter file.")
    parser.add_argument('-out', type=str, default=formatted_time, help="Folder to save the data within paths.DATA_PATH.")
    args = parser.parse_args()  
    yml_path = args.yaml
    params = load_params(yml_path)

    #### Verify data folder exists or make it
    Path(DATA_FOLDER, args.out).mkdir(parents=True, exist_ok=True)
    print("-------- Parameters loaded from:", yml_path, "--------")
    print("Data will be saved in folder:", Path(DATA_FOLDER, args.out))

    #### Copy Params YAML to data folder
    import shutil
    shutil.copy(yml_path, Path(DATA_FOLDER, args.out, '_params.yaml'))

    #### Generate Single Qubit Data
    start_time = time()
    # print("Generating single qubit data...")
    # save_single_qubit_data(params, folder_save=Path(DATA_FOLDER, args.out))
    # print("Single qubit data saved. Time taken: {:.2f} seconds".format(time() - start_time))

    #### Generate Two Qubit Data
    start_time = time()
    print("Generating two qubit data...")
    save_two_qubit_data(params, folder_save=Path(DATA_FOLDER, args.out))
    print("Two qubit data saved. Time taken: {:.2f} seconds".format(time() - start_time))


