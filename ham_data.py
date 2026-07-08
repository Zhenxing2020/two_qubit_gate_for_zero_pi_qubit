
import scqubits as scq
import numpy as np
import qutip as qt
import scipy.sparse as ssp
import sympy as sym
import yaml
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('../')

from paths import DATA_FOLDER, RESULTS_FOLDER
from truncation_estimate import trunc_by_thresh, trunc_by_graph_estimate
import utils_2Q_gate_zp as ut


def find_overlap_complex(eket, hspace_bare, num=10):
    """
    Find the top contributing bare states to an eigenstate.
    
    Parameters
    ----------
    eket : qutip.Qobj or array_like
        Eigenstate vector to analyze (will be flattened if 2D).
    hspace_bare : list
        Flat list of bare state labels corresponding to eigenstate components.
    num : int, optional
        Number of top contributors to return, by default 10.
    
    Returns
    -------
    tuple[list, array_like]
        Top bare state labels and their complex overlap values,
        ordered by decreasing absolute magnitude.
    """
    # Flatten the 2D array if needed
    if isinstance(eket, qt.Qobj):
        eket = eket.data
    if len(eket.shape) > 1:
        eket = eket.flatten()
    top_indices_flat = np.argsort(-np.abs(eket))[:num]
    hspace_select = [tuple(hspace_bare[i]) for i in top_indices_flat]
    
    return hspace_select, eket[top_indices_flat]


def dressed_state_decomp(top_states, top_values, thresh=0.001, float_round=3, dressed_state_label=None,
                         latex=True):
    """
    Generate a string representation of a dressed state in terms of bare eigenstates.
    
    Parameters
    ----------
    top_states : list
        List of tuples representing bare state indices (i, j), ordered by overlap magnitude.
    top_values : array_like
        Complex overlap values corresponding to top_states, ordered by magnitude.
    thresh : float, optional
        Threshold for overlap magnitude squared to include in output, by default 0.001.
    float_round : int, optional
        Number of digits to round coefficients to, by default 3.
    dressed_state_label : str, optional
        Label for the dressed state. If None, uses "dressed", by default None.
    latex : bool, optional
        Whether to format output as LaTeX, by default True.
    
    Returns
    -------
    str
        String representation of the dressed state decomposition.
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
                str_repr += f"{np.round(overlap_val, float_round)}|{i},{j}⟩ + "
    
    # Remove the trailing " + " and close the equation
    if str_repr.endswith(" + "):
        str_repr = str_repr[:-3]
    
    if latex:
        str_repr += r"$"
    return str_repr

def get_dressed_states_index(top_index, hspace_0, hspace_1, n=None):
    """
    Generate string labels for dressed states based on bare state indices.
    
    Parameters
    ----------
    top_index : list
        List of index pairs (tuples) for each eigenstate, representing contributions 
        from bare states.
    hspace_0 : list or array_like
        State labels/indices for the first qubit.
    hspace_1 : list or array_like  
        State labels/indices for the second qubit.
    n : int, optional
        Maximum number of states to process. If None, processes all states.
    
    Returns
    -------
    list
        List of string labels for dressed states in format "state0-state1".
    """
    index_array = []
    if not n is None:
        n = min(n, len(top_index))
        top_idx = top_index[:n]
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


# def normalize_eigenvector_phases(eigenvectors):
#     """
#     Normalize the phases of eigenvectors to ensure consistent phase convention.
    
#     The phase of each eigenvector is normalized so that the element with the 
#     largest magnitude in the first half of the vector has zero phase (is real 
#     and positive).
    
#     Parameters
#     ----------
#     eigenvectors : list
#         List of eigenvectors, can be numpy arrays or qutip Qobj objects.
    
#     Returns
#     -------
#     list
#         List of phase-normalized eigenvectors with unit norm, in the same 
#         format as the input.
#     """
#     normalized_evecs = []
    
#     for evec in eigenvectors:
#         # Handle both qutip Qobj and numpy array inputs
#         if isinstance(evec, qt.Qobj):
#             data = evec.data.toarray().flatten()
#             is_qobj = True
#             original_shape = evec.shape
#         else:
#             data = np.array(evec).flatten()
#             is_qobj = False
#             original_shape = np.array(evec).shape
        
#         # Find the phase correction based on the chosen method
#         # Find element with largest magnitude
#         half = data.size//2
#         if data.size % 2 == 1:
#             half += 1
#         max_idx = np.argmax(np.abs(data)[:half])
#         phase_element = data[max_idx]
        
#         # Calculate phase correction
#         phase_correction = np.exp(-1j * np.angle(phase_element))
        
#         # Apply phase correction
#         normalized_data = data * phase_correction
        
#         # Reshape back to original format
#         if is_qobj:
#             if len(original_shape) == 2:
#                 normalized_data = normalized_data.reshape(original_shape)
#                 normalized_evec = qt.Qobj(normalized_data)
#             else:
#                 normalized_evec = qt.Qobj(normalized_data)
#         else:
#             normalized_evec = normalized_data.reshape(original_shape)
        
#         # Add phase and normalize to unit norm
#         normalized_evecs.append(normalized_evec/np.linalg.norm(normalized_evec))
    
#     return normalized_evecs


def normalize_eigenvector_phases(eigenvectors):
    """
    Normalize the phases of eigenvectors to ensure consistent phase convention.
    """
    normalized_evecs = []
    
    for evec in eigenvectors:
        if isinstance(evec, qt.Qobj):
            data = evec.data.toarray().flatten()
            is_qobj = True
            original_shape = evec.shape
        else:
            data = np.array(evec).flatten()
            is_qobj = False
            original_shape = np.array(evec).shape
        
        # use first half as phase anchor
        half = data.size // 2
        if data.size % 2 == 1:
            half += 1

        max_idx = np.argmax(np.abs(data)[:half])
        phase_element = complex(data[max_idx])   # ⭐ 关键
        
        if np.abs(phase_element) == 0:
            phase_correction = 1.0 + 0j
        else:
            phase_correction = np.exp(-1j * np.angle(phase_element))
        
        normalized_data = data * phase_correction
        
        if is_qobj:
            normalized_data = normalized_data.reshape(original_shape)
            normalized_evec = qt.Qobj(normalized_data)
        else:
            normalized_evec = normalized_data.reshape(original_shape)
        
        normalized_evecs.append(
            normalized_evec / np.linalg.norm(normalized_evec)
        )
    
    return normalized_evecs


def _build_circuit(params, summary_file):
    """
    Build and configure the two-qubit Circuit for the given params, write the
    circuit-description header to summary_file, and extract the theta1-theta2
    coupling strength g_theta1theta2.

    Must be re-run for every distinct set of params (e.g. every point in a
    parameter sweep), since it re-derives the full capacitance matrix.

    Returns
    -------
    zp : scq.Circuit
        Configured circuit with system_hierarchy/subsystem_trunc_dims set.
    g : complex
        Coupling strength g_theta1theta2.
    """
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

    with open(summary_file, 'w') as f:
        print("Two Qubit Data Summary:", file=f)
        print("scqubits version:", scq.__version__, file=f)
        print(f"circuit modes: {zp.var_categories}", file=f)
        print(f"circuit params: {zp.symbolic_params}", file=f)
        print("-", file=f)
        print(f"lagrangian (node vars): {zp.sym_lagrangian(return_expr=True)}", file=f)
        print("-", file=f)
        print(f"hamiltonian (transformed vars): {zp.sym_hamiltonian(return_expr=True)}", file=f)
        print("-", file=f)
        print(str(zp), file=f)

    # Extract g_theta1theta2 coupling strength
    i_for_inv = []
    ith1 = None
    ith2 = None
    for i in range(params["Ztransform_2zeropi"].shape[0]):
        if i + 1 in zp.var_categories["periodic"] + zp.var_categories["extended"] + zp.var_categories["free"]:
            i_for_inv.append(i)
            if i == params["theta_mode1"]:
                ith1 = len(i_for_inv)-1
            if i == params["theta_mode2"]:
                ith2 = len(i_for_inv)-1
    Z = params["Ztransform_2zeropi"]
    cMat = zp.symbolic_circuit._capacitance_matrix(substitute_params=True)
    cTransInv = np.linalg.inv(ut.truncate_2(Z.T @ cMat @ Z, i_for_inv).full())
    g = cTransInv[ith1, ith2]

    with open(summary_file, 'a') as f:
        print(f"Coupling strength g_theta1theta2: {g}", file=f)
    print("Saved coupling strength.")

    # Define subsystems
    system_hierarchy = [[1,3],  [5,7]]  # theta and phi modes for each qubit
    subsystem_trunc_dims = [params["truc1"], params["truc2"]]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)

    print("Circuit and parameters set. Beginning calculations...")

    return zp, g


def _compute_subsystem1(zp, params):
    """
    Diagonalize the qubit-1 subsystem and build its bare n_theta1 operator.

    NOTE: qubit-1 data is NOT invariant across a sweep over qubit-2 parameters
    such as EC2. Although qubit 1's branches never reference qubit-2 params
    directly, scqubits derives each subsystem Hamiltonian from the inverse of
    the *full* capacitance matrix, and that inversion mixes EC2 into qubit-1's
    charging-energy block. Empirically, sweeping EC2 by +30% shifts eval1 by
    ~MHz and changes n_theta1 matrix elements substantially. So this must be
    recomputed at every sweep point (it is cheap to store: ~O(truc1^2)).

    Returns
    -------
    dict with keys: eval1, evecs1, n_theta1, hspace_1_charge
    """
    eval1, evecs1 = zp.subsystems[0].eigensys(params["truc1"])
    print("Finished calculating subsystem 1 eigensystem.")
    print(f"evecs1.shape = {evecs1.shape}")
    print(f"eval1.shape = {eval1.shape}")

    evecs1 = np.array(normalize_eigenvector_phases(evecs1.T))

    # ntheta operator in single qubit bare basis
    n_theta1 = (evecs1 @ getattr(zp.subsystems[0], f"n{params['theta_mode1']+1}_operator")() @ evecs1.conj().T)

    # Reduced model for individual qubit
    hspace_1_charge = trunc_by_thresh([0, 2], n_theta1, params["charge_thresh"])

    return dict(eval1=eval1, evecs1=evecs1, n_theta1=n_theta1, hspace_1_charge=hspace_1_charge)


def _compute_sweep_point(zp, g, subsystem1_data, params, summary_file):
    """
    Diagonalize the qubit-2 subsystem, build the full coupled Hamiltonian,
    solve for the full spectrum, and derive dressed-state quantities.

    Appends the energy-level/dressed-state sections to summary_file.

    Returns
    -------
    dict with keys: eval2, evecs2, n_theta2, hspace_2_charge, evals_tot,
        evecs_tot, hspace_full, top_idx, top_overlap, n_theta1_dressed,
        n_theta2_dressed, hspace_n_theta1, hspace_n_theta2
    """
    eval1 = subsystem1_data["eval1"]
    n_theta1 = subsystem1_data["n_theta1"]
    hspace_1_charge = subsystem1_data["hspace_1_charge"]

    ######################################################################################
    # Calculate subsystem 2 eigenvalues/vectors
    eval2, evecs2 = zp.subsystems[1].eigensys(params["truc2"])
    print("Finished calculating subsystem 2 eigensystem.")
    print(f"evecs2.shape = {evecs2.shape}")
    print(f"eval2.shape = {eval2.shape}")

    evecs2 = np.array(normalize_eigenvector_phases(evecs2.T))

    # ntheta operator in single qubit bare basis
    n_theta2 = (evecs2 @ getattr(zp.subsystems[1], f"n{params['theta_mode2']+1}_operator")() @ evecs2.conj().T)

    # Reduced model for individual qubit
    hspace_2_charge = trunc_by_thresh([0, 2], n_theta2, params["charge_thresh"])

    if params["trunc_before_tensor"]:
        hspace_1 = hspace_1_charge
        hspace_2 = hspace_2_charge
    else:
        hspace_1 = np.arange(params["truc1"])
        hspace_2 = np.arange(params["truc2"])

    # n_theta and n_phi operators for coupling
    n_theta1_trunc = ut.truncate_2(n_theta1, hspace_1).full()
    n_theta2_trunc = ut.truncate_2(n_theta2, hspace_2).full()
    eval1_trunc = eval1[hspace_1]
    eval2_trunc = eval2[hspace_2]

    # Chop off small elements for smaller matrix
    n_theta1_trunc[np.abs(n_theta1_trunc) < params["n_theta_clip"]] = 0
    n_theta2_trunc[np.abs(n_theta2_trunc) < params["n_theta_clip"]] = 0
    H1 = qt.Qobj(np.diag(eval1_trunc))
    H2 = qt.Qobj(np.diag(eval2_trunc))
    n_theta1_qobj = qt.Qobj(n_theta1_trunc)
    n_theta2_qobj = qt.Qobj(n_theta2_trunc)

    # Full System Hamiltonian
    H_tot = qt.tensor(H1, qt.qeye(len(hspace_2))) + qt.tensor(qt.qeye(len(hspace_1)), H2) + \
             g * qt.tensor(n_theta1_qobj, n_theta2_qobj)
    evals_tot, evecs_tot = ssp.linalg.eigsh(H_tot.data, k=params["truc_total"],  which='SA')
    
    ######################################################################################
    sorted_idx = np.argsort(evals_tot)
    evals_tot = evals_tot[sorted_idx]
    evals_tot = evals_tot - evals_tot[0]
    # evecs_tot = ssp.csr_matrix([evecs_tot[:,idx] for idx in sorted_idx])
    
    evecs_tot = np.column_stack(
        [evecs_tot[:, idx] for idx in sorted_idx]
    ).astype(complex)    
    print("dtype after reorder:", evecs_tot.dtype)
    # evecs_tot = np.array(normalize_eigenvector_phases(evecs_tot.T).astype(np.complex128))
    evecs_tot = np.array(normalize_eigenvector_phases(evecs_tot.T)) 
    
    ######################################################################################

    print("Finished calculating full system eigensystem.")

    # Construct bare/dressed basis for full system
    hspace_idx_bare = np.array(np.unravel_index(np.arange(len(hspace_1)*len(hspace_2)), (len(hspace_1), len(hspace_2)))).T
    result = [find_overlap_complex(ket, hspace_idx_bare, num=30) for ket in evecs_tot]
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
    hspace_full = get_dressed_states_index(top_idx, hspace_1, hspace_2, params["truc_total"])

    # Apply any manual label swaps
    if "manual_label_swap" in params:
        swap_dict = params["manual_label_swap"]
        hspace_full = [swap_dict.get(label, label) for label in hspace_full]
    
    # Save top overlaps and indices (i.e., dressed state decomposition)
    with open(summary_file, 'a') as f:
        eval_zero = evals_tot - evals_tot[0]
        print("Important Energy Levels:", file=f)
        print("-- logical states --", file=f)
        ket00 = "|0,0⟩"
        for state in ["0-0", "2-0", "0-2", "2-2"]:
            idx = hspace_full.index(state)
            i,j=state.split("-")
            dressed_str = f"|{i},{j}⟩"
            print(f"{ket00} ↔ {dressed_str}: {np.round(eval_zero[idx], 3)} GHz", file=f)
        print("-- X gate --", file=f)
        x_states = [("0-0", "0-8"), ("0-2", "0-8"),
                       ("2-0", "2-8"), ("2-2", "2-8")]
        x_states += [(x[0][::-1], x[1][::-1]) for x in x_states]
        for s1, s2 in x_states:
            idx1 = hspace_full.index(s1)
            idx2 = hspace_full.index(s2)
            i1,j1=s1.split("-")
            i2,j2=s2.split("-")
            ket1 = f"|{i1},{j1}⟩"
            ket2 = f"|{i2},{j2}⟩"
            try:
                print(f"{ket1} ↔ {ket2}: {np.abs(np.round(eval_zero[idx1] - eval_zero[idx2], 3))} GHz", file=f)
            except Exception as e:
                # Do NOT drop into a debugger here -- under nohup/joblib that
                # hangs or crashes the whole run. Just note the missing state.
                print(f"{ket1} ↔ {ket2}: (unavailable: {e})", file=f)
    
        print("-- CZ gate --", file=f)
        for s1, s2 in [("2-2", "2-5"), ("0-2", "0-5"),
                       ("2-0", "2-1"), ("0-0","0-1"), 
                       ("2-2","5-2"), ("2-0","5-0"),
                       ("0-2","1-2"), ("0-0","1-0")]:
            idx1 = hspace_full.index(s1)
            idx2 = hspace_full.index(s2)
            i1,j1=s1.split("-")
            i2,j2=s2.split("-")
            ket1 = f"|{i1},{j1}⟩"
            ket2 = f"|{i2},{j2}⟩"
            print(f"{ket1} ↔ {ket2}: {np.abs(np.round(eval_zero[idx1] - eval_zero[idx2], 3))} GHz", file=f)
    
        print("-- CNOT gate --", file=f)
        for s1, s2 in [("2-0", "1-4"), ("2-2","8-2"), ("2-0", "8-0"), 
                       ("0-0","1-4"), ("0-2","8-2"), ("0-0","8-0")]:
            idx1 = hspace_full.index(s1)
            idx2 = hspace_full.index(s2)
            i1,j1=s1.split("-")
            i2,j2=s2.split("-")
            ket1 = f"|{i1},{j1}⟩"
            ket2 = f"|{i2},{j2}⟩"
            print(f"{ket1} ↔ {ket2}: {np.abs(np.round(eval_zero[idx1] - eval_zero[idx2], 3))} GHz", file=f)
    
    # Save top overlaps and indices (i.e., dressed state decomposition)
    with open(summary_file, 'a') as f:
        print("Important Dressed states:", file=f)
        print("-- logical states --", file=f)
        for state in ["0-0", "2-0", "0-2", "2-2"]:
            idx = hspace_full.index(state)
            dressed_str = dressed_state_decomp(top_idx_mapped[idx], top_overlap[idx],
                                                thresh=0.001, float_round=3,
                                                dressed_state_label=state.replace("-",","),
                                                latex=False)
            print(dressed_str, file=f)
        print("-- X gate states --", file=f)
        for state in ["0-8", "2-8", "8-0", "8-2"]:
            idx = hspace_full.index(state)
            dressed_str = dressed_state_decomp(top_idx_mapped[idx], top_overlap[idx],
                                                thresh=0.001, float_round=3,
                                                dressed_state_label=state.replace("-",","),
                                                latex=False)
            print(dressed_str, file=f)
        print("-- CZ gate states --", file=f)
        for state in ["0-5", "2-5", "2-1", "0-1", "5-2", "5-0", "1-2", "1-0"]:
            idx = hspace_full.index(state)
            dressed_str = dressed_state_decomp(top_idx_mapped[idx], top_overlap[idx],
                                                thresh=0.001, float_round=3,
                                                dressed_state_label=state.replace("-",","),
                                                latex=False)
            print(dressed_str, file=f)
        print("-- CNOT gate states --", file=f)
        for state in ["8-2", "1-4", "8-0"]:
            idx = hspace_full.index(state)
            dressed_str = dressed_state_decomp(top_idx_mapped[idx], top_overlap[idx],
                                                thresh=0.001, float_round=3,
                                                dressed_state_label=state.replace("-",","),
                                                latex=False)
            print(dressed_str, file=f)
    
    # Save n_theta1 and n_theta2 operators in full dressed basis
    n_theta1_dressed = evecs_tot @ qt.tensor(n_theta1_qobj, qt.qeye(len(hspace_2))).data @ evecs_tot.conj().T
    n_theta2_dressed = evecs_tot @ qt.tensor(qt.qeye(len(hspace_1)), n_theta2_qobj).data @ evecs_tot.conj().T

    # Create hspace_charge for each
    logi_idx = [hspace_full.index(state) for state in ["0-0", "2-0", "0-2", "2-2"]]
    hspace_n_theta1 = trunc_by_thresh(logi_idx, n_theta1_dressed, params["charge_thresh"])
    hspace_n_theta2 = trunc_by_thresh(logi_idx, n_theta2_dressed, params["charge_thresh"])

    # Save reduced hspace info
    with open(summary_file, 'a') as f:
        print("-- hspace charge info --", file=f)
        print(f"hspace1_charge (single qubit) size : {len(hspace_1_charge)}", file=f)
        print(f"hspace2_charge (single qubit) size : {len(hspace_2_charge)}", file=f)
        print(f"hspace_n_theta1 (two qubit) size: {len(hspace_n_theta1)}", file=f)
        print(f"hspace_n_theta2 (two qubit) size: {len(hspace_n_theta2)}", file=f)

    return dict(
        eval2=eval2, evecs2=evecs2,
        n_theta2=n_theta2,
        hspace_2_charge=hspace_2_charge,
        evals_tot=evals_tot, evecs_tot=evecs_tot,
        hspace_full=hspace_full,
        n_theta1_dressed=n_theta1_dressed,
        n_theta2_dressed=n_theta2_dressed,
        top_idx=top_idx,
        top_overlap=top_overlap,
        hspace_n_theta1=hspace_n_theta1,
        hspace_n_theta2=hspace_n_theta2,
    )


def save_two_qubit_data(params, folder_save, get_flux_derivative=False):
    """
    Generate and save two-qubit quantum system data including
    eigenstates, operators, and Hilbert space information.

    This function performs:
    computes single-qubit subsystems, calculates coupling strength, constructs
    the full Hamiltonian, finds eigenstates, analyzes dressed state decompositions,
    and saves all data.

    NOTE: All eigenvalues and operators are saved without the * 2 pi


    Parameters
    ----------
    params : dict
        Dictionary containing circuit parameters including YAML template,
        transformation matrix, flux values, charge offsets, and analysis thresholds
    folder_save : str or Path
        Directory path where output files will be saved

    Returns
    -------
    None
        Saves two_qubit_data.npz and summary.txt files to folder_save
    """
    summary_file = str(Path(folder_save, 'two_qubit_data_summary.txt'))
    zp, g = _build_circuit(params, summary_file)

    ######################################################################################
    if get_flux_derivative:
        # get flux derivative data
        flux_vec = [0] + list(np.logspace(-6, -5, 5))
        zp0 = zp.subsystems[0].get_spectrum_vs_paramvals(
            "Φ1", flux_vec, evals_count=500, num_cpus=4, subtract_ground=True
        )
        x0 = np.array(zp0.param_vals, dtype=float)
        y0 = np.array(zp0.energy_table, dtype=float)
        dx0 = np.diff(x0)
        dy0 = np.diff(y0, axis=0)
        dydx0 = dy0 / dx0[:, None]
        d2ydx2_0 = np.diff(dydx0, axis=0) / dx0[:-1, None]

        zp1 = zp.subsystems[1].get_spectrum_vs_paramvals(
            "Φ2", flux_vec, evals_count=500, num_cpus=4, subtract_ground=True
        )
        x1 = np.array(zp1.param_vals, dtype=float)
        y1 = np.array(zp1.energy_table, dtype=float)
        dx1 = np.diff(x1)
        dy1 = np.diff(y1, axis=0)
        dydx1 = dy1 / dx1[:, None]
        d2ydx2_1 = np.diff(dydx1, axis=0) / dx1[:-1, None]

        flux_npz_path = Path(folder_save, "flux_derivative_subsystems.npz")
        np.savez(
            flux_npz_path,
            flux_vec=flux_vec,
            y_zp0=y0,
            d2ydx2_zp0=d2ydx2_0,
            y_zp1=y1,
            d2ydx2_zp1=d2ydx2_1,
        )
        return flux_npz_path

    ######################################################################################
    subsystem1_data = _compute_subsystem1(zp, params)
    sweep_point_data = _compute_sweep_point(zp, g, subsystem1_data, params, summary_file)

    # Save data
    np.savez(str(Path(folder_save, 'two_qubit_data.npz')),
                eval1=subsystem1_data["eval1"], eval2=sweep_point_data["eval2"],
                evecs1=subsystem1_data["evecs1"], evecs2=sweep_point_data["evecs2"],
                n_theta1=subsystem1_data["n_theta1"], n_theta2=sweep_point_data["n_theta2"],
                hspace_1_charge=subsystem1_data["hspace_1_charge"],
                hspace_2_charge=sweep_point_data["hspace_2_charge"],
                g_theta1theta2=g,
                evals_tot=sweep_point_data["evals_tot"], evecs_tot=sweep_point_data["evecs_tot"],
                hspace_full=sweep_point_data["hspace_full"],
                n_theta1_dressed=sweep_point_data["n_theta1_dressed"],
                n_theta2_dressed=sweep_point_data["n_theta2_dressed"],
                top_idx=sweep_point_data["top_idx"],
                top_overlap=sweep_point_data["top_overlap"],
                hspace_n_theta1=sweep_point_data["hspace_n_theta1"],
                hspace_n_theta2=sweep_point_data["hspace_n_theta2"],
                params=params
                )


_SWEEP_FIXED_SHAPE_KEYS = ["eval1", "n_theta1", "eval2", "n_theta2", "g_theta1theta2",
                           "evals_tot", "hspace_full", "top_idx", "top_overlap",
                           "n_theta1_dressed", "n_theta2_dressed"]
_SWEEP_RAGGED_KEYS = ["hspace_1_charge", "hspace_2_charge", "hspace_n_theta1", "hspace_n_theta2"]


def _stack_object(arrs):
    """1-D object array where element i is arrs[i] (works for ragged / mixed
    shapes; keeps load_two_qubit_sweep(index=i) returning the i-th point)."""
    out = np.empty(len(arrs), dtype=object)
    for i, a in enumerate(arrs):
        out[i] = a
    return out


def _point_checkpoint_path(folder_save, index):
    """Per-point checkpoint file (one completed sweep point)."""
    return Path(folder_save, "_points", f"point_{index:04d}.npz")


def _checkpoint_ok(ckpt, index, value, rtol=1e-9):
    """True if `ckpt` is a complete checkpoint for this (index, value)."""
    try:
        d = np.load(ckpt, allow_pickle=True)
        if int(d["index"]) != index:
            return False
        if not np.isclose(float(d["value"]), float(value), rtol=rtol, atol=0.0):
            return False
        return all(k in d.files for k in _SWEEP_FIXED_SHAPE_KEYS + _SWEEP_RAGGED_KEYS)
    except Exception:
        return False


def _compute_one_point(params, sweep_param_name, index, value, folder_save):
    """
    Compute one sweep point and write it to a per-point checkpoint file
    (folder_save/_points/point_{index}.npz), then return the index.

    Runs at module scope (not a closure) so joblib can pickle it to workers.
    evecs1/evecs2/evecs_tot are computed internally but excluded, so each
    checkpoint is small (~tens of MB). The checkpoint is written atomically
    (temp file + rename) so a kill mid-write cannot leave a corrupt point; and
    because each point is persisted as it finishes, a later crash (e.g. in the
    collate) or an interrupt never discards completed work -- re-running with
    resume=True skips points whose checkpoint already exists.
    """
    point_params = dict(params)
    point_params[sweep_param_name] = value

    summary_file = str(Path(folder_save, f'summary_{sweep_param_name}={value}.txt'))
    zp, g = _build_circuit(point_params, summary_file)

    subsystem1_data = _compute_subsystem1(zp, point_params)
    point_data = _compute_sweep_point(zp, g, subsystem1_data, point_params, summary_file)

    point_data["g_theta1theta2"] = g
    point_data["eval1"] = subsystem1_data["eval1"]
    point_data["n_theta1"] = subsystem1_data["n_theta1"]
    point_data["hspace_1_charge"] = subsystem1_data["hspace_1_charge"]
    result = {k: point_data[k] for k in _SWEEP_FIXED_SHAPE_KEYS + _SWEEP_RAGGED_KEYS}

    ckpt = _point_checkpoint_path(folder_save, index)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.parent / (ckpt.stem + ".tmp.npz")
    np.savez(str(tmp), index=index, value=value, **result)
    tmp.replace(ckpt)   # atomic on the same filesystem
    return index


def save_two_qubit_sweep(params, sweep_param_name, sweep_values, folder_save,
                         n_jobs=1, inner_max_num_threads=None,
                         resume=True, keep_checkpoints=True):
    """
    Sweep a single parameter (e.g. "EC2") and save the resulting two-qubit data
    for every value in one combined npz.

    Storage strategy: evecs1/evecs2/evecs_tot -- by far the largest arrays and
    only needed to build dissipative jump operators -- are computed but NOT
    saved, since this sweep targets coherent-dynamics analysis. Everything else
    is kept per sweep point. See save_two_qubit_data if the full evecs are
    needed for a single point.

    Every other quantity (including qubit-1 data eval1/n_theta1) is recomputed
    and stored per point, because none of it is invariant when a qubit-2 param
    such as EC2 changes (see _compute_subsystem1). Per-point subsystem-1 data is
    small, so this costs little storage.

    npz layout
    ----------
    saved once   : sweep_param_name, sweep_param_values, params
    swept (axis0): eval1, n_theta1, eval2, n_theta2, g_theta1theta2, evals_tot,
                   hspace_full, top_idx, top_overlap, n_theta1_dressed,
                   n_theta2_dressed  -- stacked along a leading sweep axis
    swept ragged : hspace_1_charge, hspace_2_charge, hspace_n_theta1,
                   hspace_n_theta2  -- dtype=object arrays (lengths vary per
                   point due to threshold-based truncation)

    Checkpointing: each point is written to folder_save/_points/point_XXXX.npz
    as it completes, so an interrupt or a crash in the (long) run or the final
    collate never discards finished work -- re-run with resume=True to continue.

    Parameters
    ----------
    params : dict
        Base parameter dictionary (as passed to save_two_qubit_data); the
        entry named sweep_param_name is overridden for each sweep point.
    sweep_param_name : str
        Key in params to sweep, e.g. "EC2". Must be one of the circuit
        parameters substituted into the YAML template.
    sweep_values : sequence
        Values to substitute for sweep_param_name.
    folder_save : str or Path
        Directory to save two_qubit_sweep_data.npz and per-point summary
        files (summary_{sweep_param_name}={value}.txt) to. Created if absent.
    n_jobs : int, optional
        Number of sweep points to compute in parallel (joblib/loky). Default 1
        (serial, with a tqdm progress bar). Each point is independent and, since
        evecs are dropped, ships back only ~tens of MB, so parallelism is cheap.
    inner_max_num_threads : int or None, optional
        Threads each worker may use for inner BLAS/OpenMP work when n_jobs > 1.
        None (default) inherits the environment (e.g. OMP_NUM_THREADS). Set this
        to give each worker more threads than the env allows -- useful when there
        are fewer sweep points than cores and you want each point to run faster.
        Ignored when n_jobs == 1.
    resume : bool, optional
        If True (default), skip sweep points whose checkpoint already exists in
        folder_save/_points/ (matching index + value), so an interrupted or
        crashed run continues instead of recomputing. Set False to force a fresh
        computation of every point (existing checkpoints are overwritten).
    keep_checkpoints : bool, optional
        If True (default), keep the per-point files in folder_save/_points/ after
        the combined npz is written (they enable resume/re-collate and are safe
        to delete manually once the combined npz is verified). If False, delete
        them after a successful save.

    Returns
    -------
    None
        Saves two_qubit_sweep_data.npz and per-point summary_*.txt files to
        folder_save. Each point is also checkpointed to folder_save/_points/.
    """
    folder_save = Path(folder_save)
    folder_save.mkdir(parents=True, exist_ok=True)
    (folder_save / "_points").mkdir(exist_ok=True)
    sweep_values = list(sweep_values)
    n = len(sweep_values)

    # Resume: compute only points without a valid checkpoint.
    todo = [(i, v) for i, v in enumerate(sweep_values)
            if not (resume and _checkpoint_ok(_point_checkpoint_path(folder_save, i), i, v))]
    if resume and len(todo) < n:
        print(f"[save_two_qubit_sweep] resume: {n - len(todo)}/{n} points already "
              f"checkpointed; computing {len(todo)}.")

    # Each point writes its own checkpoint as it finishes (crash/interrupt safe).
    if n_jobs == 1:
        for i, v in tqdm(todo, desc=f"sweep {sweep_param_name}"):
            _compute_one_point(params, sweep_param_name, i, v, str(folder_save))
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=inner_max_num_threads):
            Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_compute_one_point)(params, sweep_param_name, i, v, str(folder_save))
                for i, v in todo
            )

    # Collate from the per-point checkpoints (in sweep order).
    missing = [i for i in range(n) if not _point_checkpoint_path(folder_save, i).exists()]
    if missing:
        raise RuntimeError(f"missing checkpoints for indices {missing}; cannot collate.")
    collected = {k: [] for k in _SWEEP_FIXED_SHAPE_KEYS + _SWEEP_RAGGED_KEYS}
    for i in range(n):
        d = np.load(_point_checkpoint_path(folder_save, i), allow_pickle=True)
        for k in collected:
            collected[k].append(d[k])

    sweep_arrays = {k: _stack_object(collected[k]) for k in _SWEEP_RAGGED_KEYS}
    # Keys expected to be shape-uniform across points are np.stack'd; but the
    # charge-threshold truncation can occasionally change a shape (e.g. an
    # hspace size shifting by one), so fall back to a dtype=object array (and
    # report which key) rather than crashing at the end of a long sweep.
    for k in _SWEEP_FIXED_SHAPE_KEYS:
        try:
            sweep_arrays[k] = np.stack(collected[k])
        except ValueError:
            shapes = [np.asarray(a).shape for a in collected[k]]
            print(f"[save_two_qubit_sweep] WARNING: key '{k}' has non-uniform shapes "
                  f"across sweep points; storing as dtype=object. Shapes: {shapes}")
            sweep_arrays[k] = _stack_object(collected[k])

    np.savez(str(folder_save / 'two_qubit_sweep_data.npz'),
                sweep_param_name=sweep_param_name,
                sweep_param_values=np.array(sweep_values),
                params=params,
                **sweep_arrays
                )

    if not keep_checkpoints:
        for i in range(n):
            _point_checkpoint_path(folder_save, i).unlink(missing_ok=True)
        for tmp in (folder_save / "_points").glob("*.tmp.npz"):
            tmp.unlink(missing_ok=True)
        try:
            (folder_save / "_points").rmdir()
        except OSError:
            pass


def load_two_qubit_sweep_values(folder_load):
    """
    Return the sweep axis of a two_qubit_sweep_data.npz saved by
    save_two_qubit_sweep, so a caller can see which points are available before
    loading one.

    Parameters
    ----------
    folder_load : str or Path
        Directory containing two_qubit_sweep_data.npz.

    Returns
    -------
    tuple[str, numpy.ndarray]
        (sweep_param_name, sweep_param_values) -- e.g. ("EC2", array([...])).
    """
    data = np.load(Path(folder_load, 'two_qubit_sweep_data.npz'), allow_pickle=True)
    return str(data['sweep_param_name']), np.array(data['sweep_param_values'], dtype=float)


def load_two_qubit_sweep(folder_load, value=None, index=None, return_full=False, rtol=1e-6):
    """
    Load a single point of a parameter sweep saved by save_two_qubit_sweep.

    The returned list matches load_two_qubit_data's format exactly (same order,
    same 2*pi scaling), so downstream fidelity/optimization code can consume a
    sweep point wherever it currently consumes a single two_qubit_data.npz --
    with ONE difference: eket_tot is None, because the sweep intentionally drops
    the full eigenvectors (evecs_tot). This is sufficient for coherent-dynamics
    (Schrodinger) work, which only needs eval_tot and the dressed drive
    operators; it is NOT sufficient for building noisy collapse operators, which
    require eket_tot. See load_two_qubit_sweep_values to list available points.

    Parameters
    ----------
    folder_load : str or Path
        Directory containing two_qubit_sweep_data.npz.
    value : float, optional
        Swept parameter value to load. Matched against sweep_param_values with
        np.isclose(rtol=rtol). Exactly one of value/index must be given.
    index : int, optional
        Integer position along the sweep axis. Exactly one of value/index must
        be given.
    return_full : bool, optional
        If True, return (npz_object, index) for the resolved point instead of
        the processed list. Default False.
    rtol : float, optional
        Relative tolerance for matching value against sweep_param_values.

    Returns
    -------
    list or tuple
        If return_full=False: [hspace_full, eket_tot (None), eval_tot, n_theta0,
        n_theta1, n_theta0_dress, n_theta1_dress, hspace_0, hspace_1,
        hspace_n_theta1, hspace_n_theta2, logi_state] for the selected point.
        If return_full=True: (data, index) where data is the raw NpzFile.
    """
    if (value is None) == (index is None):
        raise ValueError("Provide exactly one of value or index.")

    data = np.load(Path(folder_load, 'two_qubit_sweep_data.npz'), allow_pickle=True)
    sweep_values = np.array(data['sweep_param_values'], dtype=float)

    if index is None:
        matches = np.flatnonzero(np.isclose(sweep_values, value, rtol=rtol))
        if len(matches) == 0:
            raise ValueError(
                f"No sweep point near {str(data['sweep_param_name'])}={value}; "
                f"available values: {sweep_values.tolist()}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"value={value} matches multiple sweep points {matches.tolist()} "
                f"(rtol={rtol}); tighten rtol or use index."
            )
        i = int(matches[0])
    else:
        i = int(index)

    if return_full:
        return data, i

    hspace_full = data['hspace_full'][i].tolist()
    eket_tot = None  # evecs_tot not saved in a sweep (coherent-dynamics only)
    eval_tot = 2*np.pi*data['evals_tot'][i]
    n_theta0 = 2*np.pi*data['n_theta1'][i]
    n_theta1 = 2*np.pi*data['n_theta2'][i]
    n_theta0_dress = 2*np.pi*data['n_theta1_dressed'][i]
    n_theta1_dress = 2*np.pi*data['n_theta2_dressed'][i]
    hspace_0 = np.asarray(data['hspace_1_charge'][i]).tolist()
    hspace_1 = np.asarray(data['hspace_2_charge'][i]).tolist()
    hspace_n_theta1 = np.asarray(data['hspace_n_theta1'][i]).tolist()
    hspace_n_theta2 = np.asarray(data['hspace_n_theta2'][i]).tolist()
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    return [hspace_full, eket_tot, eval_tot, n_theta0, n_theta1, n_theta0_dress,
        n_theta1_dress, hspace_0, hspace_1, hspace_n_theta1, hspace_n_theta2, logi_state]


def load_two_qubit_data(folder_load, return_full=False):
    """
    Load two-qubit quantum system data from saved .npz file.

    Parameters
    ----------
    folder_load : str or Path
        Directory path containing the two_qubit_data.npz file
    return_full : bool, optional
        If True, returns the full numpy data object. If False, returns 
        processed data in specific format. Default is False.

    Returns
    -------
    numpy.lib.npyio.NpzFile or list
        If return_full=True: raw numpy data object
        If return_full=False: list containing [hspace_full, eket_tot, eval_tot, 
        n_theta0_dress, n_theta1_dress, hspace_0, hspace_1, logi_state] with * 2 pi
    """

    data = np.load(Path(folder_load, 'two_qubit_data.npz'), allow_pickle=True)
    if return_full:
        return data
    else:
        hspace_full = data['hspace_full'].tolist()
        eket_tot = data['evecs_tot']
        eval_tot = 2*np.pi*data['evals_tot']
        n_theta0 = 2*np.pi*data['n_theta1']
        n_theta1 = 2*np.pi*data['n_theta2']
        n_theta0_dress = 2*np.pi*data['n_theta1_dressed']
        n_theta1_dress = 2*np.pi*data['n_theta2_dressed']
        hspace_0 = data['hspace_1_charge'].tolist()
        hspace_1 = data['hspace_2_charge'].tolist()
        hspace_n_theta1 = data['hspace_n_theta1'].tolist()
        hspace_n_theta2 = data['hspace_n_theta2'].tolist()
        logi_state = ['0-0', '0-2', '2-0', '2-2']
        return [hspace_full, eket_tot, eval_tot, n_theta0, n_theta1, n_theta0_dress, 
            n_theta1_dress, hspace_0, hspace_1, hspace_n_theta1, hspace_n_theta2, logi_state]

def load_1q_data_for_2q(folder_load):
    """
    Load single-qubit data from two-qubit system calculations.

    Extracts eigenvalues and charge operators for individual qubits 
    from the saved two-qubit data file.

    Parameters
    ----------
    folder_load : str or Path
        Directory path containing the two_qubit_data.npz file

    Returns
    -------
    tuple
        (eval0, eval1, n_theta0, n_theta1) - eigenvalues and charge 
        operators for both qubits, scaled by 2π
    """

    data = np.load(Path(folder_load, 'two_qubit_data.npz'), allow_pickle=True)
    eval0 = 2*np.pi*data['eval1']
    eval1 = 2*np.pi*data['eval2']
    n_theta0 = 2*np.pi*data['n_theta1']
    n_theta1 = 2*np.pi*data['n_theta2']

    return eval0,eval1,n_theta0,n_theta1

    
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
    Generate and save single-qubit Zero-Pi quantum system data.

    Initializes a Zero-Pi qubit system, computes eigenvalues and operators,
    calculates drive terms and transition frequencies for different drive types
    (phi, theta, mixed), and saves comprehensive data and summary files.

    Parameters
    ----------
    params : dict
        Dictionary containing Zero-Pi qubit parameters including energies,
        grid parameters, truncation levels, and drive amplitudes
    folder_save : str or Path
        Directory path where output files will be saved

    Returns
    -------
    None
        Saves single_qubit_data.npz and single_qubit_data_summary.txt files
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
    with open(summary_file, 'a') as f:
        print(f"First 10 Eigenvalues (GHz): {evals[:10]}", file=f)
        print(f"hspace theta size: {len(theta_drive['hspace_charge'])}", file=f)
        print(f"hspace phi size: {len(phi_drive['hspace_charge'])}", file=f)
        print(f"hspace mixed size: {len(mixed_drive['hspace_charge'])}", file=f)
        print(f"n_theta matrix elements for bottom 10 states (GHz):\n {np.round(n_theta[:10, :10], 3)}", file=f)


def load_single_qubit_data(folder_load, drive_type='theta', return_full=False,
                           mult_by_2pi=True):
    """
    Load single-qubit Zero-Pi quantum system data from saved files.

    Loads data from single_qubit_data.npz file and extracts specific drive
    type information or returns complete dataset based on parameters.

    Parameters
    ----------
    folder_load : str or Path
        Directory path containing the single_qubit_data.npz file
    drive_type : str, optional
        Type of drive to extract ('phi', 'theta', or 'mixed'). Default is 'theta'
    return_full : bool, optional
        If True, returns complete dataset. If False, returns processed data
        for specified drive type. Default is False
    mult_by_2pi : bool, optional
        If True, multiplies frequencies by 2π for angular units. Default is True

    Returns
    -------
    dict or tuple
        If return_full=True: complete single_qubit_data dictionary
        If return_full=False: tuple containing (H0, drive_term, w_trans_1, 
        w_trans_2, hspace_charge) for specified drive type
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


def load_params(file_path='params.yaml', convert_lists=True):
    """
    Load and process parameters from YAML configuration file.

    Reads YAML parameter file and converts specific list parameters 
    to numpy arrays for computational use.

    Parameters
    ----------
    file_path : str, optional
        Path to the YAML parameter file. Default is 'params.yaml'

    Returns
    -------
    dict or None
        Dictionary containing all parameters with processed numpy arrays,
        or None if file not found or parsing error occurs

    Notes
    -----
    Automatically converts 'Ztransform_2zeropi' and 'phi_range' 
    list parameters to numpy arrays.
    """
    try:
        with open(file_path, 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        
        if convert_lists:
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
    parser.add_argument('--yaml', type=str, default="params.yaml", help="Path to the YAML parameter file.")
    parser.add_argument('--out', type=str, default=formatted_time, help="Folder to save the data within paths.DATA_PATH.")
    parser.add_argument('--data-type', type=str, choices=['single', 'two', 'both'], default='both', 
                        help="Type of data to generate: 'single' for single qubit only, 'two' for two qubit only, 'both' for both types (default: both)")
    args = parser.parse_args()  
    yml_path = args.yaml
    params = load_params(yml_path)

    #### Verify data folder exists or make it
    Path(DATA_FOLDER, args.out).mkdir(parents=True, exist_ok=True)
    print("-------- Parameters loaded from:", yml_path, "--------")
    print("Data will be saved in folder:", Path(DATA_FOLDER, args.out))

    #### Copy Params YAML to data folder
    import shutil
    shutil.copy(yml_path, Path(DATA_FOLDER, args.out, Path(yml_path).name))
    print("Copied params yaml to data folder.")
    print("-------- Beginning Data Generation --------")
    if args.data_type in ['single', 'both']:
        start_time = time()
        print("Generating single qubit data...")
        save_single_qubit_data(params, folder_save=Path(DATA_FOLDER, args.out))
        print("Single qubit data saved. Time taken: {:.2f} seconds".format(time() - start_time))

    #### Generate Data Based on Flag
    if args.data_type in ['two', 'both']:
        start_time = time()
        print("Generating two qubit data...")
        # save_two_qubit_data(params, folder_save=Path(DATA_FOLDER, args.out))

        save_two_qubit_data(params, folder_save=Path(DATA_FOLDER, args.out), get_flux_derivative=True)
        print("Two qubit data saved. Time taken: {:.2f} seconds".format(time() - start_time))



    
