import sys
sys.path.append('../')

import pandas as pd
import datetime, pytz, os
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import utils_2Q_gate_zp as ut
from datetime import datetime
import numpy as np
import qutip as qt
import scqubits as scq
from joblib import Parallel, delayed
import scipy.sparse as ssp


def generate_ekettot_3ncut(test=True):
    """
    Generate eigenvalues and eigenvectors of a two-qubit coupled Hamiltonian 
    using truncated basis spaces.

    Parameters
    ----------
    test : bool, optional
        If True, use small truncation numbers for quick testing (default: True).
        If False, use larger truncation numbers for production runs.

    Process
    -------
    1. Load single-qubit data (eigenvalues, operators).
    2. Optionally apply charge-matrix-element-based truncation to reduce Hilbert space size.
    3. Construct two-qubit Hamiltonian H = H_bare + g * Hint.
    4. Solve for lowest `truc_tot` eigenvalues/eigenvectors.
    5. Save results to disk (eval_tot, eket_tot).
    """
    # Choose truncation parameters
    if test:
        truc1, truc_tot, charge_pick = 11, 15, True
    else:
        truc1, truc_tot, charge_pick = 300, 1500, True

    g = 0.030961990356445312  # coupling constant
    print(f'truc1, truc_tot, charge_pick = {truc1}, {truc_tot}, {charge_pick}')
    print(f'g={g}')

    # Load single-qubit Hamiltonian diagonalization results
    eval0, eval1, n_theta0, n_theta1 = ut.load_1q_data_for_2q(truc1)

    # --------------------------------------------------------------------------
    # Apply charge truncation to select relevant subspaces
    if charge_pick:
        hspace_0 = ut.get_truncated_subspace_xgate(n_theta0, truc1)
        hspace_1 = ut.get_truncated_subspace_xgate(n_theta1, truc1)

        # Truncate operators and eigenvalues accordingly
        n_theta0 = ut.truncate_2(n_theta0, hspace_0)
        n_theta1 = ut.truncate_2(n_theta1, hspace_1)
        eval0 = eval0[hspace_0]
        eval1 = eval1[hspace_1]

    # --------------------------------------------------------------------------
    # Construct two-qubit Hamiltonian
    print(f'len(hspace_0)={len(hspace_0)}, len(hspace_1)={len(hspace_1)}')
    Hint = qt.tensor(qt.Qobj(n_theta0), qt.Qobj(n_theta1))
    H_bare = (
        qt.tensor(qt.Qobj(np.diag(eval0)), qt.identity(len(hspace_1)))
        + qt.tensor(qt.identity(len(hspace_0)), qt.Qobj(np.diag(eval1)))
    )
    Htot = g * Hint + H_bare

    # Number of eigenpairs to compute
    k = Htot.shape[0] - 1 if truc_tot is None else truc_tot

    # Solve sparse eigenproblem (lowest-energy states)
    eval_tot, eket_tot = ssp.linalg.eigsh(Htot.data, k=k, which='SA', tol=1.e-10)

    # Clean eigenpairs (sort/normalize)
    eval_tot, eket_tot = ut.clean_eval_eket(eval_tot, eket_tot)

    # --------------------------------------------------------------------------
    # Save results
    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    os.mkdir(folder_save)
    pd.DataFrame(eval_tot).to_csv(folder_save + 'eval_tot.txt', sep=',', index=False, header=True)
    np.save(folder_save + 'eket_tot.npy', eket_tot.toarray())
    print(f'np.shape(eket_tot)={np.shape(eket_tot)}')


def generate_nop_3ncut():
    """
    Post-process previously computed eigenpairs to compute dressed operators.

    Process
    -------
    1. Load truncated Hilbert space indices and single-qubit operators.
    2. Load two-qubit eigenvalues and eigenvectors from disk.
    3. Compute overlaps between bare and dressed states.
    4. Determine dressed-state index mapping.
    5. Compute dressed number operators (n_theta0_dress, n_theta1_dress).
    6. Save results to disk.

    Notes
    -----
    Requires that `generate_ekettot_3ncut` has been run first, 
    since it loads its saved data.
    """
    truc1, truc_tot, charge_pick = 300, 2000, True

    # Load truncated Hilbert spaces
    if charge_pick:
        folder = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
        hspace_0 = pd.read_csv(folder + 'hspace_0.txt').to_numpy().flatten()
        hspace_1 = pd.read_csv(folder + 'hspace_1.txt').to_numpy().flatten()
    else:
        hspace_0 = np.arange(truc1)
        hspace_1 = np.arange(truc1)

    n0, n1 = len(hspace_0), len(hspace_1)

    # Load single-qubit number operators
    folder = f'../../data/3ncut_two_zeropi/truc1=500/'
    n_theta0 = np.load(folder + 'n_theta0.npy')
    n_theta1 = np.load(folder + 'n_theta1.npy')

    # Apply truncation
    n_theta0 = ut.truncate_2(n_theta0, hspace_0)
    n_theta1 = ut.truncate_2(n_theta1, hspace_1)

    # Load two-qubit eigenvalues and eigenvectors
    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = pd.read_csv(folder_save + 'eval_tot.txt').to_numpy().flatten()
    eket_tot = ssp.csr_matrix(np.load(folder_save + 'eket_tot.npy'))

    # --------------------------------------------------------------------------
    # Build bare product states
    bare_state = [
        [qt.tensor(qt.basis(n0, i), qt.basis(n1, j)) for j in range(n1)]
        for i in range(n0)
    ]
    arg = [bare_state, n0, n1]

    # Compute overlaps between dressed and bare states in parallel
    print("ut.find_overlap..... Time:")
    ut.print_time()
    result = Parallel(n_jobs=100)(delayed(ut.find_overlap)(i, *arg) for i in eket_tot)

    top_index = [result[i][0] for i in range(eval_tot.shape[0])]
    top_overlap = [result[i][1] for i in range(eval_tot.shape[0])]

    np.save(folder_save + f'top_index.npy', top_index)
    np.save(folder_save + f'top_overlap.npy', top_overlap)
    print("pd.DataFrame(top_overlap)..... Time:")
    ut.print_time()

    # --------------------------------------------------------------------------
    # Compute dressed-state operators
    hspace_full = ut.get_dressed_states_index(top_index, hspace_0, hspace_1)

    # Construct number operators in dressed basis
    n_theta0_dress = ssp.kron(n_theta0, ssp.identity(n1))
    n_theta1_dress = ssp.kron(ssp.identity(n0), n_theta1)
    n_theta0_dress = (eket_tot @ n_theta0_dress @ eket_tot.conj().T).todense()
    n_theta1_dress = (eket_tot @ n_theta1_dress @ eket_tot.conj().T).todense()

    print("folder_save.... Time:")
    ut.print_time()

    # Save results
    pd.DataFrame(hspace_full).to_csv(folder_save + f'hspace_full.txt', sep=',', index=False, header=True)
    np.save(folder_save + f'n_theta0_dress.npy', n_theta0_dress)
    np.save(folder_save + f'n_theta1_dress.npy', n_theta1_dress)

    print(f'len(hspace_full) = {len(hspace_full)}')
    print(f'np.shape(n_theta0_dress) = {np.shape(n_theta0_dress)}')
    print(f'np.shape(n_theta1_dress) = {np.shape(n_theta1_dress)}')


if __name__ == '__main__':
    """
    Entry point of the script:
    - Prints basic environment info.
    - Runs main generation pipeline.
    """
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    # Main workflow
    # generate_ekettot_3ncut(test=True)   # quick test run
    generate_ekettot_3ncut(test=False)    # full run
    # generate_nop_3ncut()                # post-processing
    # other functions: import_para(), generate_data(), generate_data_3ncut(), reduce_eket(), generate_eval()

    ut.print_time()








#################################################################################
### Other previous functions (not used in main workflow)
#################################################################################

# def generate_data():
#     truc1, truc_tot, charge_pick = 300, 300, True
#     n_cut, phi_cut = 90, 300
#     print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
#     args_all = ut.get_operator_two_zeropi_v2(truc1=truc1, truc_tot=truc_tot, charge_pick=charge_pick,
#                                              n_cut=n_cut, phi_cut=phi_cut)
#     [hspace_0, hspace_1, eval_tot, eket_tot] = args_all

#     folder = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
#     os.mkdir(folder)
#     pd.DataFrame(eval_tot).to_csv(folder+ 'eval_tot.txt', sep=',', index=False, header=True)
#     pd.DataFrame(hspace_0).to_csv(folder+ 'hspace_0.txt', sep=',', index=False, header=True)
#     pd.DataFrame(hspace_1).to_csv(folder+ 'hspace_1.txt', sep=',', index=False, header=True)
#     pd.DataFrame(eket_tot.toarray()).to_csv(folder+ 'eket_tot.txt', sep=',', index=False, header=True)

# def reduce_eket():
#     truc1, truc_tot, charge_pick = 300, 1000, False
#     folder_input = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
#     eval_tot = pd.read_csv(folder_input+ 'eval_tot.txt').to_numpy().flatten()
#     eket_tot = pd.read_csv(folder_input+ 'eket_tot.txt').map(complex).to_numpy()

#     print("\n eket_tot = pd.read..... Time:", datetime.now(pytz.timezone('America/Denver')))
#     truc_output = 500
#     eval_tot = eval_tot[:truc_output]
#     eket_tot = eket_tot[:truc_output]

#     folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_output}_pick={charge_pick}/'
#     # os.mkdir(folder_save)
#     pd.DataFrame(eval_tot).to_csv(folder_save+ 'eval_tot.txt', sep=',', index=False, header=True)
#     pd.DataFrame(eket_tot).to_csv(folder_save+ 'eket_tot.txt', sep=',', index=False, header=True)

def generate_eval(Ec0=1.0, truc1=300, n_cut=90, phi_cut=300):
    zp = scq.Circuit(ut.zp_yml, from_file=False)
    zp.Ec0 = Ec0
    zp.configure(transformation_matrix=np.linalg.inv(ut.transform_2zeropi))

    ##############################################################################################
    ### Construct subsystem, calculate eigenvalues
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [10, 10]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)
    zp.Φ1 = 0.001
    zp.Φ2 = 0.001
    zp.cutoff_ext_1, zp.cutoff_ext_5 = phi_cut, phi_cut
    zp.cutoff_n_2, zp.cutoff_n_6 = n_cut, n_cut

    ### the two-line code below takes time when truc1 is large
    eval0, _ = zp.subsystems[0].eigensys(evals_count=truc1)
    eval1, _ = zp.subsystems[1].eigensys(evals_count=truc1)

    sorted_idx0 = np.argsort(eval0)
    eval0 = eval0[sorted_idx0]
    eval0 = eval0 - eval0[0]
    sorted_idx1 = np.argsort(eval1)
    eval1 = eval1[sorted_idx1]
    eval1 = eval1 - eval1[0]
    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_flux=0.001/'
    os.mkdir(folder_save)
    pd.DataFrame(eval0).to_csv(folder_save+ 'eval0.txt', index=False, header=True)
    pd.DataFrame(eval1).to_csv(folder_save+ 'eval1.txt', index=False, header=True)

















