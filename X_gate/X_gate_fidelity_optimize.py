import sys
sys.path.append('../')
from datetime import datetime
import pytz, os
import numpy as np
import scipy as sp
from tqdm import tqdm
import scqubits as scq
import scqubits.settings as settings
import qutip as qt
from multiprocessing import Pool
from joblib import Parallel, delayed
import pandas as pd
# Update scqubits settings
settings.OVERLAP_THRESHOLD = 0.3
max_step, nsteps = 1e-4, 1e4

# truncate operators or states to desired indices
def truncate_2(qobj, indices):
    """
    Truncates a qutip qobj or array
    to a desired list of indices.

    Args:
        qobj (qt.Qobj or np.array): operator or state
        indices (list of int or int): desired indices to keep, or first n indices

    Returns:
        qt.Qobj: truncated state/operator
    """
    if len(qobj.shape) == 2 and qobj.shape[0] == qobj.shape[1]:
        return qt.Qobj(qobj[np.ix_(indices, indices)])
    else:
        return qt.Qobj(qobj[indices])

# Utility function for logging optimization progress
def print_soln(xk, convergence=0):
    """Log the best solution and convergence information during optimization."""
    print("Best Solution:", np.round(xk, 8).tolist())
    print("Convergence:", np.round(convergence, 4))
    print("----------------------------")

# Functions for the pulse for qubits A and B
def drive_gauss_A(t: float, args: dict) -> float:
    """
    Compute the pulse for qubit A.

    Args:
        t (float): Time at which to evaluate the pulse (0 <= t <= gate_time).
        args (dict): Dictionary containing pulse parameters:
            - drive_amp_A (float): Drive amplitude for qubit A.
            - drive_freq_A (float): Drive frequency for qubit A.
            - gate_time (float): Total duration of the gate.

    Returns:
        float: The value of the pulse for qubit A at time `t`.
    """
    A = args.get('drive_amp_A', 0)
    wd = args.get('drive_freq_A', 0)
    tg = args.get('gate_time', 0)
    return A * (np.exp(-8 * t * (t - tg) / tg**2) - 1) * np.cos(wd * t) * (0<=t<=tg)

def drive_gauss_B(t: float, args: dict) -> float:
    """
    Compute the pulse for qubit B.

    Args:
        t (float): Time at which to evaluate the pulse (0 <= t <= gate_time).
        args (dict): Dictionary containing pulse parameters:
            - drive_amp_B (float): Drive amplitude for qubit B.
            - drive_freq_B (float): Drive frequency for qubit B.
            - gate_time (float): Total duration of the gate.

    Returns:
        float: The value of the pulse for qubit B at time `t`.
    """
    A = args.get('drive_amp_B', 0)
    wd = args.get('drive_freq_B', 0)
    tg = args.get('gate_time', 0)
    return A * (np.exp(-8 * t * (t - tg) / tg**2) - 1) * np.cos(wd * t) * (0<=t<=tg)

def parallel_mesolve(n, N, H, tlist, c_op_list, args, options, proj_idx, dims=None):
    """
    Helper function for parallel mesolve execution.
    """
    row_idx, col_idx = proj_idx[n]
    rho0 = qt.states.projection(N, row_idx, col_idx)
    rho0.dims = dims
    output = qt.mesolve(
        H, rho0, tlist, c_ops=c_op_list, args=args, options=options, _safe_mode=False
    )
    return output

# Parallel SESolve function
def sesolve_parallel(argz):
    """Parallel SESolve function for solving the Schrodinger equation."""
    H, psi0, tlist, pulse_args, options = argz
    return psi0, qt.sesolve(H, qt.basis(H[0].shape[0], psi0), tlist, options=options, args=pulse_args)

def get_propagator(H, tlist, num_cpus, parallel, c_op_list, pulse_args, options, logi_state, return_all_t=False):
    """
    Compute the propagator for a quantum system, supporting both noiseless and noisy systems.

    Args:
        H (list or qt.Qobj): The Hamiltonian of the system. Can be a single Qobj or a list where
                             the first element represents the static part.
        tlist (list): List of time points for the simulation.
        num_cpus (int): Number of CPUs to use for parallel computation (if `parallel=True`).
        parallel (bool): Whether to run the computation in parallel.
        c_op_list (list): List of collapse operators for modeling noise. If empty, the system is noiseless.
        pulse_args (dict): Arguments for time-dependent pulse functions in the Hamiltonian.
        options (qt.Options): Solver options for QuTiP.
        logi_state (list): List of logical states (indices of basis states) to include in the propagator.
        return_all_t (bool): Return a propagator for each timestep in tlist

    Returns:
        qt.Qobj or list of qt.Qobj:
            - For noiseless systems: A `Qobj` representing the truncated propagator for logical states.
            - For noisy systems: A `Qobj` representing the superoperator propagator for the final time step.
    """
    H0 = H[0][0] if isinstance(H[0], list) else H[0] if isinstance(H, list) else H
    if len(c_op_list) == 0:
        # Computes the propagator for noiseless systems.
        prop = np.zeros((tlist.shape[0], H0.shape[0], len(logi_state)), dtype=np.complex128)
        for i in logi_state:
            res = qt.sesolve(H, qt.basis(H[0].shape[0], i), tlist, options=options, args=pulse_args)
            for it in range(len(res.states)):
                prop[it, :, logi_state.index(i)] = res.states[it].full().flatten()
        if return_all_t:
            return [truncate_2(qt.Qobj(p), logi_state) for p in prop]
        else:
            return truncate_2(qt.Qobj(prop[-1]), logi_state)
    else:
        # Computes the propagator for noisy systems.
        dimz = len(logi_state)
        proj_idx = [(logi_state[i], logi_state[j]) for j in range(dimz) for i in range(dimz)]
        N = H0.shape[0]
        u = np.zeros([N * N, dimz * dimz, len(tlist)], dtype=complex)

        if parallel:
            output = qt.parallel.parallel_map(
                parallel_mesolve, range(dimz * dimz),
                task_args=(N, H, tlist, c_op_list, pulse_args, options, proj_idx),
                task_kwargs={"dims": H0.dims}, num_cpus=num_cpus
            )
            for n in range(dimz * dimz):
                for k, t in enumerate(tlist):
                    u[:, n, k] = qt.superoperator.mat2vec(output[n].states[k].full()).T
        else:
            for n, idx in enumerate(proj_idx):
                row_idx, col_idx = idx
                rho0 = qt.states.projection(N, row_idx, col_idx)
                rho0.dims = H0.dims
                output = qt.mesolve(
                    H, rho0, tlist, c_op_list, args=pulse_args, options=options, _safe_mode=False
                )
                for k, t in enumerate(tlist):
                    u[:, n, k] = qt.superoperator.mat2vec(output.states[k].full()).T
        if return_all_t:
            return [qt.Qobj(u[:, :, k], dims=[[[N], [N]], [[dimz], [dimz]]]) for k in range(len(tlist))]
        else:
            return [qt.Qobj(u[:, :, k], dims=[[[N], [N]], [[dimz], [dimz]]]) for k in range(len(tlist))][-1]

# Initialize the parameters and operators for the Zero-Pi qubit system.
def zero_pi_initialize(drive_phi, drive_theta, truncation=10, ncut=60, phi_cut=200):
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
    EL = 0.377  # Inductive energy
    EJ = 6.013  # Josephson energy
    EC_phi = 1.142  # Phi mode charging energy
    EC_theta = 0.092  # Theta mode charging energy

    # Compute derived parameters
    E_CJ = 2 * EC_phi
    E_C = 2 / (1 / EC_theta - 1 / EC_phi)

    # Create the grid for the phi coordinate
    phi_grid = scq.Grid1d(-6 * np.pi, 6 * np.pi, phi_cut)

    # Initialize the Zero-Pi qubit system
    zero_pi = scq.ZeroPi(
        grid=phi_grid,
        EJ=EJ,
        EL=EL,
        ECJ=E_CJ,
        EC=E_C,
        dEJ=0.0,
        ng=0.0,
        flux=0.0,
        ncut=ncut,
        truncated_dim=truncation,
    )

    # Compute matrix elements for the theta and phi operators
    n_Theta = zero_pi.matrixelement_table(operator="n_theta_operator", evals_count=truncation)
    n_Phi = zero_pi.matrixelement_table(operator="i_d_dphi_operator", evals_count=truncation)
    n_phi = 2 * np.pi * qt.Qobj(n_Phi)
    n_theta = 2 * np.pi * qt.Qobj(n_Theta)

    # Compute the eigenvalues and construct the Hamiltonian
    evals = 2 * np.pi * zero_pi.eigenvals(evals_count=truncation)
    evals -= evals[0]  # Shift the eigenvalues so the ground state energy is zero
    H0 = qt.Qobj(np.diag(evals))

    # Initialize drive term and transition frequencies based on drive type
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta
    if drive_phi and drive_theta:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = 0.976 * n_phi + 0.024 * n_theta

    # Determine the significant Hilbert space for the charge basis
    thresh = 0.01
    hspace_charge = [0, 2]  # Start with the ground and first excited states
    for s in hspace_charge:
        for i in range(truncation):
            if np.abs(drive_term[s, i] / (2 * np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()

    return H0, drive_term, w_trans_1, w_trans_2, hspace_charge

def get_fidelity_super_operator(super_op, logi_state, gate_target):
    """
    Computes the average gate fidelity for a given superoperator.

    Args:
        s_op (qt.Qobj): The superoperator representing the quantum operation.
        logi_state (list): List of logical states (indices) to consider in the truncated subspace.
        gate_target (qt.Qobj): Target quantum gate to compare against.

    Returns:
        float: The average gate fidelity of the operation.
    """
    kraus = qt.to_kraus(qt.to_super(super_op))
    # print('logi_state=', logi_state)
    # print('p0_kraus=', p0_kraus)
    # print('p0_kraus[0].shape=', np.shape(p0_kraus[0]))
    kraus = [truncate_2(i, logi_state) for i in kraus]
    super_op_post = qt.kraus_to_super(kraus)
    return qt.metrics.average_gate_fidelity(super_op_post, target=gate_target)

# Compute the X-gate fidelity in parallel
def xgate_fidelity_parallel(arg, *args):
    """Compute the X-gate fidelity using parallel processing."""
    [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu] = args
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = arg

    argz = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu,
            tg, drive_amp_A, drive_amp_B, detune_A, detune_B]
    return xgate_fidelity_log(argz)

def xgate_fidelity_log(argz):
    """
    Compute the X-gate fidelity for a noiseless system.

    Args:
        argz (list): A list containing the following parameters:
            - H0 (qt.Qobj): The static Hamiltonian.
            - drive_term (qt.Qobj): The drive Hamiltonian term.
            - w_trans_1 (float): Transition frequency for qubit A.
            - w_trans_2 (float): Transition frequency for qubit B.
            - hilbert_space (list): List of states defining the Hilbert space.
            - n_cpu (int): Number of CPUs for parallelization.
            - tg (float): Gate time.
            - drive_amp_A (float): Drive amplitude for qubit A.
            - drive_amp_B (float): Drive amplitude for qubit B.
            - detune_A (float): Detuning for qubit A.
            - detune_B (float): Detuning for qubit B.

    Returns:
        float: Logarithm of the infidelity for the X-gate.
    """
    [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu,
     tg,drive_amp_A, drive_amp_B, detune_A, detune_B] = argz

    H0_truc = truncate_2(H0, hilbert_space)
    drive_truc = truncate_2(drive_term, hilbert_space)
    H_qbt_drive = [H0_truc, [drive_truc, drive_gauss_A], [drive_truc, drive_gauss_B]]

    pulse_args = {
        'drive_amp_A': drive_amp_A,
        'drive_freq_A': w_trans_1 + 2 * np.pi * detune_A,
        'drive_amp_B': drive_amp_B,
        'drive_freq_B': w_trans_2 + 2 * np.pi * detune_B,
        'gate_time': tg,
    }
    tlist = np.linspace(0, tg, num=int(tg))

    logi_state = [0, 2]
    logi_idx = [hilbert_space.index(s) for s in logi_state]
    c_op_list = []
    parallel = True
    options = qt.Options(nsteps=nsteps)
    Uc = get_propagator(H_qbt_drive, tlist, n_cpu, parallel, c_op_list, pulse_args,
                        options, logi_state=logi_idx)
    fidelity = qt.average_gate_fidelity(Uc, target=qt.sigmax())
    return np.log10(1 - fidelity)

def xgate_fidelity_log_noise(args_indep, *args):
    """
    Computes the X-gate fidelity for a noisy system.

    Args:
        args_indep (list): Independent parameters for the pulse:
            - tg (float): Gate time.
            - drive_amp_A (float): Drive amplitude for qubit A.
            - drive_amp_B (float): Drive amplitude for qubit B.
            - detune_A (float): Detuning for qubit A.
            - detune_B (float): Detuning for qubit B.
        *args: Additional parameters:
            - H_qbt_drive (list): Hamiltonian with drive terms.
            - w_trans_1 (float): Transition frequency for qubit A.
            - w_trans_2 (float): Transition frequency for qubit B.
            - n_cpu (int): Number of CPUs for parallelization.
            - c_op_list (list): Collapse operators for modeling noise.
            - logi_state (list): Logical state indices for truncation.
            - parallel (bool): Whether to enable parallel computation.
            - gate_target (qt.Qobj): Target gate for fidelity comparison.

    Returns:
        float: Logarithm of the infidelity for the X-gate in a noisy system.
    """
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = args_indep
    [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state, parallel, gate_target] = args

    pulse_args = {
        'drive_amp_A': drive_amp_A,
        'drive_freq_A': w_trans_1 + 2 * np.pi * detune_A,
        'drive_amp_B': drive_amp_B,
        'drive_freq_B': w_trans_2 + 2 * np.pi * detune_B,
        'gate_time': tg,
    }

    tlist = np.linspace(0, tg, num=3 * int(tg))
    # options = qt.Options(max_step=max_step, nsteps=nsteps, num_cpus=n_cpu)
    options = qt.Options(num_cpus=n_cpu)
    p_simple_2_a = get_propagator(
        H_qbt_drive, tlist, n_cpu, parallel, c_op_list, pulse_args,
        options=options, logi_state=logi_state
    )
    return np.log10(1 - get_fidelity_super_operator(p_simple_2_a, logi_state, gate_target))

# Optimize fidelity with differential evolution
def fidelity_optimize_x():
    """
    Perform optimization of fidelity using differential evolution.

    The function optimizes gate parameters for high fidelity in a quantum system
    using the `differential_evolution` method from `scipy.optimize`. It also evaluates
    fidelity for various truncations of the Hilbert space and prints results.
    """

    # Drive parameters
    drive_phi, drive_theta = False, True

    # Parameter bounds
    amp1_bounds = (0.1, 0.45)
    amp2_bounds = (0.025, 0.08)
    detune1_bounds = (-0.15, 0.01)
    detune2_bounds = (-0.06, 0.01)
    tg_bound = (0, 0.01)

    # Target gate times
    tg_vec = np.arange(10, 20, step=5).tolist()

    # Optimization parameters
    workers, popsize = 100, 10
    recombination, tol, mutation = 0.7, 0.01, (0.5, 1.0)

    # Truncations
    truc1, truc_full = 30, 50

    # charge and phase basis
    ncut, phi_cut = 30, 100

    print('drive_phi =', drive_phi, ', drive_theta =', drive_theta)
    print('amp1_bounds =', amp1_bounds, ', amp2_bounds =', amp2_bounds, ', tg_bound =', tg_bound)
    print('detune1_bounds =', detune1_bounds, ', detune2_bounds =', detune2_bounds)
    print('workers =', workers, ', popsize =', popsize)
    print('recombination =', recombination, ', tol =', tol, ', mutation =', mutation)

    # Initialize system
    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = zero_pi_initialize(
        drive_phi, drive_theta, truncation=truc1, ncut=ncut, phi_cut=phi_cut
    )
    [H0_full, drive_full, _, _, _] = zero_pi_initialize(
        drive_phi, drive_theta, truncation=truc_full, ncut=ncut, phi_cut=phi_cut
    )
    hspace_full = np.arange(truc_full).tolist()

    print('truc1 =', truc1, ', truc2 (in optimization) =', len(hspace_charge))
    print('truc1_full =', truc_full, ', truc2_full =', len(hspace_full))
#############################################################

    fidelity = []
    drive_param = []
    fidelity_full = []
    n_cpu = 1
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu]

    for jdx, tg in tqdm(enumerate(tg_vec)):
        tg_bounds = (tg + tg_bound[0], tg + tg_bound[1])
        bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)

        # Optimize fidelity using differential evolution
        res = sp.optimize.differential_evolution(
            func=xgate_fidelity_parallel,
            bounds=bounds,
            args=args,
            disp=True,
            callback=print_soln,
            init="sobol",
            workers=workers,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=False,
        )

        fidelity.append(res.fun)
        drive_param.append(res.x)

        # Print optimization results
        print(res, '\n')
        print(f'\ntg = {np.array(tg_vec[:jdx + 1]).tolist()}')
        print(f'\nlog of gate error (truc1={len(hspace_charge)}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i + 4], 8))), ',')

        print(f'\ndrive_param (truc1={len(hspace_charge)}) = ')
        for param in drive_param:
            print(np.round(param, 6).tolist(), ',')

        # Evaluate fidelity for optimal parameters in the full system
        tg, drive_amp_A, drive_amp_B, detune_A, detune_B = drive_param[jdx]

        n_cpu2 = 30
        argz = [
            H0_full, drive_full, w_trans_1, w_trans_2, hspace_full, n_cpu2,
            tg, drive_amp_A, drive_amp_B, detune_A, detune_B
        ]
        fidelity_full.append(xgate_fidelity_log(argz))

        print(f'\nlog of gate error (truc_full={truc_full}) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i + 4], 8))), ',')

        print("\n***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')

    print('amp1_bounds =', amp1_bounds, ', amp2_bounds =', amp2_bounds)
    print('detune1_bounds =', detune1_bounds, ', detune2_bounds =', detune2_bounds)

def import_para():
    drive_phi, drive_theta = False, True
    # drive_phi, drive_theta =  True, False
    # folder = 'data_xgate_theta.txt' if drive_theta else 'data_xgate_phi.txt'
    # f_xgate = pd.read_csv('data/'+folder)
    # para_tot = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2',
    #                     'detune_1', 'detune_2']].to_numpy()[:3]
    para_tot = np.array([
        [10.108462, 0.41212, 0.073451, -0.084451, -0.044706] ,
        [15.074864, 0.303821, 0.065666, -0.036206, -0.016563] ,
        [20.041257, 0.241329, 0.059044, -0.019402, -0.009802] ,
        [25.020473, 0.180208, 0.046132, -0.003789, 0.001544] ,
    ])

    n_cpu, n_job = 50, 30*len(para_tot)
    print('n_job=', n_job, ', n_cpu=', n_cpu)
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta)
    print('para_tot =')
    for para in para_tot:
        print(para.tolist(), ',')

#####################################################################
    truncation, truc1 = 1000, 300
    folder = 'data/'
    new_specdata = scq.read(folder + f'zeropi_specdata_truc={truncation}_3ncut.h5')
    n_theta = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
    n_phi = scq.read(folder + f'zeropi_n_theta_truc={truncation}_3ncut.h5')
    evals = 2*np.pi* new_specdata.energy_table
    n_Theta = 2*np.pi* n_theta.matrixelem_table
    n_Phi = 2*np.pi* n_phi.matrixelem_table
    evals = evals - evals[0]

    H0 = qt.Qobj(np.diag(evals[:truc1]))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_Phi[:truc1, :truc1]
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_Theta[:truc1, :truc1]

    ## find hilbert space
    thresh = 0.01
    hspace_charge = [0, 2]
    for s in hspace_charge:
        for i in range(truc1):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
#####################################################################
    hspace = np.arange(truc1).tolist()

    hilbert_space = hspace_charge
    print('\ntruc1_a=', truc1, '; truc1_b=', len(hspace_charge))
    args = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu]
    f_theta = Parallel(n_jobs=n_job, verbose=0)(delayed(xgate_fidelity_parallel)(arg, *args)
                                                for arg in para_tot)
    print(f'log of gate error (truc={len(hilbert_space)}) = ')
    for i in range(0, len(f_theta), 4):
        print(', '.join(map(str, np.round(f_theta[i:i+4], 8))), ',')

def import_para_noise():
    """
    Imports parameters and computes gate fidelities for noisy systems.
    """
    drive_phi, drive_theta, truc = False, True, 100
    # drive_phi, drive_theta, truc = True, False, 30
    parallel = False
    folder = 'data/data_xgate_theta_3ncut.txt' if drive_theta else 'data/data_xgate_phi.txt'
    f_xgate = pd.read_csv(folder)
    params = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()
    n_cpu, n_job = 4, len(params)
    logi_state = [0, 2]
    # [H0, drive_term, w_trans_1, w_trans_2, _] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc)
    folder = f'data/data_one_zeropi_truncation=1000/'
    evals = 2*np.pi* pd.read_csv(folder+ 'evals.txt').to_numpy().flatten()
    n_theta = 2*np.pi* pd.read_csv(folder+ 'n_theta.txt').to_numpy()
    n_phi = 2*np.pi* pd.read_csv(folder+ 'n_phi.txt').map(complex).to_numpy()
    gate_target = qt.sigmax()
    gamma2 =  1 / 1600e3
    # gammas =  1 / 2e3
    gammas = 1 / 5e3
    gamma2_p = 0 / 100e3
    # ratio = 10
    # gammas =  gamma_2 * ratio
    # gammas_p = 1 / 9000
    gammas_p = 1 / 5e3

    jump_t1   = []
    jump_tphi = []
    gamma_t1   = [0, gammas,  gamma2]  + [gammas]  * (truc-3)
    gamma_tphi = [0, gammas_p, gamma2_p] + [gammas_p] * (truc-3)
    for i in range(1,truc):
        jump_t1.append( np.sqrt(gamma_t1[i]) * qt.basis(truc,0) * qt.basis(truc,i).dag() )
        jump_tphi.append( np.sqrt(2*gamma_tphi[i]) * qt.basis(truc,i).proj() )

    H0 = qt.Qobj(np.diag(evals))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta
    hilbert_space = np.arange(truc).tolist()
    H0_truc = truncate_2(H0, hilbert_space)
    drive_truc = truncate_2(drive_term, hilbert_space)
    H_qbt_drive = [H0_truc, [drive_truc, drive_gauss_A],
                            [drive_truc, drive_gauss_B],]
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta, '; truc = ', truc)
    print('parallel = ', parallel)
    print('params =')
    for para in params:
        print(para.tolist(), ',')
    print("n_cpu = ", n_cpu, ";   n_job = ", n_job)
    print("gamma_2 = ", gamma2, "gamma_p2 = ", gamma2_p)
    print("gammas = ", gammas, "gammas_p = ", gammas_p)
    print(f"T1_2 = {1/gamma2} ns") if gamma2 != 0 else None
    print(f"T1_other = {1/gammas} ns") if gammas != 0 else None
    print(f"Tphi_2 = {1/gamma2_p} ns") if gamma2_p != 0 else None
    print(f"Tphi_other = {1/gammas_p} ns") if gammas_p != 0 else None

    ############################################################

    c_op_list = [qt.Qobj(np.zeros((truc, truc)))]
    # args = [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state, parallel, gate_target]
    # f_ideal = [xgate_fidelity_log_noise(params[0], *args)]
    # f_ideal = Parallel(n_jobs=n_job, verbose=0)(delayed(xgate_fidelity_log_noise)(args_indep, *args)
    #                                             for args_indep in params)
    args = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu]
    f_ideal = Parallel(n_jobs=n_job, verbose=0)(delayed(xgate_fidelity_parallel)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_ideal = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, f_ideal[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
    ############################################################
    c_op_list = jump_t1 + jump_tphi
    args = [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state, parallel, gate_target]
    f_noise = Parallel(n_jobs=n_job, verbose=0)(delayed(xgate_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_noise = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, f_noise[i:i+4])), ',')
    print(']')

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

if __name__ == '__main__':
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_para()
    import_para_noise()



