import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from qutip.qip.operations import rz, cz_gate, cnot, rx, hadamard_transform, swap
import cmath
import scipy.sparse as ssp
from sympy import symbols
from joblib import Parallel, delayed
from multiprocessing import Pool
from IPython.display import display, Math
import pandas as pd
from datetime import datetime
import pytz, os
import networkx as nx


XI = qt.tensor(qt.sigmax(), qt.identity(2))   


# max_step, nsteps = 1e-3, 1e4
### Define circuit and variable transform
 # EC = 0.20012190476190478
 # EJ2 = 5.4117
zp_yml ="""
# zero-pi circuit
branches:
- ["JJ", 1, 2, EJ1=6.01, ECJ=2.28]
- ["JJ", 3, 4, EJ1, ECJ]
- ["L", 2, 3, EL=0.38]
- ["L", 1, 4, EL]
- ["C", 1, 3, EC=0.2]
- ["C", 2, 4, EC]

- ["JJ", 5, 6, EJ2=5.41, ECJ]
- ["JJ", 7, 8, EJ2, ECJ]
- ["L", 6, 7, EL]
- ["L", 5, 8, EL]
- ["C", 5, 7, EC]
- ["C", 6, 8, EC]

- ["C", 1, 0, Ec0=1.0]
- ["C", 2, 9, Ec0]
- ["C", 3, 9, Ec0]
- ["C", 4, 0, Ec0]
- ["C", 5, 0, Ec0]
- ["C", 6, 9, Ec0]
- ["C", 7, 9, Ec0]
- ["C", 8, 0, Ec0]
"""

transform_2zeropi = np.array([
    [-0.5,  0.5, -0.5,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ],
    [-0.5,  0.5,  0.5, -0.5,  0. ,  0. ,  0. ,  0. ,  0. ],
    [ 0.5,  0.5, -0.5, -0.5,  0. ,  0. ,  0. ,  0. ,  0. ],
    [ 0.5,  0.5,  0.5,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ],
    [ 0. ,  0. ,  0. ,  0. , -0.5,  0.5, -0.5,  0.5,  0. ],
    [ 0. ,  0. ,  0. ,  0. , -0.5,  0.5,  0.5, -0.5,  0. ],
    [ 0. ,  0. ,  0. ,  0. ,  0.5,  0.5, -0.5, -0.5,  0. ],
    [ 0. ,  0. ,  0. ,  0. ,  0.5,  0.5,  0.5,  0.5,  0. ],
    [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ]])

logi_space = ['00', '02', '20',  '22',]

def set_fig_font():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE, labelsize=SMALL_SIZE)     # fontsize of the axes
    plt.rc(['xtick', 'ytick' ], labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

# Utility function for logging optimization progress
def print_soln(xk, convergence=0):
    """Log the best solution and convergence information during optimization."""
    print("Best Solution:", np.round(xk, 8).tolist())
    print("Convergence:", np.round(convergence, 4))
    print("----------------------------")

def transition_frequency(s0: int, s1: int, eval_tot) -> float:
    return ( (   eval_tot[s1]- eval_tot[s0] )* 2* np.pi)

# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

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
    # if isinstance(indices, (int, np.int8, np.int16, np.int32, np.int64)):
    #     indices = np.arange(indices)
    # if len(qobj.shape) == 1:
    #     return qt.Qobj(qobj[indices])
    if len(qobj.shape) == 2 and qobj.shape[0] == qobj.shape[1]:
        return qt.Qobj(qobj[np.ix_(indices, indices)])
    else:
        return qt.Qobj(qobj[indices])

# Factor global phase so that upper-left corner of matrix is real
def remove_global_phase(op):
    return op * np.exp(-1j * cmath.phase(op[0, 0]))

def dphi(prop, idx):
    return (-np.angle(prop[idx[0], idx[1]])
            +np.angle(prop[0,0]))

def gate_fidelity(Utarg, Ucand):
    ''' Compute gate fidelity between two unitaries (not quantum channels).

      F_gate(Utarg, Ucand)
        target unitary  and candidate unitary .

        Both gates shoud have the same dimension and we remove the global phase
    '''
    if Ucand.shape[0] != Utarg.shape[0]:
        raise ValueError(" Hilbert space dims don't match.")

    Ucandp = remove_global_phase(Ucand)
    dim =  Ucandp.shape[0]

    # Special case of Eq 5 in https://arxiv.org/pdf/quant-ph/0701138.
    denom = dim * (dim + 1)
    fidelity = (((Ucandp.dag() * Ucandp).tr() + np.abs((Ucandp.dag() * Utarg).tr()) ** 2) / denom )
    return fidelity

# Drive Coefficient on qubit A
def drive_cos_A(t: float, args: dict) -> float:
    A = args.get('drive_amp_A', 0)
    wd = args.get('drive_freq_A', 0)
    tg = args.get('gate_time', 0)
    return A * np.cos(wd * t) * (0<=t<=tg)

# Drive Coefficient on qubit B
def drive_cos_B(t: float, args: dict) -> float:
    A = args.get('drive_amp_B', 0)
    wd = args.get('drive_freq_B', 0)
    tg = args.get('gate_time', 0)
    return A * np.cos(wd * t) * (0<=t<=tg)

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

# Functions for the DRAG pulse for qubits A and B
def drag_A(t: float, args: dict) -> float:
    """
    Compute the DRAG pulse for qubit A.

    Args:
        t (float): Time at which to evaluate the pulse (0 <= t <= gate_time).
        args (dict): Dictionary containing pulse parameters:
            - drive_amp_A (float): Drive amplitude for qubit A.
            - drive_freq_A (float): Drive frequency for qubit A.
            - gate_time (float): Total duration of the gate.
            - alpha_A (float): DRAG parameter for qubit A.

    Returns:
        float: The value of the DRAG pulse for qubit A at time `t`.
    """
    A = args.get('drive_amp_A', 0)
    wd = args.get('drive_freq_A', 0)
    tg = args.get('gate_time', 0)
    alpha = args.get('alpha_A', 0)
    vg = A * (np.exp(-8 * t * (t - tg) / tg**2) - 1)
    return (vg * np.cos(wd * t) + alpha * (vg + A) * (-8 * (2 * t - tg) / tg**2) * np.sin(wd * t)) * (0 <= t <= tg)

def drag_B(t: float, args: dict) -> float:
    """
    Compute the DRAG pulse for qubit B.

    Args:
        t (float): Time at which to evaluate the pulse (0 <= t <= gate_time).
        args (dict): Dictionary containing pulse parameters:
            - drive_amp_B (float): Drive amplitude for qubit B.
            - drive_freq_B (float): Drive frequency for qubit B.
            - gate_time (float): Total duration of the gate.
            - alpha_B (float): DRAG parameter for qubit B.

    Returns:
        float: The value of the DRAG pulse for qubit B at time `t`.
    """
    A = args.get('drive_amp_B', 0)
    wd = args.get('drive_freq_B', 0)
    tg = args.get('gate_time', 0)
    alpha = args.get('alpha_B', 0)
    vg = A * (np.exp(-8 * t * (t - tg) / tg**2) - 1)
    return (vg * np.cos(wd * t) + alpha * (vg + A) * (-8 * (2 * t - tg) / tg**2) * np.sin(wd * t)) * (0 <= t <= tg)

def make_power_spectrum(pulse, tlist, drive_freq):
        # Power spectrum of pulse
        mag = np.fft.fftshift(np.abs(np.fft.fft(pulse)))
        freqs = np.fft.fftshift(np.fft.fftfreq(pulse.size, tlist[1]-tlist[0]))
        peak_loc = np.argmin(np.abs(freqs-drive_freq/(2*np.pi)))
        ps_db = 20*np.log10(mag/mag[peak_loc])
        return mag, freqs, peak_loc, ps_db

def geometric_phase_integral(xx, yy, zz):
    phi = np.arctan2(yy, xx)
    dphi = np.diff(phi, append=phi[0])
    costheta = zz
    return np.sum(-0.5*(1-costheta)*dphi)

def labmert_proj(xx, yy, zz):
    XX = np.sqrt(2/(1-zz))*xx
    YY = np.sqrt(2/(1-zz))*yy
    return XX, YY

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_coupling_rate():
    zp = scq.Circuit(zp_yml, from_file=False)
    zp.configure(transformation_matrix=np.linalg.inv(transform_2zeropi))
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [100, 100]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)    
    zp.cutoff_ext_1, zp.cutoff_ext_5 = 50, 50
    zp.cutoff_n_2, zp.cutoff_n_6 = 20, 20        
    n2, n6 = symbols('n2 n6')
    g = float(zp.sym_interaction((1,0), return_expr=True).coeff(n2*n6) )
    return g

def clean_eval_eket(eval, eket):
    sorted_idx = np.argsort(eval)
    eval = eval[sorted_idx]
    eval = eval - eval[0]
    eket = ssp.csr_matrix([eket[:,idx] for idx in sorted_idx])
    return eval,eket

def get_operator_two_zeropi(Ec0=1.0, truc1=30, truc_tot=50, charge_pick=False, 
                            n_cut=60, phi_cut=200, test=False):
    
    zp = scq.Circuit(zp_yml, from_file=False)
    zp.Ec0 = Ec0
    zp.configure(transformation_matrix=np.linalg.inv(transform_2zeropi))

    ##############################################################################################
    ### Construct subsystem, calculate eigenvalues
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [100, 100]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)
    if test:
        zp.cutoff_ext_1, zp.cutoff_ext_5 = 50, 50
        zp.cutoff_n_2, zp.cutoff_n_6 = 20, 20        
    else:
        zp.cutoff_ext_1, zp.cutoff_ext_5 = phi_cut, phi_cut
        zp.cutoff_n_2, zp.cutoff_n_6 = n_cut, n_cut
    n2, n6 = symbols('n2 n6')
    g = float(zp.sym_interaction((1,0), return_expr=True).coeff(n2*n6) )
    # print(f'zp.cutoff_ext_1, zp.cutoff_n_2 = {zp.cutoff_ext_1}, {zp.cutoff_n_2}')

    ### the two-line code below takes time when truc1 is large
    eval0, eket0 = zp.subsystems[0].eigensys(evals_count=truc1)
    eval1, eket1 = zp.subsystems[1].eigensys(evals_count=truc1)

    eval0, eket0 = clean_eval_eket(eval0, eket0)
    eval1, eket1 = clean_eval_eket(eval1, eket1)
    # eket0 = ssp.csr_matrix([eket0[:,idx] for idx in range(truc1)])
    # eket1 = ssp.csr_matrix([eket1[:,idx] for idx in range(truc1)])

    # get the n-operator in qubit basis of single qubit
    n_theta0 = (eket0 @ zp.subsystems[0].n2_operator() @ eket0.conj().T).todense()
    n_theta1 = (eket1 @ zp.subsystems[1].n6_operator() @ eket1.conj().T).todense()
    ##############################################################################################
    ###  Truncate two qubits using charge matrix elements
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    if charge_pick:
        hspace_0 = get_truncated_subspace_xgate(n_theta0, truc1)
        hspace_1 = get_truncated_subspace_xgate(n_theta1, truc1)
        # if hspace_theta != None:
        #     hspace_0 = hspace_theta
        #     hspace_1 = hspace_theta
        n_theta0 = truncate_2(n_theta0, hspace_0)
        n_theta1 = truncate_2(n_theta1, hspace_1)
        eval0 = eval0[hspace_0]
        eval1 = eval1[hspace_1]
        # eket0 = eket0[hspace_0]
        # eket1 = eket1[hspace_1]

    ##############################################################################################
    ###  Compute eigenvalues and eigenvectors for coupling H

    Hint = qt.tensor(qt.Qobj(n_theta0) , qt.Qobj(n_theta1))
    H_bare = (  qt.tensor(qt.Qobj(np.diag(eval0)),  qt.identity(len(hspace_1)))
            +  qt.tensor(qt.identity(len(hspace_0)),  qt.Qobj(np.diag(eval1))) )
    # Htot = (g* Hint + H_bare).tidyup(atol=1e-8)
    Htot = g* Hint + H_bare

    ### the one-line code below takes time when truc1 is large
    k = Htot.shape[0] - 1
    if truc_tot != None:
        k = truc_tot
    eval_tot, eket_tot = ssp.linalg.eigsh(Htot.data, k=k, which='SA', tol=1.e-10)

    sorted_idx_tot = np.argsort(eval_tot)
    eval_tot = eval_tot[sorted_idx_tot]
    eval_tot = eval_tot - eval_tot[0]
    eket_tot = ssp.csr_matrix([eket_tot[:,idx] for idx in sorted_idx_tot])

    ##############################################################################################
    ###  Get wavefunction overlap for the truncated dressed states
    bare_state = [[qt.tensor(qt.basis(len(hspace_0), i), qt.basis(len(hspace_1), j))
                            for j in range(len(hspace_1))]
                                for i in range(len(hspace_0))]

    arg = [bare_state, len(eval0), len(eval1)]
    result = Parallel(n_jobs=10, verbose=0)(delayed(find_overlap)(eket, arg) for eket in eket_tot)
    top_index = [result[i][0] for i in range(eval_tot.shape[0])]
    top_overlap = [result[i][1] for i in range(eval_tot.shape[0])]
    # for i in range(truc_tot):
    #     print(i, top3_index[i], top3_overlap[i])

    ##############################################################################################
    ### Get the dressed states index
    hspace_full = get_dressed_states_index(top_index, hspace_0, hspace_1)
    n_theta0_dress = ssp.kron(n_theta0, ssp.identity(len(hspace_1)))
    n_theta0_dress = np.abs(np.round(eket_tot @ n_theta0_dress @ eket_tot.conj().T, 8)).todense()
    n_theta1_dress = ssp.kron(ssp.identity(len(hspace_0)), n_theta1)
    n_theta1_dress = np.abs(np.round(eket_tot @ n_theta1_dress @ eket_tot.conj().T, 8)).todense()

    # hspace_logi = ['0-0', '0-2', '2-0', '2-2']
    # hspace_index = [index_state.index(i) for i in hspace_logi]
    # for s in hspace_index:
    #     for i in range(truc_optimize):
    #         if np.abs(n_theta1_dress[s, i]) > thresh_matrix_element and i not in hspace_index:
    #             hspace_index.append(i)
    # hspace_index.sort()
    # return [hspace_0, hspace_1, top_index, top_overlap,
    #         n_theta0_dress, n_theta1_dress, eval_tot, hspace_full]
    return [hspace_0, hspace_1, eval_tot, eket_tot]

def find_overlap(eket, *arg):
    bare_state, dim_0, dim_1 = arg
    num = 20
    overlaps = np.array([[np.abs( (eket @ bare_state[i][j].data).todense()[0,0] )
                        for j in range(dim_1)]
                            for i in range(dim_0)])
    flat_array = overlaps.flatten() # Flatten the 2D array
    # Find the indices of the top 10 largest values (in the flattened 1D array)
    top_indices_flat = np.argpartition(-flat_array, num)[:num]
    # Convert the flat indices to 2D indices
    top_indices_2d = np.unravel_index(top_indices_flat, overlaps.shape)
    # Extract the values corresponding to the indices
    top_values = overlaps[top_indices_2d]
    # Sort the values in descending order
    sorted_indices = np.argsort(-top_values)  # Use a negative sign for descending order
    sorted_top_indices = [tuple(zip(top_indices_2d[0], top_indices_2d[1]))[i] for i in sorted_indices]
    sorted_top_values = top_values[sorted_indices]
    return sorted_top_indices, sorted_top_values

###################################################################
# CNOT-gate

def cnot_fidelity_log(arg_all):
    if len(arg_all) == 14:
        use_qt_fidelity = True
    elif len(arg_all) == 15:
        use_qt_fidelity = arg_all[-1]
        arg_all = arg_all[:-1]
    else:
        raise ValueError(
            "cnot_fidelity_log expects 14 arguments plus optional "
            "use_qt_fidelity"
        )
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
     H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx,
     mid_state, option_ideal, option_noisy] = arg_all
    pulse_args = {'drive_amp_A': drive_amp_A ,
                'drive_freq_A': w_0_2 + 2*np.pi*detune_A,
                'drive_amp_B': drive_amp_B ,
                'drive_freq_B': w_1_2 + 2*np.pi*detune_B,
                'gate_time': tg }
    tlist = np.linspace(0, tg, num=3*int(tg))  # total time

    propagator = get_propagator(H_qbt_drive, tlist, num_cpus, c_op_list, pulse_args, 
                          logi_idx, option_ideal, option_noisy)
    # print(f'np.shape(propagator)={np.shape(propagator)}')    
    # print(f'propagator={propagator}')    
    
    fidelity = get_fidelity_super_operator(
        propagator, logi_idx, cnot(), c_op_list, mid_state,
        use_qt_fidelity=use_qt_fidelity,
    )
    
    # if prop.isoper:
    #     U_final = cnot_phase_correct(prop, mid_state)
    # else:
    #     U_kraus = qt.to_kraus(qt.to_super(prop))
    #     U_kraus_zz = [cnot_phase_correct(truncate_2(u, logi_idx), mid_state) for u in U_kraus ]
    #     U_final = qt.kraus_to_super(U_kraus_zz)
    # fidelity = qt.average_gate_fidelity(U_final, target=cnot())
    return np.log10(1-fidelity)

def cnot_fidelity_log_noise(arg_optimize, *args):
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = arg_optimize
    if len(args) == 9:
        use_qt_fidelity = True
    elif len(args) == 10:
        use_qt_fidelity = args[-1]
        args = args[:-1]
    else:
        raise ValueError(
            "cnot_fidelity_log_noise expects 9 system arguments plus "
            "optional use_qt_fidelity"
        )
    [H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx,
     mid_state, option_ideal, option_noisy] = args

    arg_all = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
     H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, 
     mid_state, option_ideal, option_noisy, use_qt_fidelity]

    return cnot_fidelity_log(arg_all)

def cnot_phase_correct(U_kraus, mid_state):
    U_final = []
    # print(f'np.shape(U_kraus)={np.shape(U_kraus)}')    
    # print(f'U_kraus={U_kraus}')    
    for prop in U_kraus:    
        prop = qt.Qobj(prop, dims=[[2, 2], [2, 2]])
        XI = qt.tensor(qt.sigmax(), qt.qeye(2))
        if mid_state in [ '4-1', '1-4', '8-0' ]:
            prop = XI* prop * XI  
        # print(f'np.shape(prop)={np.shape(prop)}')    
        # print(f'prop={prop}')    
        Uc_prime = swap()* prop* swap() # for |45> state
        phase = np.angle(Uc_prime)
        x1 = 0.5* (- phase[1,1] + phase[2,3] - phase[3,2] + phase[0,0])
        x2 = 0.5* (  phase[1,1] - phase[2,3] - phase[3,2] + phase[0,0])
        x3 = 0.5* (- phase[1,1] - phase[2,3] + phase[3,2] + phase[0,0])
        U_bef = qt.Qobj(np.diag([1, np.exp(1j* x1), np.exp(1j* x2), np.exp(1j* (x1+x2))])
                , dims=[[2, 2], [2, 2]])
        U_aft = qt.Qobj(np.diag([1, np.exp(1j* x3), 1, np.exp(1j* x3)])
                , dims=[[2, 2], [2, 2]])
        U_final.append(np.exp(-1j* phase[0,0])* U_bef* Uc_prime* U_aft)
    return U_final

###################################################################
# XI-gate


def XI_fidelity_log(arg_all):
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
     H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, 
     mid_state, option_ideal, option_noisy] = arg_all
    pulse_args = {'drive_amp_A': drive_amp_A ,
                'drive_freq_A': w_0_2 + 2*np.pi*detune_A,
                'drive_amp_B': drive_amp_B ,
                'drive_freq_B': w_1_2 + 2*np.pi*detune_B,
                'gate_time': tg }
    tlist = np.linspace(0, tg, num=10*int(tg))  # total time

    propagator = get_propagator(H_qbt_drive, tlist, num_cpus, c_op_list, pulse_args, 
                          logi_idx, option_ideal, option_noisy)

    fidelity = get_fidelity_super_operator(propagator, logi_idx, XI, c_op_list, mid_state)
    
    return np.log10(1-fidelity)

def XI_fidelity_log_noise(arg_optimize, *args):
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = arg_optimize
    [H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, 
     mid_state, option_ideal, option_noisy] = args

    arg_all = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
     H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, 
     mid_state, option_ideal, option_noisy]

    return XI_fidelity_log(arg_all)

def XI_phase_correct(U_kraus):
    U_final = [] 
    for prop in U_kraus:    
        prop = qt.Qobj(prop, dims=[[2, 2], [2, 2]]) 
        Uc_prime = swap()* prop* swap() # for |45> state
        phase = np.angle(Uc_prime)
        x1 = - phase[0,2]
        x2 = - phase[3,1]
        U_bef = qt.Qobj(np.diag([1, np.exp(1j* x1), np.exp(1j* x2), np.exp(1j* (x1+x2))])
                , dims=[[2, 2], [2, 2]])
        U_final.append(np.exp(-1j* phase[2,0])* Uc_prime* U_bef)
    return U_final

# def cnot_phase_correct(U_kraus):
#     U_final = []
#     for u in U_kraus:
#         phase = np.angle(u)
#         x2 = 0.5* (  phase[0,0] + phase[1,3] - phase[2,2] - phase[3,1])
#         x3 = 0.5* (  phase[0,0] - phase[1,3] + phase[2,2] - phase[3,1])
#         x4 = 0.5* (  phase[0,0] - phase[1,3] - phase[2,2] + phase[3,1])
#         Ur = qt.Qobj( np.diag([ 1, np.exp(1j*x3), np.exp(1j*x4), np.exp(1j*(x3+x4)) ]))
#         Ul = qt.Qobj( np.diag([ 1, 1, np.exp(1j*x2), np.exp(1j*x2) ]))
#         U_final.append( Ul * u * Ur)
#     return U_final

# def cnot_fidelity_log_noise_wrong(arg_all):
#     [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
#      H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx] = arg_all
#     pulse_args = {'drive_amp_A': drive_amp_A ,
#                 'drive_freq_A': w_0_2 + 2*np.pi*detune_A,
#                 'drive_amp_B': drive_amp_B ,
#                 'drive_freq_B': w_1_2 + 2*np.pi*detune_B,
#                 'gate_time': tg }
#     tlist = np.linspace(0, tg, num=int(tg))  # total time

#     U_noise = get_propagator(H_qbt_drive, tlist, num_cpus, c_op_list, pulse_args, logi_idx)
#     p0_kraus = qt.to_kraus(qt.to_super(U_noise))
#     if len(c_op_list) != 0:
#         p0_kraus = [truncate_2(i, logi_idx) for i in p0_kraus]

#     p0_kraus_zz = cnot_phase_correct(p0_kraus)
#     p0_super_2 = qt.kraus_to_super(p0_kraus_zz)
#     f_noise = qt.metrics.average_gate_fidelity(p0_super_2, target=qt.Qobj(cnot().full()))
#     return np.log10(1-f_noise)

# def cnot_fidelity_log_notg(arg_optimize, *args):
#     [drive_amp_A, drive_amp_B, detune_A, detune_B] = arg_optimize

#     [H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, tg] = args

#     arg_all = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
#      H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx]

#     return cnot_fidelity_log(arg_all)

# def get_fidelity_cnot_1A0(arg_de, *args):
#     drive_amp, tg, detune = arg_de

#     (w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
#     n_theta2_20_14, state_tot, H_qbt_drive, n1n1) = args

#     if n1n1:
#         pulse_args = {'drive_amp_A': drive_amp* (n_theta1_20_14/n_theta1_00_14) ,
#                     'drive_freq_A': w_00_14 + 2*np.pi*detune,
#                     'drive_amp_B': drive_amp ,
#                     'drive_freq_B': w_20_14 + 2*np.pi*detune,
#                     'gate_time': tg }
#     else:
#         pulse_args = {'drive_amp_A': drive_amp* (n_theta1_20_14/n_theta2_00_14) ,
#                     'drive_freq_A': w_00_14 + 2*np.pi*detune,
#                     'drive_amp_B': drive_amp ,
#                     'drive_freq_B': w_20_14 + 2*np.pi*detune,
#                     'gate_time': tg }
#     tlist = np.linspace(0, tg, num=int(tg))  # total time
#     prop = qt.propagator( H=H_qbt_drive,
#                           t=tlist,
#                           args=pulse_args,
#                         #   c_op_list=c_op_list,
#                           )[-1]  # get the propagator at the final time step
#     fidelity = cnot_fidelity( prop, state_tot )
#     return np.log10(1-fidelity)

# def get_transition_freq_cnot(args, trans_goal='02-45_22-45'):
# # def get_transition_freq(args, trans_goal='00-14_20-14'):

#     [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
#     eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot  ] = args

#     transition_ik_jk = []
#     w_ik = []
#     w_jk = []
#     n_theta1_ik = []
#     n_theta2_ik = []
#     n_theta1_jk = []
#     n_theta2_jk = []
#     thresh = 0.01
#     for idx in range(len(logi_space)):
#         for jdx in range(idx+1, len(logi_space)):
#             state_i = logi_space[idx]
#             state_j = logi_space[jdx]

#             i = order_sort.index(state_i)
#             j = order_sort.index(state_j)
#             for k in np.arange(0, len(eval_tot)):

#                 if (( np.abs(n_theta1_dress[i,k]) > thresh or np.abs(n_theta2_dress[i,k]) > thresh ) and (
#                     np.abs(n_theta1_dress[j,k]) > thresh or np.abs(n_theta2_dress[j,k]) > thresh ) ):
#                     transition_ik_jk.append(f'{state_i}-{order_sort[k]}_{state_j}-{order_sort[k]}')

#                     w_ik.append( 2*np.pi*( np.abs( eval_tot[i] - eval_tot[k] )))
#                     w_jk.append( 2*np.pi*( np.abs( eval_tot[j] - eval_tot[k] )))

#                     n_theta1_ik.append(np.abs(n_theta1_dress[i ,k]))
#                     n_theta2_ik.append(np.abs(n_theta2_dress[i ,k]))
#                     n_theta1_jk.append(np.abs(n_theta1_dress[j ,k]))
#                     n_theta2_jk.append(np.abs(n_theta2_dress[j ,k]))

    # print('ik_jk' + ';         w_ik' + ';     w_jk' + ';  n_theta1_ik' + ';  n_theta2_ik' + ';  n_theta1_jk' + ';  n_theta2_jk' )
    # for i in range(len(transition_ik_jk)):
    #     print(transition_ik_jk[i] + ';  %.3f'%w_ik[i] + ';  %.3f'%w_jk[i]
    #     + ';  %.3f'%n_theta1_ik[i] + ';  %.3f'%n_theta2_ik[i] + ';  %.3f'%n_theta1_jk[i] + ';  %.3f'%n_theta2_jk[i] )

    # thresh = 0.02 :
    # ['00-08_02-08', '00-80_20-80', '00-14_20-14', '02-82_22-82', '02-45_22-45', '20-28_22-28']

    # thresh = 0.01 :
    # ['00-01_02-01', '00-08_02-08', '00-80_02-80', '00-80_20-80',
    # '00-14_20-14', '00-41_20-41', '02-80_20-80', '02-12_22-12',
    # '02-82_22-82', '02-45_22-45', '20-21_22-21', '20-28_22-28']

#     idx = transition_ik_jk.index(trans_goal)
#     # print(transition_ik_jk)
# # Trans ;  w_ij;    n_theta1; n_theta2; sum
# # 00-14 ;  45.925 ;  0.032 ;  0.006 ;  0.038
# # 20-14 ;  24.289 ;  0.046 ;  0.001 ;  0.047

#     w_00_14 = w_ik[idx]
#     w_20_14 = w_jk[idx]
#     n_theta1_00_14 = n_theta1_ik[idx]
#     n_theta2_00_14 = n_theta2_ik[idx]
#     n_theta1_20_14 = n_theta1_jk[idx]
#     n_theta2_20_14 = n_theta2_jk[idx]
#     return w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14, n_theta2_20_14

# def get_pop(args, *argz, state_0='00'):

#     drive_amp_A, drive_amp_B, tg, detune_A, detune_B = args
#     (n_theta1_20_14, n_theta1_00_14, w_00_14, w_20_14,
#      trunc_states, logi_space, H_qbt_drive, e_ops ) = argz

#     pulse_args = {'drive_amp_A': drive_amp_A* (n_theta1_20_14/n_theta1_00_14) ,
#                 'drive_freq_A': w_00_14 + 2*np.pi*detune_A,
#                 'drive_amp_B': drive_amp_B ,
#                 'drive_freq_B': w_20_14 + 2*np.pi*detune_B,
#                 'gate_time': tg }

#     tlist = np.linspace(0, tg, num=1000)  # total time
#     options = qt.Options(nsteps=10000, store_states=True)
#     result = qt.mesolve(
#         H=H_qbt_drive,
#         rho0=qt.basis( len(trunc_states), logi_space.index(state_0) ),
#         tlist=tlist,
#         e_ops=e_ops,
#         args=pulse_args,
#         options=options
#     )
#     return result.expect[0][-1]


###################################################################
# CZ-gate


def cz_phase_correct(U_kraus):
    U_final = []
    for u in U_kraus:
        Uz = remove_global_phase(qt.tensor( rz(dphi(u, (2,2))),
                                            rz(dphi(u, (1,1)))))
        Uc_reshaped = qt.Qobj(u.data, dims=[[2, 2], [2, 2]])
        U_final.append(remove_global_phase(Uz * Uc_reshaped))
    return U_final

def cz_fidelity_log(arg_all):
    # print_time()
    # print('debug: good in cz_fidelity_log beginning')
    if len(arg_all) == 10:
        arg_all = [*arg_all, True]
    [tg, drive_amp, detune,
     H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx,
     option_ideal, option_noisy, use_qt_fidelity] = arg_all

    pulse_args = {'drive_amp_A': drive_amp,
                'drive_freq_A': W_target + 2*np.pi*detune,
                'gate_time': tg}
    tlist = np.linspace(0, tg, num=3*int(tg))  # total time

    # print_time()
    # print('debug: good in cz_fidelity_log  before get_propagator')
    propagator = get_propagator(H_qbt_drive, tlist, num_cpus, c_op_list, pulse_args, 
                                logi_idx, option_ideal, option_noisy)
    
    # print_time()
    # print('debug: good in cz_fidelity_log  before get_fidelity_super_operator')
    fidelity = get_fidelity_super_operator(
        propagator, logi_idx, cz_gate(), c_op_list,
        use_qt_fidelity=use_qt_fidelity,
    )
    # p0_kraus = qt.to_kraus(qt.to_super(propagator))
    # if len(c_op_list) != 0:
    #     p0_kraus = [truncate_2(i, logi_idx) for i in p0_kraus]
    # p0_kraus_zz = cz_phase_correct(p0_kraus)
    # p0_super_2 = qt.kraus_to_super(p0_kraus_zz)
    # fidelity = qt.metrics.average_gate_fidelity(p0_super_2, target=cz_gate())
    return np.log10(1-fidelity)

def cz_fidelity_log_noise(arg_optimize, *args):
    [tg, drive_amp, detune] = arg_optimize # Independent arguments that can be optimized over
    if len(args) == 7:
        args = (*args, True)
    [H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx,
     option_ideal, option_noisy, use_qt_fidelity] = args # System arguments

    arg_all = [tg, drive_amp, detune, H_qbt_drive, W_target, num_cpus,
               c_op_list, logi_idx, option_ideal, option_noisy,
               use_qt_fidelity]

    return cz_fidelity_log(arg_all)

def sesolve_parallel(argz):
    """Parallel SESolve function for solving the Schrodinger equation."""
    H, psi0, tlist, pulse_args, options = argz
    return psi0, qt.sesolve(H, qt.basis(H[0].shape[0], psi0), tlist, options=options, args=pulse_args)

def cz_fidelity_log_old(arg_all):
    [tg, drive_amp, detune, n_cpu, hspace, W_20_50, H_drive, logic_idx] = arg_all

    pulse_args = {'drive_amp_A': drive_amp,
                'drive_freq_A': W_20_50 + 2*np.pi*detune,
                'gate_time': tg}
    tlist = np.linspace(0, tg, num=int(tg))  # total time

    sesolve_args = []
    prop = np.zeros((len(hspace), len(logic_idx)), dtype=np.complex128)
    if n_cpu > 1:
        options = qt.Options(num_cpus=n_cpu, nsteps=1e4)
        for i in logic_idx:
            sesolve_args.append([H_drive, i, tlist, pulse_args, options])
        pool = Pool(processes=n_cpu)
        for i, res in pool.imap_unordered(sesolve_parallel, sesolve_args):
            prop[:, logic_idx.index(i)] = res.states[-1].full().flatten()
    else:
        options = qt.Options(num_cpus=5, nsteps=1e4)
        for i in logic_idx:
            sesolve_args.append([H_drive, i, tlist, pulse_args, options])
        for se_arg in sesolve_args:
            i, res = sesolve_parallel(se_arg)
            prop[:, logic_idx.index(i)] = res.states[-1].full().flatten()

    Uc = truncate_2(qt.Qobj(prop), logic_idx)
    Uz = remove_global_phase(qt.tensor( rz(dphi(Uc, (2,2))),
                                        rz(dphi(Uc, (1,1)))))
    Uc_reshaped = qt.Qobj(Uc.data, dims=[[2, 2], [2, 2]])
    Ucprime = remove_global_phase(Uz * Uc_reshaped)
    fidelity = qt.average_gate_fidelity(Ucprime, target=cz_gate())
    return np.log10(1-fidelity)




# def cz_fidelity_optimize(arg, *args):
#     detune, drive_amp = arg # Independent arguments that can be optimized over
#     [H_qbt_drive, W_target, state_logi, tg] = args # System arguments
#     pulse_args = {'drive_amp_A': drive_amp,
#                 'drive_freq_A': W_target + 2*np.pi*detune,
#                 'gate_time': tg}
#     tlist = np.linspace(0, tg, num=int(3*tg))  # total time
#     prop = qt.propagator( H=H_qbt_drive,
#                           t=tlist,
#                           args=pulse_args,)[-1]  # get the propagator at the final time step
#     fidelity = cz_fidelity( prop, state_logi )
#     return np.log10(1-fidelity)

# def cz_fidelity_parallel(arg, *args):
#     tg, detune, drive_amp = arg # Independent arguments that can be optimized over
#     [num_cpus, H_qbt_drive, W_target, state_logi] = args # System arguments

#     pulse_args = {'drive_amp_A': drive_amp,
#                 'drive_freq_A': W_target + 2*np.pi*detune,
#                 'gate_time': tg}
#     tlist = np.linspace(0, tg, num=int(3*tg))  # total time
#     prop = qt.propagator( H=H_qbt_drive,
#                         t=tlist,
#                         args=pulse_args,
#                         options=qt.Options( num_cpus=1 ),
#                         num_cpus=num_cpus,
#                         parallel=True,
#                         )[-1]  # get the propagator at the final time step
#     fidelity = cz_fidelity_log_noise( prop, state_logi )
#     return np.log10(1-fidelity)

# def cz_fidelity_log(arg_all):
#     [tg, drive_amp, detune, n_cpu, hspace, W_20_50, H_drive, logic_idx] = arg_all

#     pulse_args = {'drive_amp_A': drive_amp,
#                 'drive_freq_A': W_20_50 + 2*np.pi*detune,
#                 'gate_time': tg}
#     tlist = np.linspace(0, tg, num=int(tg))  # total time

#     sesolve_args = []
#     prop = np.zeros((len(hspace), len(logic_idx)), dtype=np.complex128)
#     if n_cpu > 1:
#         options = qt.Options(num_cpus=n_cpu, nsteps=1e4)
#         for i in logic_idx:
#             sesolve_args.append([H_drive, i, tlist, pulse_args, options])
#         pool = Pool(processes=n_cpu)
#         for i, res in pool.imap_unordered(sesolve_parallel, sesolve_args):
#             prop[:, logic_idx.index(i)] = res.states[-1].full().flatten()
#     else:
#         options = qt.Options(num_cpus=5, nsteps=1e4)
#         for i in logic_idx:
#             sesolve_args.append([H_drive, i, tlist, pulse_args, options])
#         for se_arg in sesolve_args:
#             i, res = sesolve_parallel(se_arg)
#             prop[:, logic_idx.index(i)] = res.states[-1].full().flatten()

#     Uc = truncate_2(qt.Qobj(prop), logic_idx)
#     Uz = remove_global_phase(qt.tensor( rz(dphi(Uc, (2,2))),
#                                         rz(dphi(Uc, (1,1)))))
#     Uc_reshaped = qt.Qobj(Uc.data, dims=[[2, 2], [2, 2]])
#     Ucprime = remove_global_phase(Uz * Uc_reshaped)
#     fidelity = qt.average_gate_fidelity(Ucprime, target=cz_gate())
#     return np.log10(1-fidelity)

# def cz_fidelity_log_noise_qutip(args_indep, *args):
#     tg, drive_amp, detune = args_indep # Independent arguments that can be optimized over
#     [H0, drive_term, W_target, max_steps, n_cpu, c_op_list] = args # System arguments

#     H_qbt_drive = [H0, [2*np.pi* drive_term, drive_gauss_A] ]

#     pulse_args = {'drive_amp_A': drive_amp,
#                 'drive_freq_A': W_target + 2*np.pi*detune,
#                 'gate_time': tg}
#     tlist = np.linspace(0, tg, num=3*int(tg))  # total time
#     options =qt.Options(max_step=max_steps, nsteps=1e4, num_cpus=1 )
#     p = qt.propagator( H=H_qbt_drive,
#                             t=tlist,
#                             c_op_list=c_op_list,
#                             options=options,
#                             args=pulse_args,
#                             num_cpus=n_cpu,
#                             parallel=True,
#                             )[-1]  # get the propagator at the final time step
#     p0_kraus = qt.to_kraus(qt.to_super(p))

#     # the following could lead to error
#     p0_kraus = [truncate(i, 4) for i in p0_kraus]
#     p0_kraus_zz = cz_phase_correct(p0_kraus)
#     p0_super_2 = qt.kraus_to_super(p0_kraus_zz)
#     f_noise = qt.metrics.average_gate_fidelity(p0_super_2, target=cz_gate())
#     return np.log10(1-f_noise)

###################################################################
# X-gate

def compute_drive_xgate(evals, n_theta, n_phi, drive_phi, drive_theta, gamma_t1=None):
    """
    Computes the transition frequencies and selects the appropriate drive term.
    need lowest 10 evals for this func to work.

    Parameters:
        evals (np.ndarray): Energy levels.
        n_theta (np.ndarray): Matrix elements for theta-drive.
        n_phi (np.ndarray): Matrix elements for phi-drive.
        drive_phi (bool): Whether phi-drive is used.
        drive_theta (bool): Whether theta-drive is used.
        gamma_t1 (float): Amplitude damping rate.

    Returns:
        w1 (float): Transition frequency 1.
        w2 (float): Transition frequency 2.
        drive_term (np.ndarray): Matrix elements for selected drive.
        Gamma_t1 (float): the decay rate coefficient fixed by certain transition matrix element
    """
    if drive_phi and not drive_theta:
        w1 = evals[9] - evals[0]
        w2 = evals[9] - evals[2]
        drive_term = n_phi
        if gamma_t1 != None:
            Gamma_t1 = gamma_t1 / (np.abs(n_phi[4,9])**2)
    elif drive_theta and not drive_phi:
        w1 = evals[7] - evals[0]
        w2 = evals[7] - evals[2]
        drive_term = n_theta
        if gamma_t1 != None:
            Gamma_t1 = gamma_t1 / (np.abs(n_theta[4,7])**2)
    else:
        w1 = evals[9] - evals[0]
        w2 = evals[9] - evals[2]
        drive_term = 0.976 * n_phi + 0.024 * n_theta
        if gamma_t1 != None:
            Gamma_t1 = gamma_t1 / (np.abs(n_phi[4,9])**2) # close to phi drive
    if gamma_t1 != None:
        return w1, w2, drive_term, Gamma_t1
    else: 
        return w1, w2, drive_term

def get_truncated_subspace_xgate(drive_term, n_full, hspace_charge=[0,2], thresh=0.01):
    """
    Determines a reduced Hilbert space based on magnitude of charge matrix elements.
    
    Parameters:
        drive_term (np.ndarray): Drive matrix.
        n_full (int): Dimension of the full space.
        hspace_charge (list): Initial state list to include.
        thresh (float): Magnitude threshold for inclusion.
    Returns:
        hspace_charge (list): List of basis indices to include.
    """
    for s in hspace_charge:
        for i in range(n_full):
            if np.abs(drive_term[s, i] / (2 * np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
    return hspace_charge

def build_hamiltonian_xgate(evals, drive_term, hspace, logi_state):
    """
    Constructs the truncated Hamiltonian and drive terms.

    Parameters:
        H0 (Qobj): Diagonalized bare Hamiltonian.
        drive_term (np.ndarray): Drive matrix.
        hspace_charge (list): Truncated Hilbert space indices.
        logi_state (list): Logical states, e.g., [0, 2].

    Returns:
        H_qbt_drive (list): Full driven Hamiltonian.
        drive_truc (Qobj): Truncated drive matrix.
        logi_idx (list): Logical state indices in truncated space.
    """
    H0 = qt.Qobj(np.diag(evals))
    H0_truc = truncate_2(H0, hspace)
    drive_truc = truncate_2(drive_term, hspace)
    logi_idx = [hspace.index(s) for s in logi_state] # 1 state may or may not be in the truncated model, 
    H_qbt_drive = [H0_truc, [drive_truc, drive_gauss_A], [drive_truc, drive_gauss_B]]
    return H_qbt_drive, drive_truc, logi_idx



def construct_c_ops_xgate(n_hspace, drive_truc, Gamma_t1, gamma_dephase_new, tphi, hspace, state_idx_tphi, 
                            apply_decay=True, apply_dephase=True):
    """
    Constructs collapse operators for dissipation.

    Parameters:
        n_hspace (int): Hilbert space dimension.
        drive_truc (Qobj): Drive operator.
        Gamma_t1 (float): Amplitude decay prefactor.
        gamma_dephase_new (np.ndarray): Dephasing rates (for 50μs).
        tphi (float): Desired Tphi in μs.
        hspace (list): Hilbert space indices.
        state_idx_tphi (list): Indices of states for imported dephasing rates.

    Returns:
        list: All collapse operators (amplitude + dephasing).
    """
    gamma_decay_new = Gamma_t1 * np.abs(drive_truc.full()) ** 2 # 
    # the dephasing rate is calculated in some file for 50μs for 2 state, the line below change dephasing coeffs to the input tphi (170, 30, 3μs)
    
    # print('gamma_dephase_new=', gamma_dephase_new)
    gamma_dephase_new = gamma_dephase_new * 50 / tphi
    # print('gamma_dephase_new after scaling=', gamma_dephase_new)
    
    jump_t1, jump_tphi = [], []
    if apply_decay:
        for i in range(1, n_hspace):
            for j in range(i): # only consider downwards deacy
                jump_t1.append(np.sqrt(gamma_decay_new[j, i]) * qt.basis(n_hspace, j) * qt.basis(n_hspace, i).dag())
    if apply_dephase:
        for i, state in enumerate(hspace):
            if state in list(state_idx_tphi):
                idx = list(state_idx_tphi).index(state) 
                jump_tphi.append(np.sqrt(2 * gamma_dephase_new[idx]) * qt.basis(n_hspace, i).proj())
            # else:
            #     jump_tphi.append(qt.Qobj(np.zeros((n_hspace, n_hspace))))    
    print('np.shape(jump_t1)=',  np.shape(jump_t1), '; np.shape(jump_tphi)=',  np.shape(jump_tphi))
    return jump_t1 + jump_tphi


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

def xgate_fidelity_log(argz):
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, 
     H_qbt_drive, w_trans_1, w_trans_2, num_cpus, 
     c_op_list, logi_idx, option_ideal, option_noisy] = argz

    pulse_args = {
        'drive_amp_A': drive_amp_A,
        'drive_freq_A': w_trans_1 + 2 * np.pi * detune_A,
        'drive_amp_B': drive_amp_B,
        'drive_freq_B': w_trans_2 + 2 * np.pi * detune_B,
        'gate_time': tg,
    }
    tlist = np.linspace(0, tg, num= 3*int(tg))
    propagator = get_propagator(H_qbt_drive, tlist, num_cpus, 
                                c_op_list, pulse_args, logi_idx, option_ideal, option_noisy)
    fidelity = get_fidelity_super_operator(propagator, logi_idx, qt.sigmax(), c_op_list)
    return np.log10(1 - fidelity)

def xgate_fidelity_log_noise(args_indep, *args):
    """
    Computes the X-gate infidelity (log10(1-F)) for a noisy system.

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
            - num_cpus (int): Number of CPUs for parallelization.
            - c_op_list (list): Collapse operators for modeling noise.
            - logi_idx (list): Logical state indices for truncation.
            - gate_target (qt.Qobj): Target gate for fidelity comparison.
            - option_ideal, option_noisy (qt.Options): Solver options for QuTiP.

    Returns:
        float: Logarithm of the infidelity for the X-gate in a noisy system.
    """
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = args_indep
    if len(args) == 8:
        # Backward compatibility for existing optimization/archive scripts.
        args = (*args, True)
    [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx,
     option_ideal, option_noisy, use_qt_fidelity] = args

    pulse_args = {
        'drive_amp_A': drive_amp_A,
        'drive_freq_A': w_trans_1 + 2 * np.pi * detune_A,
        'drive_amp_B': drive_amp_B,
        'drive_freq_B': w_trans_2 + 2 * np.pi * detune_B,
        'gate_time': tg,
    }
    tlist = np.linspace(0, tg, num= 10*int(tg))
    propagator = get_propagator(H_qbt_drive, tlist, num_cpus, 
                                c_op_list, pulse_args, logi_idx, option_ideal, option_noisy)
    fidelity = get_fidelity_super_operator(
        propagator,
        logi_idx,
        qt.sigmax(),
        c_op_list,
        use_qt_fidelity=use_qt_fidelity,
    )

    # print(f'\n len(c_op_list)={len(c_op_list)}')
    # print(f'U_noise.istp={propagator.istp}')
    # print(f'U_noise.iscp={propagator.iscp}')
    # print(f'U_noise={propagator}')
    # print('fidelity=', fidelity)
    # if len(c_op_list) != 0:
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     qt.qsave(propagator, f'U_noise_tg={tg:.0f}_truc={H_qbt_drive[0].shape[0]}_{current_time}')
    ### U_loaded = qt.qload('U_noise')
    
    return np.log10(1 - fidelity)

def parallel_sesolve(n, N, H, tlist, args, options):
    """Parallel SESolve function for solving the Schrodinger equation."""
    psi0 = qt.basis(N, n)
    output = qt.sesolve(H, psi0, tlist, [], args, options, _safe_mode=False)
    return output

def get_propagator(H, tlist, num_cpus, c_op_list, pulse_args, logi_idx, option_ideal=None, option_noisy=None):
    """
    Compute the propagator for a quantum system, supporting both noiseless and noisy systems.

    Args:
        H (list or qt.Qobj): The Hamiltonian of the system. Can be a single Qobj or a list where
                             the first element represents the static part.
        tlist (list): List of time points for the simulation.
        num_cpus (int): Number of CPUs to use for parallel computation (if `parallel=True`).
        c_op_list (list): List of collapse operators for modeling noise. If empty, the system is noiseless.
        pulse_args (dict): Arguments for time-dependent pulse functions in the Hamiltonian.
        option_ideal, option_noisy (qt.Options): Solver options for QuTiP.
        logi_idx (list): List of logical states (indices of basis states) to include in the propagator.

    Returns:
        qt.Qobj or list of qt.Qobj:
            - For noiseless systems: A `Qobj` representing the truncated propagator for logical states.
            - For noisy systems: A `Qobj` representing the superoperator propagator for the final time step.
    """
    dimz = len(logi_idx)
    H0 = H[0][0] if isinstance(H[0], list) else H[0] if isinstance(H, list) else H
    if len(c_op_list) == 0:

        N = H0.shape[0]
        if num_cpus > 1:
            u = np.zeros([N, dimz, len(tlist)], dtype=complex)
            output = qt.parallel.parallel_map(parallel_sesolve, logi_idx,
                                    task_args=(N, H, tlist, pulse_args, option_ideal),
                                    num_cpus=num_cpus)
            for n in range(dimz):
                for k, t in enumerate(tlist):
                    u[:, n, k] = output[n].states[k].full().T
            prop = [qt.Qobj(u[:, :, k], dims=[[[N], [N]], [[dimz], [dimz]]]) for k in range(len(tlist))][-1]
            return truncate_2(prop, logi_idx)
        else:
            # Computes the propagator for noiseless systems.
            prop = np.zeros((H0.shape[0], dimz), dtype=np.complex128)
            for i in logi_idx:
                res = qt.sesolve(H, qt.basis(H[0].shape[0], i), tlist, options=option_ideal, args=pulse_args)
                prop[:, logi_idx.index(i)] = res.states[-1].full().flatten()
            Uc = truncate_2(qt.Qobj(prop), logi_idx)
            return Uc

    else: # noise

        # Computes the propagator for noisy systems.
        proj_idx = [(i, j) for j in logi_idx for i in logi_idx]
        N = H0.shape[0]
        u = np.zeros([N * N, dimz * dimz, len(tlist)], dtype=complex)

        if num_cpus > 1:
            output = qt.parallel.parallel_map(
                parallel_mesolve, range(dimz * dimz),
                task_args=(N, H, tlist, c_op_list, pulse_args, option_noisy, proj_idx),
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
                    H, rho0, tlist, c_op_list, args=pulse_args, options=option_noisy, _safe_mode=False
                )
                for k, t in enumerate(tlist):
                    u[:, n, k] = qt.superoperator.mat2vec(output.states[k].full()).T

        return [qt.Qobj(u[:, :, k], dims=[[[N], [N]], [[dimz], [dimz]]]) for k in range(len(tlist))][-1]

def get_fidelity_super_operator(
    propagator, logi_idx, gate_target, c_op_list, mid_state=None,
    use_qt_fidelity=True,
):
    """
    Computes the average gate fidelity for a given superoperator.

    Args:
        propagator (qt.Qobj): The superoperator representing the quantum operation.
        logi_idx (list): List of logical states indices to consider in the truncated subspace.
        gate_target (qt.Qobj): Target quantum gate to compare against.
        c_op_list (list): Collapse operators for modeling noise.
        mid_state (qt.Qobj, optional): A mid-state to use for fidelity calculation in cnot gate. Defaults to None.
    Returns:
        float: The average gate fidelity of the operation.
    """
    if len(c_op_list) == 0: 
        # Ideal system
        if gate_target == cz_gate():
            kraus = qt.to_kraus(qt.to_super(propagator))
            # print(f'qt.to_super(propagator)={qt.to_super(propagator)}')
            propagator = qt.kraus_to_super(cz_phase_correct(kraus))
        if gate_target == cnot():
            kraus = qt.to_kraus(qt.to_super(propagator))
            # print(f'qt.to_super(propagator)={qt.to_super(propagator)}')
            # print(f'kraus={kraus}')
            propagator = qt.kraus_to_super(cnot_phase_correct(kraus, mid_state=mid_state))         
        super_op_post = qt.to_super(propagator)
        if gate_target == XI:
            kraus = qt.to_kraus(qt.to_super(propagator))
            propagator = qt.kraus_to_super(XI_phase_correct(kraus))         
        super_op_post = qt.to_super(propagator)        
    else:
        # Noisy system, convert to Kraus operators and then to superoperator
        # QuTiP 4's Choi-to-Kraus conversion takes sqrt of small negative
        # eigenvalues caused by solver roundoff.  Clip those numerical
        # negatives to zero before constructing the Kraus operators.
        choi = qt.to_choi(propagator)
        vals, vecs = np.linalg.eigh(choi.full())
        shape = (
            int(np.prod(choi.dims[0][1])),
            int(np.prod(choi.dims[0][0])),
        )
        kraus_dims = choi.dims[0][::-1]
        kraus = [
            qt.Qobj(
                np.sqrt(val) * vec.reshape(shape, order="F"),
                dims=kraus_dims,
            )
            for val, vec in zip(vals, vecs.T)
            if val > 1e-9
        ]
        kraus = [truncate_2(i, logi_idx) for i in kraus]    
        if gate_target == cz_gate():
            kraus = cz_phase_correct(kraus)    
        if gate_target == cnot():
            kraus = cnot_phase_correct(kraus, mid_state=mid_state)    
        super_op_post = qt.kraus_to_super(kraus)

    if not use_qt_fidelity:
        fidelity_input = super_op_post if len(c_op_list) == 0 else kraus
        return average_gate_fidelity_trace_decreasing(
            fidelity_input, target=gate_target
        )

    if len(c_op_list) == 0:
        return qt.metrics.average_gate_fidelity(
            super_op_post, target=gate_target
        )

    # This is QuTiP's trace-preserving average_gate_fidelity formula,
    # evaluated on the Kraus list directly to avoid another unstable
    # Kraus -> superoperator -> Choi -> Kraus round trip in QuTiP 4.
    d = kraus[0].shape[0]
    overlap = sum(
        np.abs((K * gate_target.dag()).tr()) ** 2 for K in kraus
    )
    return float(np.real_if_close((d + overlap) / (d * (d + 1))))

# def xgate_fidelity_optimize(arg, *args):
#     [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, tg, drag] = args
#     alpha_B = 0
#     if drag == 0:
#         alpha_A = 0
#         [drive_amp_A, drive_amp_B, detune_A, detune_B] = arg
#     else:
#         [drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = arg

#     n_cpu = 1
#     argz = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, tg,
#             drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
#     return xgate_fidelity_log(argz)

# def xgate_fidelity_parallel(arg, *args):
#     """Compute the X-gate fidelity using parallel processing."""
#     [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu] = args
#     [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = arg

#     argz = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu,
#             tg, drive_amp_A, drive_amp_B, detune_A, detune_B]
#     return xgate_fidelity_log(argz)

###################################################################
# Fast MESolve functions

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

def print_trans_freq(state_i, state_j, evals, hspace, n_Theta=None, n_Phi=None):
    """
    Display the transition frequency and phase/momentum matrix elements between two states in a quantum system.

    Parameters:
    -----------
    state_i : int
        The index of the initial quantum state.
    state_j : int
        The index of the final quantum state.
    evals : array-like
        The energy eigenvalues of the quantum system. `evals[state_i]` and `evals[state_j]` are
        the energy levels of the initial and final states, respectively.
    n_Theta : 2D array-like, optional
        The matrix of angular momentum-like operator elements. `n_Theta[state_j, state_i]` is the
        transition matrix element for the operator. If None, only `n_Phi` will be displayed.
    n_Phi : 2D array-like, optional
        The matrix of phase/momentum operator elements. `n_Phi[state_j, state_i]` is the transition
        matrix element for the operator. If None, only `n_Theta` will be displayed.

    Returns:
    --------
    None
        The function displays a LaTeX-formatted string using `IPython.display.Math`, including the
        transition frequency and relevant matrix elements.
    """
    index_i = hspace.index(state_i)
    index_j = hspace.index(state_j)

    w_i_j = np.abs(evals[index_j] - evals[index_i])
    n_Theta_i_j = np.abs(n_Theta[index_j, index_i]) if n_Theta is not None else None
    n_Phi_i_j = np.abs(n_Phi[index_j, index_i]) if n_Phi is not None else None

    if n_Theta is not None:
        display(Math(f"\omega_{{{state_i}\_{state_j}}} = %.3f \cdot 2\pi\ GHz = %.3f\ GHz," %(w_i_j/(2*np.pi), w_i_j)
                    + f"\ n_\\theta^{{{state_i}\_{state_j}}} = %.3f \cdot 2\pi = %.3f" %(n_Theta_i_j/(2*np.pi), n_Theta_i_j) ))
    else:
        display(Math(f"\omega_{{{state_i}\_{state_j}}} = %.3f \cdot 2\pi\ GHz = %.3f\ GHz," %(w_i_j/(2*np.pi), w_i_j)
                    + f"\ n_{{\phi}}^{{{state_i}\_{state_j}}} = %.4f \cdot 2\pi = %.4f" %(n_Phi_i_j/(2*np.pi), n_Phi_i_j) ))


def top_population(pop_integral, hspace_full, num=10):
    arr = pop_integral
    # Get the indices of the top 10 largest values
    top_indices = np.argsort(arr)[-num:]  # Sort and take the last 10 indices
    # Get the top 10 largest values
    top_values = arr[top_indices]
    # Sorting in descending order (optional)
    sorted_order = np.argsort(top_values)[::-1]
    top_indices = top_indices[sorted_order]
    top_values = np.round(top_values[sorted_order] / np.sum(top_values), 4)
    print("Top 10 values:", top_values.tolist())
    print("State of large population:", np.array(hspace_full)[top_indices.tolist()])
    return None


def get_w_trans(evals, hspace_full, n_op, state_pop):
    w_trans = []
    trans_int = []
    n_trans = []
    for state_i in state_pop:
        i = hspace_full.index(state_i)
        for j, state_j in enumerate(hspace_full):
            w_trans_ij = np.round( np.abs(evals[i] - evals[j]), 8)
            if w_trans_ij not in w_trans and np.abs(n_op[i, j]/(2*np.pi)) > 0.001:
                w_trans.append(w_trans_ij)
                trans_int.append((state_i, state_j))
                n_trans.append( np.round( np.abs(n_op[i, j]/(2*np.pi)) , 3))
    sort_idx = np.argsort(w_trans)
    w_trans = np.sort(np.round(w_trans, 3))
    n_trans = [np.round(n_trans[i],3) for i in sort_idx]
    trans_int = [trans_int[i] for i in sort_idx]
    return w_trans, n_trans, trans_int


def get_jump_op(state_i, *args):
    truc1, gamma_decay, gamma_dephase, eket_tot = args
    # t_1
    ladder_0i = qt.basis(truc1,0) * qt.basis(truc1, state_i).dag()
    a_0i_I = qt.tensor(ladder_0i, qt.qeye(truc1))
    a_I_0i = qt.tensor(qt.qeye(truc1), ladder_0i)
    jump_t1_a = qt.Qobj( eket_tot @ ( np.sqrt(gamma_decay[state_i])* a_0i_I ).data @ eket_tot.conj().T )
    jump_t1_b = qt.Qobj( eket_tot @ ( np.sqrt(gamma_decay[state_i])* a_I_0i ).data @ eket_tot.conj().T )
    # t_phi
    proj_ii = qt.basis(truc1, state_i).proj()
    a_ii_I = qt.tensor(proj_ii, qt.qeye(truc1))
    a_I_ii = qt.tensor(qt.qeye(truc1), proj_ii)
    jump_tphi_a = qt.Qobj( eket_tot @ ( np.sqrt(2*gamma_dephase[state_i] )* a_ii_I  ).data @ eket_tot.conj().T)
    jump_tphi_b = qt.Qobj( eket_tot @ ( np.sqrt(2*gamma_dephase[state_i] )* a_I_ii  ).data @ eket_tot.conj().T)
    return [jump_t1_a, jump_t1_b, jump_tphi_a, jump_tphi_b]

# def get_jump_op_charge_pick(state_vec, *args):
#     state_i, state_j = state_vec
#     dim_0, dim_1, n_theta, gamma_dephase, eket_tot, qubit_a = args
#     if qubit_a:
#         # t_1
#         ladder_ij = qt.basis(dim_0, state_i) * qt.basis(dim_0, state_j).dag()
#         a_ij_I = qt.tensor(ladder_ij, qt.qeye(dim_1))
#         jump_t1 = qt.Qobj( eket_tot @ ( np.sqrt(abs(n_theta[state_i, state_j]))* a_ij_I ).data @ eket_tot.conj().T )
#         # t_phi
#         proj_jj = qt.basis(dim_0, state_j).proj()
#         a_jj_I = qt.tensor(proj_jj, qt.qeye(dim_1))
#         jump_tphi = qt.Qobj( eket_tot @ ( np.sqrt(2*gamma_dephase[state_j] )* a_jj_I  ).data @ eket_tot.conj().T)
#     else: # qubit_b
#         # t_1
#         ladder_ij = qt.basis(dim_1, state_i) * qt.basis(dim_1, state_j).dag()
#         a_I_ij = qt.tensor(qt.qeye(dim_0), ladder_ij)
#         jump_t1 = qt.Qobj( eket_tot @ ( np.sqrt(abs(n_theta[state_i, state_j]))* a_I_ij ).data @ eket_tot.conj().T )
#         # t_phi
#         proj_jj = qt.basis(dim_1, state_j).proj()
#         a_I_jj = qt.tensor(qt.qeye(dim_0), proj_jj)
#         jump_tphi = qt.Qobj( eket_tot @ 
#                             ( np.sqrt(2*gamma_dephase[state_j] )* a_I_jj  ).data @
#                               eket_tot.conj().T)
#     return [jump_t1, jump_tphi]

def get_transitions_for_collapse(hspace, n_theta, low_states=50, filter_ratio=0.01):
    low = [s for s in hspace if s < low_states]
    n_theta_low = truncate_2(n_theta, low)
    n_max = np.abs(n_theta_low.data).max()

    n_theta_trunc = truncate_2(n_theta, hspace)
    def ratio(sj):
        return filter_ratio if sj < low_states else 10 * filter_ratio
    transition = [
        (i, j)
        for i, si in enumerate(hspace)
        for j, sj in enumerate(hspace)
        if i < j and abs(n_theta_trunc[i, j]) > n_max * ratio(sj)
    ]
    return transition, n_theta_trunc

def get_jump_op_decay(transition, *args):
    state_i, state_j = transition
    dim_0, dim_1, n_theta_trunc, eket_tot, qubit_a, Gamma_decay = args
    decay_rate = np.sqrt(Gamma_decay * abs(n_theta_trunc[state_i, state_j])**2)
    if qubit_a:
        ladder_ij = qt.basis(dim_0, state_i) * qt.basis(dim_0, state_j).dag()
        jump_t1 = ( decay_rate* qt.tensor(ladder_ij, qt.qeye(dim_1)) )
    else: # qubit_b
        ladder_ij = qt.basis(dim_1, state_i) * qt.basis(dim_1, state_j).dag()
        jump_t1 = ( decay_rate* qt.tensor(qt.qeye(dim_0), ladder_ij) )        
    jump_t1 = qt.Qobj( eket_tot @ jump_t1.data @ eket_tot.conj().T )
    return jump_t1

def get_jump_op_dephase(state_j, *args):
    dim_0, dim_1, gamma_dephase, eket_tot, qubit_a = args
    dephasing_rate = np.sqrt(2 * gamma_dephase[state_j])
    if qubit_a:
        proj_jj = qt.basis(dim_0, state_j).proj()
        jump_tphi = dephasing_rate* qt.tensor(proj_jj, qt.qeye(dim_1))
    else: # qubit_b
        proj_jj = qt.basis(dim_1, state_j).proj()
        jump_tphi = dephasing_rate* qt.tensor(qt.qeye(dim_0), proj_jj)
    jump_tphi = qt.Qobj( eket_tot @ jump_tphi.data @ eket_tot.conj().T)
    return jump_tphi

def construct_c_ops_2q(dim_0, dim_1, n_theta0_trunc, n_theta1_trunc, 
                        gamma_dephase_02_q0, gamma_dephase_02_q1, eket_tot, 
                        Gamma_decay_q0, Gamma_decay_q1, 
                        transition_a, transition_b, 
                        apply_decay=True, apply_dephase=True, decay_enlarge=1):
    """
    Constructs collapse operators for dissipation.
    """
    jump_t1_a, jump_tphi_a, jump_t1_b, jump_tphi_b = [], [], [], []
    qubit_a = True
    if apply_decay:
        arg_a_decay = [dim_0, dim_1, n_theta0_trunc, eket_tot, qubit_a, Gamma_decay_q0] 
        jump_t1_a = Parallel(n_jobs=10)(delayed(get_jump_op_decay)
                                        (transition, *arg_a_decay) 
                                        for transition in transition_a)
    if apply_dephase:
        arg_a_dephase = [dim_0, dim_1, gamma_dephase_02_q0, eket_tot, qubit_a] 
        jump_tphi_a = Parallel(n_jobs=10)(delayed(get_jump_op_dephase)
                                        (state_j, *arg_a_dephase) 
                                        for state_j in range(1,dim_0))

    qubit_a = False
    if apply_decay:
        arg_b_decay = [dim_0, dim_1, n_theta1_trunc, eket_tot, qubit_a, Gamma_decay_q1] 
        jump_t1_b = Parallel(n_jobs=10)(delayed(get_jump_op_decay)
                                        (transition, *arg_b_decay) 
                                        for transition in transition_b)
    if apply_dephase:
        arg_b_dephase = [dim_0, dim_1, gamma_dephase_02_q1, eket_tot, qubit_a] 
        jump_tphi_b = Parallel(n_jobs=10)(delayed(get_jump_op_dephase)
                                        (state_j, *arg_b_dephase) 
                                        for state_j in range(1,dim_1))
    
    # print(f'decay_enlarge = {decay_enlarge}')
    # print(f'len(jump_tphi_a) = {len(jump_tphi_a)}, len(jump_tphi_b) = {len(jump_tphi_b)}')
    # print(f'len(jump_t1_a) = {len(jump_t1_a)}, len(jump_t1_b) = {len(jump_t1_b)}')
    return (jump_tphi_a + jump_tphi_b 
            + [jump_t1_a[i] * decay_enlarge for i in range(len(jump_t1_a))]
            + [jump_t1_b[i] * decay_enlarge for i in range(len(jump_t1_b))] )


def zeropi_eval(flux=0, truncation=10):
    ncut, phi_cut = 90, 300
    # Define system parameters (in GHz)
    EL = 0.377  # Inductive energy
    EJ = 6.013  # Josephson energy # 5.412, 6.013
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
        flux=flux,
        ncut=ncut,
        truncated_dim=truncation,
    )

    # Compute the eigenvalues and construct the Hamiltonian
    evals, _ = zero_pi.eigensys(evals_count=truncation)
    evals = np.sort(evals)
    evals = evals - evals[0]
    return evals


# def get_collapse_op(t1tphi_other, eket_tot_select, hspace_0_dim, hspace_1_dim):

    # tphi_logi = 100 # μs
    # gamma_decay_logi =  1 / 1600e3
    # gamma_dephase_logi = 1 / 1e3 / tphi_logi
    # gamma_decay_other =  1 / 1e3 / t1tphi_other
    # gamma_dephase_other = 1 / 1e3 / t1tphi_other
    # gamma_decay_old   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * 300
    # gamma_dephase_old = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * 300

    # if charge_pick == False:
    #     args = [truc1, gamma_decay_old, gamma_dephase_old, eket_tot]
    #     jump_op = Parallel(n_jobs=100)(delayed(get_jump_op)(state, *args) for state in range(1,truc1))
    #     jump_t1 = np.array(jump_op)[:,:2]
    #     jump_tphi = np.array(jump_op)[:,2:]
    #     jump_t1_list = [qt.Qobj(matrix) for row in jump_t1 for matrix in row]
    #     jump_tphi_list = [qt.Qobj(matrix) for row in jump_tphi for matrix in row]

    # if drive_theta:
    #     gamma_decay   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (hspace_len-3)
    #     gamma_dephase = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (hspace_len-3)
    # else:
    #     gamma_decay   = [0,  gamma_decay_logi]  + [gamma_decay_other]  * (hspace_len-2)
    #     gamma_dephase = [0,  gamma_dephase_logi] + [gamma_dephase_other] * (hspace_len-2)

    # print("gamma_decay_logi = ", gamma_decay_logi, ", gamma_dephase_logi = ", gamma_dephase_logi)
    # print("gamma_decay_other = ", gamma_decay_other, ", gamma_dephase_other = ", gamma_dephase_other)
    # print(f"T1_logi = {1/gamma_decay_logi} ns") if gamma_decay_logi != 0 else None
    # print(f"Tphi_logi = {1/gamma_dephase_logi} ns") if gamma_dephase_logi != 0 else None
    # print(f"T1_other = {1/gamma_decay_other} ns") if gamma_decay_other != 0 else None
    # print(f"Tphi_other = {1/gamma_dephase_other} ns") if gamma_dephase_other != 0 else None

    # folder = f'../data/3ncut_two_zeropi/truc1=500/'
    # gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    # gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    # gamma_decay_48_q0 = gamma_q0['t1_50us_48'].to_numpy() *50 /t1tphi_other
    # gamma_decay_48_q1 = gamma_q1['t1_50us_48'].to_numpy() *50 /t1tphi_other
    # gamma_dephase_02_q0 = gamma_q0['tphi_02'].to_numpy() *50 /t1tphi_other
    # gamma_dephase_02_q1 = gamma_q1['tphi_02'].to_numpy() *50 /t1tphi_other

    # qubit_a = True
    # arg_a = [hspace_0_dim, hspace_1_dim, gamma_decay_48_q0, gamma_dephase_02_q0, eket_tot_select, qubit_a]
    # jump_op_a = Parallel(n_jobs=100)(delayed(get_jump_op_charge_pick)(state, *arg_a) for state in range(1,hspace_0_dim))

    # qubit_a = False
    # arg_b = [hspace_0_dim, hspace_1_dim, gamma_decay_48_q1, gamma_dephase_02_q1, eket_tot_select, qubit_a]
    # jump_op_b = Parallel(n_jobs=100)(delayed(get_jump_op_charge_pick)(state, *arg_b) for state in range(1,hspace_1_dim))
    # jump_t1_list = np.array(jump_op_a)[:,0].tolist() + np.array(jump_op_b)[:,0].tolist()
    # jump_tphi_list = np.array(jump_op_a)[:,1].tolist() + np.array(jump_op_b)[:,1].tolist()
    # jump_t1_list = [qt.Qobj(matrix) for matrix in jump_t1_list]
    # jump_tphi_list = [qt.Qobj(matrix) for matrix in jump_tphi_list]
    # c_op_list = jump_t1_list + jump_tphi_list

    # return c_op_list

def print_fidelity(label, data, num_each_row=4, num_digits=None, 
               n_make_blank_line=None):
    """
    Pretty-print fidelity array in readable blocks.
    Parameters:
        label (str): Label for the data array.
        num_each_row (int): Entries per row in output.
    """
    print(f"\n{label} = np.array([")
    for i in range(0, len(data), num_each_row):
        if n_make_blank_line != None and i%n_make_blank_line==0:
            print('')
        if isinstance(data[0], str):
            print(", ".join(f"'{x}'" for x in data[i:i + num_each_row]), ',')
        else:
            # if num_digits != None and isinstance(data[0], float):
            #     print(', '.join(map(str, np.round(data[i:i + num_each_row], num_digits).tolist())), ',')
            if num_digits is not None and isinstance(data[0], float):
                print(', '.join(f"{x:.{num_digits}f}" for x in data[i:i + num_each_row]), ',')                
            else:
                print(', '.join(map(str, data[i:i + num_each_row])), ',')
    print('])')

def print_pulse_params(label, params):
    """Print optimized drive parameters."""
    print(f"{label} = np.array([")    
    for i in params:
        print(np.round(i, 6).tolist(), ',')
    print("])")

def print_time():
    """Print the current time."""
    # # Get current time in Mountain Time (Denver)
    # mountain_tz = pytz.timezone('America/Denver')
    # mountain_time = datetime.now(mountain_tz)

    # Convert to Beijing Time
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)    

    print("\nCurrent China Time:", beijing_time)    


def build_hamiltonian_2q(cz_run, index_select, eval_tot, eket_tot, drive_term):
    """
    Constructs the truncated Hamiltonian and drive terms.
    """
    H0_full = qt.Qobj(np.diag(eval_tot))
    H0_select = truncate_2( H0_full, index_select)
    eket_tot = eket_tot[index_select]
    drive_select = truncate_2(drive_term, index_select)
    if cz_run:
        H_drive_select = [ H0_select,   [drive_select, drive_gauss_A] ]
    else:
        H_drive_select = [ H0_select,   [drive_select, drive_gauss_A],
                                        [drive_select, drive_gauss_B]  ]          
    return H_drive_select, eket_tot


def compare_two_lists(list1, list2):
    only_in_list1 = [item for item in list1 if item not in list2]
    only_in_list2 = [item for item in list2 if item not in list1]
    unique_elements = only_in_list1 + only_in_list2
    common_elements = [item for item in list1 if item in list2]

    print(f'list1 and list2 have {len(common_elements)} common elements'
          +f' and {len(list1) - len(common_elements)} unique elements.')

    print_fidelity(f'Only in list1 (len={len(only_in_list1)})', only_in_list1, 
                  num_each_row=10, n_make_blank_line=50)
    list1_index = [list1.index(i) for i in only_in_list1]
    print_fidelity(f'index Only in list1 (len={len(only_in_list1)})', list1_index, 
                  num_each_row=10, n_make_blank_line=50)
    
    print_fidelity(f'Only in list2 (len={len(only_in_list2)})', only_in_list2, 
                  num_each_row=10, n_make_blank_line=50)    
    list2_index = [list2.index(i) for i in only_in_list2]
    print_fidelity(f'index Only in list2 (len={len(only_in_list2)})', list2_index, 
                  num_each_row=10, n_make_blank_line=50)    


def max_index_2d_array(arr, arr_name=None):
    row, col = np.unravel_index(np.argmax(arr), arr.shape)
    print(f'{arr_name}.shape = {np.shape(arr)}, '
          +f'maximum = {np.round(abs(arr[row, col]), 8)}, '
          +f'row={row}, column={col}')

def get_qutip_options(max_step_ideal, max_step_noisy, num_cpus=1, print_flag=False):
    """
    Get the qutip options 
    """
    if max_step_ideal != 0:
        nsteps_ideal = int(1/ max_step_ideal )  # Set nsteps to a large number for serial execution
    else:
        nsteps_ideal = 1e5  # Set nsteps to a large number for parallel execution

    if max_step_noisy != 0:
        nsteps_noisy = 1e5 # int(1/ max_step_noisy )  # Set nsteps to a large number for serial execution
    else:
        nsteps_noisy = 1e5  # Set nsteps to a large number for parallel execution        

    option_ideal =qt.Options(max_step=max_step_ideal, nsteps=nsteps_ideal, num_cpus=num_cpus)  
    option_noisy =qt.Options(max_step=max_step_noisy, nsteps=nsteps_noisy, num_cpus=num_cpus) 
    if print_flag:
        print(f'Ideal: max_step = {option_ideal.max_step}, nsteps = {option_ideal.nsteps}')
        print(f'Noisy: max_step = {option_noisy.max_step}, nsteps = {option_noisy.nsteps}')
    return option_ideal, option_noisy

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

def load_qubit_data_xgate(qubit_0 = True, folder = '../figure/data/'):
    """
    folder = '../../data/3ncut_one_zeropi/'
    Loads the energy spectrum and matrix elements (n_theta, n_phi) for the 0-π qubit.
    The function "generate_data()" in sigmaX_fidelity_import_paras.py can generate the data

    Parameters:
        qubit_0 (bool): If True, load data for qubit 0; otherwise, load for qubit 1.
        folder (str): Path to the directory containing the data files.
    Returns:
        evals (np.ndarray): Energy levels.
        n_theta (np.ndarray): Matrix elements for theta drive.
        n_phi (np.ndarray): Matrix elements for phi drive.
    """    
    suffix = '0' if qubit_0 else '1'
    evals = 2 * np.pi * scq.read(folder + f'zeropi_{suffix}_specdata_truc=1000_3ncut.h5').energy_table
    n_theta = 2 * np.pi * scq.read(folder + f'zeropi_{suffix}_n_theta_truc=1000_3ncut.h5').matrixelem_table
    n_phi = 2 * np.pi * scq.read(folder + f'zeropi_{suffix}_n_phi_truc=1000_3ncut.h5').matrixelem_table
    evals -= evals[0]
    logi_state = [0, 2]
    return evals, n_theta, n_phi, logi_state

    ##############################################################################################
    #### data loading functions ######################################################
    ##############################################################################################

def load_dephasing_data_xgate(drive_theta, q0 = True):
    """
    Load dephasing rates calculated for 50μs.

    Parameters:
        drive_theta (bool): If True, load theta dephasing; otherwise, phi.

    Returns:
        np.ndarray: Dephasing rates for each state.
    """
    # print('use 1st order dephasing data')
    # gamma_file = 'data/data_gamma_theta_500.txt' if drive_theta else 'data/data_gamma_phi_500.txt'
    # gamma_new = pd.read_csv(gamma_file)
    # gamma_dephase = gamma_new['tphi_50us_02'].to_numpy()
    # state_idx = gamma_new['hspace'].to_numpy()

    print('use 2nd order dephasing data, correct 50us factor')
    gamma_file = '../data/data_flux_derivative_q0_eval_500.npz' if q0 else '../data/data_flux_derivative_q1_eval_500.npz'
    gamma_new = np.load(gamma_file)
    gamma_dephase_label = 'gamma_phi_1us_02_theta_mode' if drive_theta else 'gamma_phi_1us_02_phi_mode'
    gamma_dephase = np.abs(gamma_new[gamma_dephase_label]) / 50
    state_idx_label = 'hspace_theta_256' if drive_theta else 'hspace_phi_263'
    state_idx = gamma_new[state_idx_label]
    return state_idx, gamma_dephase

def load_drive_params_xgate(drive_theta, drive_phi):
    """
    Load X-gate drive parameters from a CSV file.

    Parameters:
        drive_theta (bool): If True, load theta-drive data; else load phi-drive data.

    Returns:
        np.ndarray: Parameters array. different rows mean different gate time. 
        columns mean 'tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
    """

    if drive_phi and not drive_theta:
        folder = 'data_xgate_phi_3ncut_mstep_1e3.txt'
    elif drive_theta and not drive_phi:
        folder = 'data_xgate_theta_3ncut_mstep_3e4.txt'
    else:
        return np.array([[828.759495, 0.013563, 0.034964, -0.003029, -0.003182]])

    folder = 'data_xgate_theta_mstep_3e4_npz.txt' if drive_theta else 'data_xgate_phi_mstep_1e3_npz.txt'
    f_xgate = pd.read_csv('../figure/data/' + folder)
    return f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2']].to_numpy()

def load_1q_data_for_2q(truc1, folder = '../../data/3ncut_two_zeropi/truc1=500/'):
    
    eval0 = pd.read_csv(folder+ 'eval0.txt').to_numpy().flatten()
    eval1 = pd.read_csv(folder+ 'eval1.txt').to_numpy().flatten()
    n_theta0 = np.load(folder+'n_theta0.npy')
    n_theta1 = np.load(folder+'n_theta1.npy')

    eval0 = eval0[:truc1]
    eval1 = eval1[:truc1]
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    n_theta0 = truncate_2(n_theta0, hspace_0)
    n_theta1 = truncate_2(n_theta1, hspace_1)
    return eval0,eval1,n_theta0,n_theta1

def load_qubit_data_2q(truc1=300, truc_full = 2000, charge_pick=True,**kwargs):
    """
    Loads the energy spectrum and matrix elements (n_theta, n_phi) for the 0-π qubit.
    The function "generate_data()" in sigmaX_fidelity_import_paras.py can generate the data
    """    
     # If charge_pick = True, n_full=2000, else 1000
    folder = kwargs.get("folder", f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2=2000_pick={charge_pick}/')
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    if charge_pick:
        hspace_0 = pd.read_csv(folder+ 'hspace_0.txt').to_numpy().flatten()
        hspace_1 = pd.read_csv(folder+ 'hspace_1.txt').to_numpy().flatten()             
    else:
        hspace_0 = np.arange(truc1)
        hspace_1 = np.arange(truc1)
   
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()[:truc_full]
    eket_tot = ssp.csr_matrix(np.load(folder+ 'eket_tot.npy'))[:truc_full]
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()[:truc_full]
    n_theta0_dress = 2*np.pi* np.load(folder+'n_theta0_dress.npy')
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')
    n_theta0_dress = truncate_2(n_theta0_dress, np.arange(truc_full))
    n_theta1_dress = truncate_2(n_theta1_dress, np.arange(truc_full))   
    return [hspace_full, eket_tot, eval_tot, n_theta0_dress, 
            n_theta1_dress, hspace_0, hspace_1, logi_state]

def load_drive_params_2q(cz_run, folder=None):
    """
    Load two-qubit-gate drive parameters from a CSV file.
    """
    if cz_run:
        if folder is None:
            folder = '../figure/data/data_cz_3ncut_truc1=300.txt'
        params = pd.read_csv(folder)[['tg', 'drive_amp', 'detune']].to_numpy()
    else:
        if folder is None:
            # folder = '../cnot/data/data_cnot_fidelity_3ncut.txt'
            folder = '../figure/data/data_cnot_fidelity_npz.txt'
        params = pd.read_csv(folder)[['tg', 'drive_amp_1', 'drive_amp_2', 
                                      'detune_1', 'detune_2']].to_numpy()
    return params


def load_noise_data_2q(t1_tphi_other, 
                    folder = '../../data/3ncut_two_zeropi/truc1=500/'):
    """
    Load dephasing rates calculated for 50μs.    
    """    
    gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    n_theta0 = np.load(folder+'n_theta0.npy')
    n_theta1 = np.load(folder+'n_theta1.npy')
    gamma_dephase_02_q0 = gamma_q0['tphi_02'].to_numpy() *50 /t1_tphi_other
    gamma_dephase_02_q1 = gamma_q1['tphi_02'].to_numpy() *50 /t1_tphi_other
    return n_theta0, n_theta1, gamma_dephase_02_q0, gamma_dephase_02_q1

def average_gate_fidelity_trace_decreasing(oper, target=None):
    """
    Average gate fidelity for a possibly trace-decreasing quantum map.

    The quantum map is

        E(rho) = sum_k K_k rho K_k.dag()

    and the average fidelity relative to target unitary U is

        F_avg =
            [sum_k Tr(K_k.dag() K_k)
             + sum_k |Tr(U.dag() K_k)|^2]
            / [d(d + 1)].

    For a trace-preserving map,

        sum_k K_k.dag() K_k = I,

    and the first term reduces to d, recovering the standard
    average-gate-fidelity formula.

    Parameters
    ----------
    oper : qutip.Qobj or list[qutip.Qobj]
        A square operator, superoperator, or list of Kraus operators.

        A square non-unitary operator V is interpreted as the
        one-Kraus map

            rho -> V rho V.dag().

    target : qutip.Qobj or None
        Target unitary. If None, the identity is used.

    Returns
    -------
    fidelity : float
        Leakage-inclusive average gate fidelity.
    """

    # Convert input to Kraus representation.
    if isinstance(oper, (list, tuple)):
        kraus_form = list(oper)

    elif isinstance(oper, qt.Qobj):
        if oper.issuper:
            kraus_form = qt.to_kraus(oper)
        elif oper.isoper:
            kraus_form = [oper]
        else:
            raise TypeError(
                "oper must be an operator, superoperator, "
                "or a list of Kraus operators."
            )

    else:
        raise TypeError(
            "oper must be a qutip.Qobj or a list of qutip.Qobj."
        )

    if len(kraus_form) == 0:
        raise ValueError("The Kraus list is empty.")

    d_out, d_in = kraus_form[0].shape

    if d_out != d_in:
        raise ValueError(
            "Only square Kraus operators are supported."
        )

    d = d_in

    for i, K in enumerate(kraus_form):
        if not isinstance(K, qt.Qobj):
            raise TypeError(
                f"Kraus operator {i} is not a qutip.Qobj."
            )

        if K.shape != (d, d):
            raise ValueError(
                f"Kraus operator {i} has shape {K.shape}; "
                f"expected {(d, d)}."
            )

    if target is None:
        target = qt.qeye(d)

    if not isinstance(target, qt.Qobj):
        raise TypeError("target must be a qutip.Qobj.")

    if target.shape != (d, d):
        raise ValueError(
            f"target has shape {target.shape}; expected {(d, d)}."
        )

    # Average survival contribution.
    survival_term = sum(
        np.real((K.dag() * K).tr())
        for K in kraus_form
    )

    # Target-overlap contribution.
    overlap_term = sum(
        np.abs((target.dag() * K).tr()) ** 2
        for K in kraus_form
    )

    fidelity = (
        survival_term + overlap_term
    ) / (d * (d + 1))

    return float(np.real_if_close(fidelity))

hspace_theta_256 = [
0, 1, 2, 4, 5, 7, 8, 11, 12, 16 ,
17, 18, 21, 23, 25, 27, 30, 31, 32, 33 ,
37, 38, 40, 41, 42, 45, 46, 47, 48, 51 ,
52, 56, 59, 62, 63, 64, 65, 67, 72, 73 ,
74, 76, 78, 79, 82, 83, 85, 87, 88, 90 ,

92, 93, 96, 97, 98, 102, 103, 106, 107, 112 ,
114, 118, 119, 120, 121, 122, 123, 124, 127, 128 ,
131, 132, 133, 134, 137, 138, 142, 143, 150, 151 ,
154, 155, 156, 157, 158, 161, 162, 163, 166, 170 ,
171, 174, 175, 176, 177, 179, 180, 186, 187, 188 ,

191, 192, 195, 196, 199, 202, 203, 206, 207, 208 ,
210, 213, 214, 218, 219, 220, 221, 222, 225, 226 ,
230, 231, 232, 236, 237, 238, 242, 243, 246, 247 ,
248, 251, 252, 258, 259, 260, 262, 263, 264, 270 ,
271, 274, 275, 279, 280, 281, 283, 284, 285, 288 ,

289, 290, 294, 296, 297, 298, 299, 300, 301, 302 ,
307, 308, 312, 313, 314, 315, 324, 325, 326, 327 ,
328, 333, 334, 337, 338, 339, 340, 345, 346, 347 ,
348, 350, 351, 354, 355, 358, 359, 364, 365, 366 ,
367, 372, 373, 374, 375, 376, 377, 378, 381, 384 ,

385, 388, 389, 390, 398, 399, 400, 401, 407, 408 ,
410, 411, 413, 414, 418, 419, 420, 425, 426, 427 ,
428, 431, 432, 433, 436, 437, 438, 439, 442, 446 ,
447, 450, 451, 452, 453, 454, 455, 459, 460, 463 ,
464, 467, 468, 472, 473, 474, 475, 476, 486, 488 ,

489, 492, 493, 494, 496, 497 ,
]

hspace_phi_263 = [
0, 2, 3, 4, 8, 9, 10, 11, 15, 16 ,
18, 19, 20, 23, 24, 25, 28, 31, 33, 35 ,
37, 39, 40, 42, 43, 46, 47, 49, 51, 53 ,
55, 56, 57, 60, 62, 64, 66, 67, 69, 70 ,
73, 76, 77, 79, 81, 83, 85, 86, 88, 91 ,

92, 95, 96, 98, 100, 101, 102, 104, 106, 108 ,
110, 112, 113, 116, 118, 120, 121, 123, 126, 128 ,
130, 132, 133, 136, 137, 139, 141, 143, 145, 146 ,
148, 151, 153, 154, 156, 158, 159, 162, 164, 166 ,
167, 169, 170, 173, 174, 176, 179, 181, 183, 185 ,

186, 188, 190, 192, 194, 196, 198, 200, 202, 205 ,
206, 208, 209, 212, 213, 216, 218, 220, 222, 223 ,
226, 227, 229, 230, 232, 234, 237, 239, 241, 242 ,
245, 246, 248, 250, 252, 253, 255, 257, 259, 262 ,
264, 265, 267, 269, 271, 273, 275, 277, 279, 281 ,

282, 283, 286, 288, 290, 291, 293, 294, 297, 299 ,
302, 303, 305, 306, 308, 310, 313, 315, 317, 319 ,
320, 322, 325, 326, 328, 330, 332, 333, 336, 337 ,
340, 341, 343, 346, 348, 349, 350, 352, 354, 357 ,
358, 361, 363, 364, 366, 368, 370, 372, 374, 376 ,

378, 379, 382, 384, 386, 388, 390, 391, 393, 395 ,
397, 398, 400, 403, 405, 407, 409, 411, 414, 415 ,
417, 418, 420, 422, 424, 426, 428, 430, 432, 435 ,
436, 438, 441, 442, 444, 445, 447, 449, 451, 453 ,
455, 457, 460, 461, 464, 465, 468, 469, 471, 472 ,

474, 476, 478, 480, 482, 484, 487, 488, 491, 492 ,
494, 497, 498 ,
]

truc_model = {}
# truc_model['n_theta_dress_charge_truc'] = [
# 0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 14, 16, 20, 21, 25, 26, 27, 28, 31, 33 ,
# 35, 37, 39, 43, 45, 49, 50, 54, 55, 58, 62, 63, 64, 68, 71, 73, 76, 78, 80, 82 ,
# 85, 87, 93, 96, 98, 99, 101, 105, 106, 110, 112, 115, 117, 119, 120, 121, 127, 129, 133, 134 ,
# 135, 136, 139, 140, 144, 146, 151, 155, 160, 163, 167, 169, 171, 172, 174, 181, 182, 187, 189, 190 ,
# 193, 194, 197, 200, 204, 208, 209, 210, 211, 212, 216, 218, 220, 225, 226, 227, 230, 238, 240, 241 ,

# 244, 245, 247, 252, 254, 255, 258, 259, 261, 262, 264, 267, 269, 274, 275, 279, 282, 288, 296, 303 ,
# 304, 305, 306, 308, 310, 312, 315, 321, 323, 324, 333, 334, 340, 343, 344, 346, 353, 354, 356, 359 ,
# 361, 364, 365, 370, 372, 374, 375, 377, 378, 379, 383, 387, 388, 389, 391, 393, 398, 403, 406, 409 ,
# 412, 414, 416, 417, 418, 420, 424, 425, 426, 429, 430, 432, 434, 437, 450, 451, 453, 454, 461, 463 ,
# 466, 468, 470, 472, 473, 475, 483, 486, 494, 495, 496, 501, 502, 514, 524, 527, 528, 535, 536, 537 ,

# 539, 545, 549, 550, 553, 555, 556, 558, 560, 561, 565, 566, 572, 575, 576, 578, 579, 582, 583, 585 ,
# 586, 587, 589, 593, 594, 597, 602, 608, 619, 620, 621, 622, 625, 626, 628, 629, 630, 632, 634, 636 ,
# 641, 644, 646, 648, 656, 657, 658, 664, 671, 676, 679, 680, 685, 689, 691, 693, 695, 697, 701, 703 ,
# 704, 709, 713, 715, 719, 720, 729, 731, 734, 737, 739, 745, 748, 753, 759, 764, 772, 773, 777, 780 ,
# 781, 783, 784, 785, 787, 790, 793, 798, 799, 801, 802, 806, 816, 820, 823, 824, 825, 826, 833, 835 ,

# ### 300-600
# 840, 841, 844, 850, 852, 854, 857, 859, 860, 862, 863, 866, 867, 871, 872, 873, 878, 880, 891, 894 ,
# 895, 897, 901, 906, 907, 908, 912, 913, 921, 923, 927, 929, 933, 938, 939, 940, 944, 946, 947, 949 ,
# 952, 953, 955, 958, 971, 981, 983, 984, 986, 988, 998, 1002, 1004, 1019, 1021, 1022, 1024, 1027, 1029, 1032 ,
# 1034, 1037, 1039, 1040, 1042, 1044, 1051, 1052, 1056, 1061, 1064, 1069, 1071, 1072, 1074, 1078, 1079, 1080, 1083, 1084 ,
# 1086, 1087, 1091, 1092, 1095, 1097, 1099, 1103, 1108, 1110, 1112, 1114, 1117, 1125, 1127, 1128, 1134, 1135, 1137, 1138 ,

# 1143, 1149, 1150, 1151, 1153, 1164, 1169, 1175, 1177, 1178, 1179, 1181, 1186, 1189, 1192, 1193, 1200, 1204, 1207, 1211 ,
# 1216, 1217, 1220, 1223, 1224, 1225, 1227, 1228, 1239, 1247, 1249, 1250, 1252, 1253, 1256, 1258, 1260, 1262, 1268, 1271 ,
# 1272, 1278, 1281, 1286, 1287, 1290, 1291, 1292, 1300, 1301, 1303, 1314, 1321, 1322, 1323, 1328, 1333, 1339, 1343, 1344 ,
# 1348, 1355, 1359, 1360, 1362, 1364, 1373, 1374, 1377, 1380, 1384, 1387, 1390, 1400, 1401, 1404, 1407, 1411, 1413, 1415 ,
# 1416, 1419, 1422, 1423, 1425, 1426, 1427, 1428, 1430, 1436, 1437, 1439, 1443, 1446, 1455, 1458, 1462, 1463, 1467, 1468 ,

# 1471, 1473, 1482, 1484, 1487, 1490, 1494, 1495, 1496, 1497, 1503, 1506, 1508, 1509, 1511, 1517, 1522, 1524, 1528, 1532 ,
# 1533, 1543, 1544, 1548, 1555, 1557, 1559, 1565, 1566, 1567, 1570, 1573, 1574, 1575, 1578, 1579, 1580, 1582, 1585, 1587 ,
# 1591, 1594, 1597, 1600, 1601, 1614, 1615, 1618, 1627, 1629, 1634, 1636, 1639, 1642, 1647, 1651, 1652, 1655, 1658, 1659 ,
# 1665, 1666, 1671, 1673, 1677, 1679, 1683, 1684, 1685, 1686, 1692, 1695, 1702, 1705, 1708, 1712, 1714, 1718, 1722, 1723 ,
# 1725, 1726, 1731, 1732, 1735, 1740, 1742, 1745, 1758, 1759, 1760, 1765, 1766, 1767, 1769, 1773, 1775, 1778, 1781, 1789 ,

# ### 600-900
# 1792, 1793, 1797, 1798, 1799, 1810, 1815, 1819, 1820, 1823, 1826, 1827, 1831, 1835, 1837, 1838, 1844, 1848, 1852, 1853 ,
# 1855, 1859, 1864, 1871, 1872, 1874, 1877, 1883, 1884, 1893, 1894, 1898, 1901, 1903, 1905, 1913, 1917, 1918, 1929, 1933 ,
# 1940, 1942, 1943, 1945, 1954, 1961, 1964, 1966, 1968, 1971, 1974, 1976, 1977, 1981, 1985, 1988, 1990, 1991, 1992, 1993 ,
# 1996, 1997, 2000, 2002, 2004, 2005, 2009, 2011, 2014, 2017, 2018, 2020, 2025, 2034, 2036, 2041, 2043, 2052, 2053, 2061 ,
# 2062, 2066, 2071, 2072, 2073, 2078, 2081, 2082, 2084, 2085, 2091, 2092, 2093, 2100, 2103, 2104, 2107, 2112, 2113, 2118 ,

# 2119, 2122, 2128, 2130, 2131, 2136, 2141, 2143, 2145, 2149, 2152, 2158, 2159, 2161, 2162, 2169, 2173, 2174, 2175, 2176 ,
# 2180, 2184, 2185, 2187, 2191, 2205, 2209, 2211, 2213, 2216, 2218, 2220, 2222, 2224, 2225, 2230, 2239, 2240, 2241, 2249 ,
# 2253, 2257, 2259, 2261, 2263, 2268, 2269, 2277, 2282, 2284, 2285, 2290, 2293, 2297, 2300, 2302, 2304, 2309, 2314, 2316 ,
# 2318, 2321, 2322, 2325, 2326, 2333, 2335, 2336, 2340, 2342, 2345, 2352, 2354, 2356, 2360, 2361, 2366, 2367, 2368, 2372 ,
# 2373, 2374, 2375, 2376, 2379, 2383, 2396, 2399, 2403, 2405, 2415, 2418, 2419, 2421, 2432, 2434, 2439, 2441, 2444, 2451 ,

# 2453, 2460, 2464, 2483, 2484, 2487, 2488, 2496, 2497, 2500, 2504, 2505, 2510, 2511, 2513, 2516, 2518, 2526, 2529, 2531 ,
# 2532, 2534, 2544, 2547, 2548, 2551, 2552, 2558, 2560, 2568, 2569, 2573, 2575, 2580, 2582, 2584, 2585, 2590, 2600, 2601 ,
# 2603, 2608, 2612, 2614, 2615, 2616, 2619, 2620, 2621, 2625, 2626, 2633, 2639, 2644, 2645, 2646, 2651, 2655, 2656, 2658 ,
# 2659, 2661, 2665, 2667, 2668, 2670, 2671, 2672, 2675, 2690, 2693, 2694, 2697, 2698, 2704, 2711, 2714, 2722, 2724, 2733 ,
# 2738, 2739, 2743, 2750, 2753, 2755, 2759, 2762, 2763, 2764, 2765, 2766, 2767, 2769, 2773, 2774, 2782, 2785, 2786, 2789 ,

# ### 900-966
# 2796, 2797, 2798, 2801, 2802, 2805, 2806, 2808, 2810, 2817, 2818, 2821, 2825, 2830, 2832, 2833, 2835, 2837, 2839, 2843 ,
# 2844, 2858, 2862, 2864, 2866, 2869, 2871, 2873, 2879, 2881, 2890, 2894, 2903, 2906, 2908, 2912, 2913, 2914, 2918, 2919 ,
# 2921, 2924, 2926, 2927, 2933, 2936, 2944, 2946, 2949, 2951, 2956, 2958, 2960, 2962, 2966, 2967, 2969, 2977, 2979, 2980 ,
# 2986, 2987, 2990, 2994, 2995, 2999 ,
# ]
