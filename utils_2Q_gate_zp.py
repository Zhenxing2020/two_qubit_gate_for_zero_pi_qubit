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
    
    fidelity = get_fidelity_super_operator(propagator, logi_idx, cnot(), c_op_list, mid_state)
    
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
    [H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, 
     mid_state, option_ideal, option_noisy] = args

    arg_all = [tg, drive_amp_A, drive_amp_B, detune_A, detune_B,
     H_qbt_drive, w_0_2, w_1_2, num_cpus, c_op_list, logi_idx, 
     mid_state, option_ideal, option_noisy]

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
    [tg, drive_amp, detune, 
     H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx, 
     option_ideal, option_noisy] = arg_all

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
    fidelity = get_fidelity_super_operator(propagator, logi_idx, cz_gate(), c_op_list)
    # p0_kraus = qt.to_kraus(qt.to_super(propagator))
    # if len(c_op_list) != 0:
    #     p0_kraus = [truncate_2(i, logi_idx) for i in p0_kraus]
    # p0_kraus_zz = cz_phase_correct(p0_kraus)
    # p0_super_2 = qt.kraus_to_super(p0_kraus_zz)
    # fidelity = qt.metrics.average_gate_fidelity(p0_super_2, target=cz_gate())
    return np.log10(1-fidelity)

def cz_fidelity_log_noise(arg_optimize, *args):
    [tg, drive_amp, detune] = arg_optimize # Independent arguments that can be optimized over
    [H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx, option_ideal, option_noisy] = args # System arguments

    arg_all = [tg, drive_amp, detune, H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx, option_ideal, option_noisy]

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
    gamma_dephase_new = gamma_dephase_new * 50 / tphi
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
            else:
                jump_tphi.append(qt.Qobj(np.zeros((n_hspace, n_hspace))))    
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
    [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx, option_ideal, option_noisy] = args

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
    fidelity = get_fidelity_super_operator(propagator, logi_idx, qt.sigmax(), c_op_list)

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

def get_fidelity_super_operator(propagator, logi_idx, gate_target, c_op_list, mid_state=None):
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
        kraus = qt.to_kraus(propagator)
        kraus = [truncate_2(i, logi_idx) for i in kraus]    
        if gate_target == cz_gate():
            kraus = cz_phase_correct(kraus)    
        if gate_target == cnot():
            kraus = cnot_phase_correct(kraus, mid_state=mid_state)    
        super_op_post = qt.kraus_to_super(kraus)

    return qt.metrics.average_gate_fidelity(super_op_post, target=gate_target)

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
    print("\nTop 10 values:", top_values.tolist())
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

def print_data(label, data, num_each_row=4, num_digits=None, 
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




def shortest_path_to_core(G, core_states, target_state):
    """Finds the sortest path to the specified core states

    Args:
        G (nx.Graph): graph where nodes are states and edges
                      represent a population transfer rate under
                      the specified drive
        core_states (list[int, str]): list of core states, as they are
                                      labeled in the graph
        target (int or str): target state

    Returns:
       target, (shortest path length, shortest path)
    """
    shortest_path = ""
    shortest_path_len = np.inf
    for source in core_states[:-1]:
        if nx.has_path(G, source, target_state):
            path = nx.shortest_path(G, source=source, target=target_state,
                                    weight="weight")
            path_len = nx.path_weight(G, path,'weight')
            # path_len = nx.shortest_path_length(G, source=source, target=target, weight="weight")            
        if path_len < shortest_path_len:
            shortest_path_len = path_len
            shortest_path = ",".join([str(x) for x in path])
    return shortest_path_len, shortest_path


def all_path_to_core(G, core_states, target_state, cutoff=2):
    """Finds all paths to the specified core states
    under a specified length

    Args:
        G (nx.Graph): graph where nodes are states and edges
                      represent a population transfer rate under
                      the specified drive
        core_states (list[int, str]): list of core states, as they are
                                      labeled in the graph
        target (int or str): target state
        cutoff (float): maximum length to consider for paths

    Returns:
       target, (total length of paths, all paths)
    """
    path_tot = []
    if target_state in core_states:
        weight_tot = 1
    else:
        weight_tot = 0
        for source in core_states:
            for path in nx.all_simple_paths(G, source, target_state, cutoff=cutoff):
                weight_tot += np.exp( - nx.path_weight(G, path,'weight') )
                path_tot.append(path)
    return -np.log(weight_tot), path_tot


# def trunc_by_thresh(core_states_index, drive_term, thresh=1e-2, total_trunc=None):
#     """
#     Returns a list of state indices that are connected to the specified core states
#     via large entries in the given drive term

#     Args:
#         core_states_index (list[int]): The indices of the states to begin with
#                                    (should be the logical states).
#         drive_term (np.array complex): The operator used to drive a gate.
#         thresh (float, optional): The threshold above which states count as connected.
#                                   Defaults to 1e-2.
#         total_trunc (int, optional): Highest index to consider. If none is given
#                                      then considers all entries in drive_term

#     Returns:
#         list[int]: list of state indices
#     """

#     if total_trunc is None:
#         total_trunc = drive_term.shape[1]

#     hspace_index = [s for s in core_states_index]
#     # Add every state that is connected by entries above thresh
#     # to the core states in the drive term
#     # By adding to hspace_index as you're looping through it,
#     # we consider as many degrees of connection as we need
#     for s in hspace_index:
#         for i, s2 in enumerate(np.arange(total_trunc)):
#             if np.abs(drive_term[s, i]) > thresh and i not in hspace_index:
#                 hspace_index.append(i)
#     return sorted(hspace_index)


def pop_rate(A, n_ij, delta):
    """Calculates the maximum population that could transfer
    between two states if it were considered as a two state system

    Args:
        A (float): drive amplitude
        n_ij (complex): entry i,j of the operator used to drive
        delta (float): detuning from drive transition
    Returns:
        float: max population that could transfer (between 0-1)
    """
    return (abs(A*n_ij)**2) / (abs(A*n_ij)**2 + delta**2)
    # return (np.abs(A*n_ij)**2) / (np.abs(A*n_ij)**2 + delta**2)


def make_rate_graph(drive_term, evals, wd, A, labels = None, normalization=True):
    """
    Makes a graph that represents the population transfer rate
    of a system under the presence of the specified monotone drive

    Args:
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float): drive frequency
        A (float): drive amplitude
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given

    Returns:
        nx.Graph: Graph representing the system
    """

    if labels is None:
        labels = np.arange(drive_term.shape[0])

    G = nx.DiGraph()
    max_n_ij = np.max(np.abs(drive_term.data))
    for i, s_i in enumerate(labels):
        for j, s_j in enumerate(labels):
            if i < j:
                n_ij = np.abs(drive_term[i, j])
                if normalization:
                    n_ij *= n_ij/max_n_ij
                delta = abs(wd - (evals[j] - evals[i]))
                population_rate = pop_rate(A, n_ij, delta)
                if population_rate > 0:
                    G.add_edge(s_i, s_j, weight=-np.log(population_rate))
    return G


def make_leakage_df(core_states, drive_term, evals, wd, A, labels=None,
                    path_func=shortest_path_to_core, G=None):
    """
    Makes a dataframe where each row is a state rated by how much
    leakage is expected

    Args:
        core_states (list[int]): The indices of the states to begin with
                                   (should be the logical states).
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float or list of float): drive frequency(s)
        A (float or list of float): drive amplitude(s)
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given
        path_func (func): function that takes in (G, core_states, s) and returns
                          a distance from s to core_states. Either shortest_path_to_core
                          or all_path_to_core
        G (nx.Graph): pre-computed rate-graph. Makes one if none is given.

    Returns:
        dataframe
    """
    if labels is None:
        labels = np.arange(drive_term.shape[0])

    df = []
    if G is None:
        G = make_rate_graph(drive_term, evals, wd, A, labels = labels)

    for target_state in labels:
        shortest_path_len, shortest_path = path_func(G, core_states, target_state)
        entry = {}
        entry["i"] = target_state
        entry["path"] = shortest_path
        entry["path_len"] = np.exp(-shortest_path_len)
        df.append(entry)

    return pd.DataFrame(df).sort_values(by="path_len", ascending=False)


def trunc_by_graph_estimate(n, core_states, drive_term, evals, wd, A, labels=None,
                            path_func=shortest_path_to_core):
    """
    Returns indices/state labels for a truncated model, keeping the n most important states
    according to the graph search estimate.

    You can give multiple drive pulses by making wd and A lists. In this case it
    will combine the dataframes, keeping the maximum entry for each state.

    Args:
        n (int): number of states to include in the reduced model
        core_states (list[int]): The indices of the states to begin with
                                   (should be the logical states).
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float or list of float): drive frequency(s)
        A (float or list of float): drive amplitude(s)
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given
        path_func (func): function that takes in (G, core_states, s) and returns
                          a distance from s to core_states. Either shortest_path_to_core
                          or all_path_to_core
        G (nx.graph or list of nx.graph): precomputed rate graphs, must match len of A,wd

    Returns:
        list of state indices/labels
    """

    if isinstance(wd, float):
        wd = [wd]
    if isinstance(A, float):
        A = [A]

    df_list = []
    for i in range(len(wd)):
        df = make_leakage_df(core_states, drive_term, evals, wd[i], A[i], 
                             G=None, labels=labels, path_func=path_func)
        df_list.append(df)
    df = pd.concat(
                    df_list
                    ).sort_values(
                                    "path_len", ascending=False
                                    ).drop_duplicates("i", keep="first")
    return list(df["i"].values[:n])

def compare_two_lists(list1, list2):
    only_in_list1 = [item for item in list1 if item not in list2]
    only_in_list2 = [item for item in list2 if item not in list1]
    unique_elements = only_in_list1 + only_in_list2
    common_elements = [item for item in list1 if item in list2]

    print(f'list1 and list2 have {len(common_elements)} common elements'
          +f' and {len(list1) - len(common_elements)} unique elements.')

    print_data(f'Only in list1 (len={len(only_in_list1)})', only_in_list1, 
                  num_each_row=10, n_make_blank_line=50)
    list1_index = [list1.index(i) for i in only_in_list1]
    print_data(f'index Only in list1 (len={len(only_in_list1)})', list1_index, 
                  num_each_row=10, n_make_blank_line=50)
    
    print_data(f'Only in list2 (len={len(only_in_list2)})', only_in_list2, 
                  num_each_row=10, n_make_blank_line=50)    
    list2_index = [list2.index(i) for i in only_in_list2]
    print_data(f'index Only in list2 (len={len(only_in_list2)})', list2_index, 
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
    nsteps_ideal = 1/ max_step_ideal  # Set nsteps to a large number for parallel execution
    nsteps_noisy = 1/ max_step_noisy  # Set nsteps to a large number for parallel execution

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

def load_qubit_data_xgate(qubit_0 = True, folder = '../../data/3ncut_one_zeropi/'):
    """
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

def load_dephasing_data_xgate(drive_theta):
    """
    Load dephasing rates calculated for 50μs.

    Parameters:
        drive_theta (bool): If True, load theta dephasing; otherwise, phi.

    Returns:
        np.ndarray: Dephasing rates for each state.
    """
    gamma_file = 'data/data_gamma_theta_500.txt' if drive_theta else 'data/data_gamma_phi_500.txt'
    gamma_new = pd.read_csv(gamma_file)
    gamma_dephase = gamma_new['tphi_50us_02'].to_numpy()
    state_idx = gamma_new['hspace'].to_numpy()
    return state_idx, gamma_dephase

def load_drive_params_xgate(drive_theta):
    """
    Load X-gate drive parameters from a CSV file.

    Parameters:
        drive_theta (bool): If True, load theta-drive data; else load phi-drive data.

    Returns:
        np.ndarray: Parameters array. different rows mean different gate time. 
        columns mean 'tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
    """
    folder = 'data_xgate_theta_3ncut_mstep_3e4.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/' + folder)
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

def load_qubit_data_2q(truc1=300, truc_full = 2000, charge_pick=True):
    """
    Loads the energy spectrum and matrix elements (n_theta, n_phi) for the 0-π qubit.
    The function "generate_data()" in sigmaX_fidelity_import_paras.py can generate the data
    """    
     # If charge_pick = True, n_full=2000, else 1000
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2=2000_pick={charge_pick}/'
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

def load_drive_params_2q(cz_run):
    """
    Load two-qubit-gate drive parameters from a CSV file.
    """
    if cz_run:
        folder = 'data/data_cz_3ncut_truc1=300_select.txt'
        params = pd.read_csv(folder)[['tg', 'drive_amp', 'detune']].to_numpy()
    else:
        folder = '../cnot/data/data_cnot_fidelity_3ncut.txt'
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



truc_model = {}
truc_model['cz_all_path'] = [
'0-0', '5-0', '0-2', '2-0', '2-2', '5-2', '5-1', '0-1', '2-1', '1-0' ,
'0-5', '2-5', '1-2', '9-0', '4-0', '2-4', '5-5', '1-1', '5-4', '1-5' ,
'9-2', '0-4', '4-2', '2-8', '0-8', '9-1', '2-12', '2-9', '0-9', '0-12' ,
'12-0', '2-16', '0-16', '5-8', '1-8', '4-5', '2-13', '2-21', '0-18', '2-20' ,
'9-4', '4-9', '8-0', '2-18', '0-21', '2-24', '1-4', '18-0', '5-26', '0-13' ,

'13-0', '0-26', '15-0', '2-26', '1-12', '5-16', '8-1', '0-24', '15-1', '2-35' ,
'15-4', '2-33', '2-45', '0-33', '2-39', '0-45', '5-12', '0-39', '5-9', '8-12' ,
'5-33', '12-2', '4-4', '1-9', '4-1', '5-34', '2-30', '2-46', '1-16', '0-34' ,
'2-34', '8-2', '5-21', '2-52', '0-52', '0-42', '2-42', '2-59', '0-59', '5-18' ,
'0-65', '5-24', '2-55', '0-55', '0-20', '9-8', '8-9', '22-0', '1-25', '8-5' ,

'12-1', '4-8', '2-53', '2-36', '2-25', '5-20', '5-13', '9-24', '15-8', '1-30' ,
'0-25', '1-20', '5-25', '9-5', '1-13', '1-24', '13-2', '1-33', '18-1', '0-68' ,
'18-2', '20-0', '2-44', '9-12', '4-25', '9-9', '5-30', '0-83', '5-39', '9-16' ,
'0-36', '0-73', '12-4', '18-5', '22-2', '15-16', '1-21', '5-35', '9-13', '2-60' ,
'2-57', '15-2', '15-5', '2-28', '13-1', '4-12', '0-35', '9-20', '2-54', '1-18' ,

'0-81', '24-1', '4-39', '0-30', '1-26', '8-4', '4-16', '12-5', '1-35', '8-8' ,
'24-0', '12-9', '5-44', '0-77', '4-21', '0-57', '5-28', '1-39', '25-0', '8-18' ,
'1-28', '4-44', '33-0', '5-42', '12-12', '0-54', '13-12', '9-26', '4-24', '20-9' ,
'8-26', '24-4', '8-16', '0-46', '13-4', '1-34', '37-1', '0-44', '18-16', '22-4' ,
'1-45', '0-78', '9-25', '5-36', '24-9', '1-44', '4-35', '18-8', '0-53', '1-60' ,

'1-54', '8-21', '12-20', '4-34', '1-46', '9-28', '1-36', '0-64', '0-66', '24-2' ,
'22-1', '59-1', '37-0', '4-26', '0-76', '4-30', '1-53', '1-52', '4-13', '4-18' ,
'20-2', '4-20', '0-28', '1-42', '4-46', '4-33', '0-60', '12-24', '20-8', '1-65' ,
'0-82', '8-25', '8-20', '1-55', '1-59', '25-9', '13-16', '1-57', '34-0', '24-13' ,
'20-1', '26-8', '22-12', '1-64', '0-70', '13-25', '30-9', '46-0', '12-16', '9-30' ,

'4-53', '30-0', '33-1', '12-8', '18-12', '18-4', '12-18', '20-5', '4-36', '22-5' ,
'13-20', '12-13', '9-18', '46-2', '13-9', '9-21', '4-28', '35-2', '20-12', '8-13' ,
'28-4', '41-1', '4-42', '15-13', '25-2', '8-35', '15-12', '30-5', '39-1', '15-9' ,
'22-9', '13-18', '4-45', '8-34', '13-5', '8-24', '15-18', '12-26', '25-4', '44-0' ,
'58-0', '13-24', '45-1', '24-8', '35-0', '0-86', '60-0', '37-2', '13-8', '13-21' ,

'12-21', '35-5', '18-9', '13-13', '8-33', '20-4', '35-8', '22-8', '26-0', '15-24' ,
'26-2', '20-13', '12-25', '8-30', '44-1', '35-1', '25-1', '15-21', '34-4', '24-12' ,
'30-2', '54-1', '28-0', '41-0', '4-52', '28-2', '41-2', '54-2', '28-1', '45-0' ,
'37-4', '33-8', '34-2', '15-20', '26-1', '45-2', '46-1', '20-16', '25-12', '8-36' ,
'30-4', '39-2', '25-5', '54-0', '75-0', '33-2', '50-0', '8-28', '39-5', '64-0' ,

'22-13', '25-8', '66-0', '50-2', '34-1', '33-5', '51-0', '34-5', '35-4', '70-0' ,
'46-4', '18-13', '26-12', '22-16', '30-1', '59-0', '56-0', '28-5', '56-2', '45-4' ,
'24-5', '41-5', '65-0', '39-0', '50-1', '26-5', '24-16', '60-1', '26-9', '69-0' ,
'44-4', '58-1', '41-4', '26-4', '81-0', '30-8', '51-1', '39-4', '34-8', '33-4' ,
'44-2', '51-2', '28-8', '37-5', '74-0', '79-0', '37-8', '77-0', '56-1', '28-9' ,
]

truc_model['cz_short_path'] = [
'0-0', '5-0', '0-2', '2-0', '2-2', '5-2', '5-1', '0-1', '2-1', '1-0' ,
'0-5', '2-5', '1-2', '9-0', '4-0', '5-5', '2-4', '1-1', '5-4', '1-5' ,
'9-2', '0-4', '4-2', '4-5', '0-8', '2-8', '2-12', '9-1', '0-9', '2-9' ,
'4-1', '0-12', '0-16', '2-16', '15-4', '1-4', '8-0', '12-0', '2-18', '9-5' ,
'2-13', '5-8', '1-8', '9-4', '2-21', '1-12', '5-9', '0-18', '0-21', '4-9' ,

'1-9', '2-20', '18-0', '5-12', '25-0', '0-13', '2-24', '0-26', '5-26', '2-26' ,
'13-0', '5-16', '15-0', '0-24', '1-18', '8-5', '2-45', '2-39', '8-1', '0-45' ,
'0-39', '8-12', '12-2', '15-1', '2-35', '0-33', '2-33', '8-2', '0-34', '5-33' ,
'1-16', '4-4', '2-34', '5-21', '5-34', '0-52', '2-52', '2-30', '2-46', '15-8' ,
'4-12', '0-42', '2-42', '2-59', '0-59', '0-65', '0-55', '2-55', '5-18', '1-25' ,

'8-9', '22-0', '18-1', '5-13', '4-8', '0-20', '9-8', '12-1', '5-24', '1-30' ,
'5-20', '5-25', '2-53', '1-13', '9-24', '0-25', '13-2', '2-25', '1-20', '18-2' ,
'0-68', '2-36', '15-2', '20-0', '1-24', '4-16', '18-5', '9-12', '22-1', '1-33' ,
'9-9', '4-25', '5-39', '13-16', '5-30', '0-83', '24-1', '0-73', '0-36', '9-16' ,
'2-44', '12-9', '15-5', '9-13', '2-57', '46-0', '25-1', '5-35', '22-2', '1-21' ,

'12-8', '15-16', '12-4', '2-60', '13-1', '2-28', '0-35', '4-39', '4-13', '1-35' ,
'9-20', '0-30', '1-26', '0-81', '4-21', '0-57', '5-28', '5-44', '2-54', '12-5' ,
'8-4', '1-28', '33-0', '1-39', '0-77', '24-0', '5-42', '9-26', '24-4', '8-8' ,
'0-54', '33-1', '20-9', '41-2', '0-46', '8-18', '8-16', '0-78', '12-12', '1-44' ,
'13-12', '8-26', '1-54', '1-60', '4-44', '1-45', '0-53', '0-44', '13-4', '4-34' ,

'18-16', '37-1', '5-36', '22-4', '9-25', '4-24', '9-28', '1-34', '1-46', '24-9' ,
'4-35', '18-8', '24-2', '0-64', '12-20', '59-1', '8-21', '1-36', '4-26', '1-53' ,
'20-2', '4-18', '0-76', '1-52', '4-33', '0-66', '37-0', '8-20', '39-0', '8-25' ,
'54-0', '1-42', '1-57', '1-65', '24-13', '0-82', '4-30', '12-24', '9-21', '0-28' ,
'20-1', '1-64', '0-70', '13-5', '4-46', '4-20', '1-59', '1-55', '12-16', '18-4' ,

'0-60', '34-0', '12-18', '26-8', '20-8', '35-4', '25-9', '15-12', '22-12', '15-13' ,
'18-12', '4-53', '22-5', '9-18', '30-9', '44-0', '13-25', '35-2', '15-9', '9-30' ,
'4-36', '13-9', '20-5', '13-20', '8-13', '20-12', '25-2', '30-0', '12-13', '46-1' ,
'4-42', '46-2', '8-35', '13-18', '4-28', '28-4', '41-1', '4-45', '24-8', '30-5' ,
'8-24', '12-26', '39-1', '22-9', '13-8', '8-34', '13-21', '28-2', '15-18', '25-4' ,

'45-1', '22-8', '13-13', '44-1', '58-0', '13-24', '35-8', '18-9', '0-86', '60-0' ,
'12-21', '26-0', '15-24', '26-2', '35-5', '35-0', '37-2', '8-33', '41-0', '20-4' ,
'12-25', '20-13', '54-2', '30-2', '4-52', '8-30', '45-0', '34-4', '15-21', '28-0' ,
'34-2', '54-1', '28-1', '25-5', '45-2', '35-1', '24-12', '33-8', '15-20', '39-2' ,
'37-4', '8-36', '50-0', '26-1', '33-2', '30-4', '41-5', '50-2', '33-4', '20-16' ,

'22-13', '24-5', '75-0', '25-12', '34-5', '64-0', '39-5', '8-28', '25-8', '66-0' ,
'18-13', '41-4', '34-1', '33-5', '51-0', '22-16', '56-0', '59-0', '70-0', '28-5' ,
'56-2', '46-4', '65-0', '44-4', '26-12', '30-1', '45-4', '50-1', '26-5', '69-0' ,
'26-4', '24-16', '60-1', '26-9', '30-8', '58-1', '39-4', '79-0', '44-2', '51-1' ,
'51-2', '28-8', '81-0', '74-0', '34-8', '37-5', '77-0', '37-8', '56-1', '28-9' ,
]

truc_model['cz_all_500_detune1']  = [

'0-0', '5-0', '0-2', '2-0', '2-2', '5-1', '0-1', '2-1', '0-5', '2-5' ,
'1-0', '5-2', '1-2', '9-0', '5-4', '2-4', '0-4', '1-1', '0-9', '2-9' ,
'5-5', '4-0', '9-1', '1-5', '4-2', '0-12', '2-12', '2-8', '0-8', '9-2' ,
'2-16', '0-16', '0-13', '2-13', '2-21', '12-0', '5-8', '4-5', '2-18', '0-18' ,
'0-21', '1-16', '2-20', '0-26', '1-8', '18-0', '5-26', '2-24', '2-26', '0-24' ,

'15-0', '9-4', '1-12', '8-0', '0-20', '2-45', '0-45', '5-9', '2-39', '1-4' ,
'0-25', '0-39', '8-12', '2-25', '4-9', '2-33', '0-33', '5-12', '5-20', '2-35' ,
'0-34', '5-33', '2-34', '15-4', '5-16', '5-24', '0-36', '5-34', '0-52', '2-52' ,
'2-36', '13-0', '2-46', '9-8', '1-9', '15-1', '2-59', '0-59', '2-30', '2-65' ,
'0-65', '0-42', '2-42', '0-55', '2-55', '8-1', '1-25', '12-1', '5-35', '0-35' ,

'4-8', '5-21', '8-9', '4-4', '4-1', '0-57', '1-30', '1-21', '9-5', '18-1' ,
'2-53', '0-30', '2-57', '5-13', '1-39', '5-25', '5-45', '2-68', '9-24', '0-68' ,
'1-20', '1-26', '5-30', '9-16', '5-52', '1-18', '5-18', '8-2', '1-24', '1-13' ,
'12-2', '0-54', '0-83', '0-46', '0-73', '12-4', '2-54', '2-66', '18-2', '2-44' ,
'20-0', '9-34', '9-26', '0-44', '0-53', '5-53', '13-1', '12-5', '2-64', '15-8' ,

'9-33', '1-45', '0-64', '8-5', '5-39', '4-12', '5-46', '2-60', '9-12', '4-44' ,
'15-16', '0-76', '1-34', '5-36', '5-42', '0-66', '15-2', '0-81', '1-33', '4-39' ,
'18-5', '5-44', '4-25', '8-18', '9-9', '0-82', '4-16', '2-28', '8-39', '4-24' ,
'0-78', '12-12', '1-35', '0-77', '9-20', '0-60', '9-13', '22-0', '13-2', '0-70' ,
'4-18', '24-1', '1-52', '15-5', '2-70', '8-21', '4-21', '8-4', '1-42', '12-9' ,

'5-28', '1-65', '37-1', '15-25', '4-13', '22-4', '8-16', '0-28', '1-28', '0-89' ,
'1-55', '4-26', '4-30', '22-2', '5-54', '20-9', '1-59', '1-60', '9-36', '12-30' ,
'20-8', '33-0', '9-35', '1-54', '1-44', '13-12', '8-8', '8-26', '4-34', '12-18' ,
'24-0', '1-46', '20-1', '1-68', '0-93', '4-20', '24-4', '18-12', '9-25', '18-18' ,
'24-9', '4-35', '18-4', '13-4', '12-20', '18-8', '1-36', '9-21', '20-12', '22-5' ,

'1-53', '0-86', '18-16', '4-42', '25-0', '13-18', '9-28', '1-73', '8-35', '13-26' ,
'1-57', '4-33', '22-1', '30-0', '8-25', '59-1', '12-26', '20-2', '1-64', '15-9' ,
'12-33', '12-13', '30-5', '8-20', '0-92', '4-57', '9-18', '15-28', '15-12', '4-46' ,
'4-54', '1-70', '37-0', '12-8', '12-16', '13-9', '0-90', '4-28', '8-34', '13-8' ,
'46-0', '20-24', '24-2', '12-24', '26-8', '12-34', '20-5', '18-9', '24-13', '22-12' ,

'25-9', '1-77', '4-53', '33-1', '50-4', '30-9', '8-44', '45-1', '13-25', '4-36' ,
'9-30', '34-0', '37-4', '9-39', '13-20', '12-21', '35-2', '4-52', '56-4', '15-13' ,
'8-30', '20-4', '15-24', '12-35', '1-78', '18-20', '46-2', '4-45', '8-24', '13-5' ,
'8-13', '13-21', '13-16', '22-18', '41-1', '15-18', '25-16', '24-8', '18-24', '35-0' ,
'28-4', '22-8', '8-33', '39-2', '22-13', '15-21', '22-9', '25-1', '1-76', '15-20' ,

'39-1', '54-1', '15-26', '8-45', '20-18', '13-24', '20-21', '28-1', '12-36', '25-4' ,
'25-2', '58-0', '35-12', '44-0', '26-0', '60-0', '25-12', '8-42', '26-2', '12-25' ,
'1-66', '30-2', '20-20', '20-13', '25-8', '35-5', '37-2', '44-1', '46-1', '4-59' ,
'35-9', '35-8', '26-12', '24-12', '26-13', '25-5', '24-16', '13-13', '24-18', '8-36' ,
'13-30', '59-2', '4-55', '35-1', '4-60', '22-16', '45-4', '45-0', '20-16', '34-1' ,

'8-28', '33-9', '34-4', '28-2', '18-21', '39-5', '41-0', '45-2', '18-13', '25-13' ,
'33-12', '33-8', '54-2', '22-20', '44-5', '30-12', '41-2', '39-0', '30-1', '60-1' ,
'28-0', '34-8', '50-0', '34-12', '26-9', '28-13', '46-5', '41-5', '41-4', '34-2' ,
'33-2', '39-4', '26-1', '12-28', '30-4', '50-2', '26-16', '58-1', '34-5', '75-0' ,
'28-12', '54-0', '64-0', '30-8', '28-8', '35-4', '66-0', '59-0', '26-5', '65-2' ,

'33-5', '69-0', '44-4', '56-0', '70-0', '56-2', '46-4', '13-28', '81-0', '65-0' ,
'51-0', '28-5', '39-8', '24-5', '37-5', '84-0', '58-2', '37-9', '34-9', '39-9' ,
'33-4', '41-8', '50-1', '66-1', '45-5', '26-4', '44-2', '51-1', '54-4', '64-2' ,
'37-8', '64-1', '74-0', '86-0', '51-5', '75-1', '85-0', '50-5', '70-1', '56-1' ,
'51-2', '69-1', '28-9', '51-4', '28-16', '79-0', '60-2', '65-1', '74-1', '77-0' ,
]

truc_model['cz_short_300_detune1']  = [

'0-0', '2-0', '2-2', '5-0', '0-2', '5-1', '5-2', '0-1', '2-1', '0-5' ,
'2-5', '1-0', '1-2', '9-0', '5-4', '0-4', '2-4', '5-5', '1-1', '0-9' ,
'2-9', '9-1', '4-0', '9-2', '1-5', '4-2', '0-12', '9-4', '2-12', '0-8' ,
'2-8', '4-1', '4-5', '9-5', '5-8', '0-16', '2-16', '1-9', '1-12', '8-0' ,
'5-9', '1-4', '0-13', '0-18', '2-13', '2-18', '5-12', '15-4', '9-8', '2-21' ,

'12-0', '4-4', '4-9', '0-21', '8-5', '2-20', '0-26', '18-0', '1-8', '5-26' ,
'2-26', '2-24', '25-0', '0-24', '15-0', '8-2', '5-16', '0-20', '2-45', '0-45' ,
'15-1', '2-39', '0-39', '8-12', '5-20', '8-1', '0-25', '12-5', '2-25', '5-13' ,
'0-33', '2-33', '1-18', '2-35', '0-34', '1-16', '5-33', '9-9', '2-34', '5-24' ,
'0-36', '12-2', '4-12', '2-36', '1-21', '5-34', '0-52', '13-0', '2-46', '5-21' ,

'18-1', '0-59', '0-65', '8-9', '2-30', '0-30', '0-42', '2-42', '15-2', '0-55' ,
'15-8', '5-18', '8-4', '5-35', '1-25', '4-8', '0-35', '12-1', '0-57', '25-1' ,
'9-12', '12-4', '1-30', '18-2', '5-25', '9-24', '0-68', '1-20', '9-16', '5-30' ,
'8-16', '12-9', '1-13', '15-5', '0-54', '4-16', '1-26', '1-24', '0-46', '18-5' ,
'0-73', '22-1', '4-18', '9-13', '4-25', '20-0', '0-53', '0-44', '4-13', '2-44' ,

'13-1', '0-64', '24-1', '13-2', '9-20', '5-36', '1-33', '12-8', '20-2', '15-16' ,
'1-39', '18-4', '1-34', '8-18', '4-39', '0-66', '24-4', '22-2', '1-45', '22-4' ,
'13-16', '2-28', '1-35', '22-0', '0-70', '0-28', '4-21', '8-8', '25-4', '0-60' ,
'37-1', '12-12', '5-28', '9-21', '33-0', '4-26', '4-24', '46-0', '15-12', '1-28' ,
'8-21', '20-9', '8-26', '9-18', '8-25', '1-52', '39-0', '33-1', '13-5', '4-20' ,

'1-36', '4-30', '24-0', '1-54', '1-44', '20-1', '1-42', '12-18', '20-5', '15-9' ,
'22-5', '1-55', '20-8', '13-12', '4-34', '8-20', '9-25', '15-13', '12-16', '54-0' ,
'8-13', '18-12', '1-46', '13-4', '22-8', '24-9', '4-35', '18-8', '8-24', '13-9' ,
'24-2', '1-53', '4-36', '4-33', '41-2', '18-9', '30-5', '12-13', '24-8', '25-2' ,
'37-4', '30-0', '44-1', '45-1', '44-0', '28-1', '37-0', '25-8', '13-8', '30-2' ,

'41-0', '46-1', '25-5', '26-8', '35-0', '39-1', '39-2', '34-0', '4-28', '20-4' ,
'35-2', '28-2', '33-4', '28-4', '35-4', '41-1', '60-0', '22-9', '13-13', '24-5' ,
'51-0', '26-0', '58-0', '30-1', '28-0', '34-4', '26-2', '39-4', '37-2', '28-8' ,
'35-1', '45-0', '34-1', '34-2', '59-0', '45-2', '28-5', '50-0', '51-1', '33-2' ,
'30-4', '26-1', '64-0', '26-4', '26-5', '56-0', '33-5', '65-0', '50-1', '44-2' ,
]

truc_model['cz_short_200_detune1']  = [
'0-0', '2-0', '2-2', '5-0', '0-2', '5-1', '5-2', '0-1', '2-1', '0-5' ,
'2-5', '1-0', '1-2', '9-0', '5-4', '0-4', '2-4', '5-5', '1-1', '0-9' ,
'2-9', '9-1', '4-0', '9-2', '1-5', '4-2', '0-12', '9-4', '2-12', '0-8' ,
'2-8', '4-1', '4-5', '9-5', '5-8', '0-16', '2-16', '1-9', '1-12', '8-0' ,
'5-9', '1-4', '0-13', '0-18', '2-13', '2-18', '5-12', '15-4', '9-8', '2-21' ,

'12-0', '4-4', '4-9', '0-21', '8-5', '2-20', '0-26', '18-0', '1-8', '2-26' ,
'2-24', '25-0', '0-24', '15-0', '8-2', '5-16', '0-20', '0-45', '15-1', '0-39' ,
'8-12', '5-20', '8-1', '0-25', '12-5', '2-25', '5-13', '0-33', '2-33', '1-18' ,
'2-35', '0-34', '1-16', '9-9', '2-34', '5-24', '0-36', '12-2', '4-12', '2-36' ,
'1-21', '0-52', '13-0', '5-21', '18-1', '8-9', '2-30', '0-30', '0-42', '15-2' ,

'0-55', '15-8', '5-18', '8-4', '1-25', '4-8', '0-35', '12-1', '0-57', '25-1' ,
'9-12', '12-4', '1-30', '18-2', '1-20', '9-16', '8-16', '12-9', '1-13', '15-5' ,
'0-54', '4-16', '1-26', '1-24', '0-46', '18-5', '22-1', '4-18', '9-13', '4-25' ,
'20-0', '0-53', '0-44', '4-13', '13-1', '24-1', '13-2', '1-33', '12-8', '20-2' ,
'1-39', '18-4', '1-34', '8-18', '24-4', '22-2', '22-4', '2-28', '1-35', '22-0' ,

'0-28', '4-21', '8-8', '25-4', '37-1', '12-12', '33-0', '4-26', '4-24', '46-0' ,
'1-28', '39-0', '33-1', '13-5', '4-20', '1-36', '24-0', '20-1', '20-5', '15-9' ,
'22-5', '8-13', '13-4', '13-9', '24-2', '25-2', '30-0', '44-0', '28-1', '37-0' ,
'13-8', '30-2', '41-0', '35-0', '34-0', '20-4', '28-2', '24-5', '51-0', '26-0' ,
'30-1', '28-0', '26-2', '35-1', '45-0', '34-1', '50-0', '33-2', '26-1', '26-4' ,
]

truc_model['cz_short_500_detune1']  = [

'0-0', '2-2', '0-2', '2-0', '5-0', '0-1', '2-1', '0-5', '2-5', '1-0' ,
'5-2', '1-2', '5-1', '0-4', '2-4', '1-1', '0-9', '2-9', '4-0', '5-5' ,
'1-5', '4-2', '9-0', '0-12', '5-4', '2-12', '0-8', '2-8', '9-2', '4-1' ,
'1-9', '9-1', '0-16', '2-16', '1-4', '8-0', '4-5', '0-13', '2-13', '0-18' ,
'2-18', '1-12', '9-5', '9-4', '2-21', '5-9', '4-4', '0-21', '1-16', '0-26' ,

'5-8', '18-0', '1-8', '5-12', '15-4', '2-26', '8-2', '0-24', '2-24', '2-45' ,
'0-20', '0-45', '2-20', '2-39', '0-39', '8-12', '8-1', '0-25', '1-18', '2-25' ,
'12-0', '0-33', '2-33', '0-34', '4-9', '2-34', '5-16', '0-36', '0-52', '2-52' ,
'2-36', '2-59', '0-59', '2-65', '0-65', '15-0', '15-1', '0-42', '2-42', '0-30' ,
'9-8', '2-30', '0-55', '2-55', '12-2', '1-20', '1-21', '5-13', '8-5', '13-0' ,

'1-25', '8-4', '5-26', '9-9', '0-35', '5-24', '5-21', '0-57', '1-30', '8-9' ,
'2-35', '2-57', '1-39', '5-25', '2-68', '4-12', '18-1', '24-1', '0-68', '1-13' ,
'15-2', '1-26', '12-4', '15-8', '5-18', '12-1', '0-54', '18-2', '0-83', '4-8' ,
'1-24', '5-20', '0-73', '20-0', '0-46', '2-54', '5-30', '5-33', '12-5', '9-12' ,
'9-16', '25-0', '2-46', '0-44', '0-53', '5-34', '4-16', '8-16', '2-44', '2-53' ,

'0-64', '2-64', '1-45', '5-39', '12-9', '5-45', '15-5', '0-76', '4-44', '5-42' ,
'22-1', '1-34', '4-39', '4-25', '13-1', '18-5', '1-33', '0-81', '5-35', '0-82' ,
'0-66', '0-78', '8-18', '2-66', '1-35', '4-24', '0-28', '2-28', '2-60', '9-13' ,
'5-52', '13-2', '0-70', '33-0', '1-52', '0-77', '0-60', '2-70', '4-21', '12-12' ,
'5-28', '8-8', '9-24', '5-36', '4-13', '1-65', '8-21', '1-42', '1-28', '0-89' ,

'24-4', '1-36', '1-59', '4-18', '1-55', '22-2', '22-4', '18-4', '22-0', '1-60' ,
'4-30', '4-26', '5-54', '15-16', '20-9', '1-54', '1-44', '8-39', '20-8', '9-34' ,
'4-34', '5-53', '20-1', '9-35', '9-26', '37-1', '9-21', '1-46', '25-1', '1-68' ,
'0-93', '5-44', '46-0', '9-33', '4-20', '15-12', '12-8', '8-25', '5-46', '1-57' ,
'9-18', '13-4', '9-20', '9-25', '33-1', '1-53', '13-5', '13-12', '24-0', '18-16' ,

'1-73', '9-28', '0-86', '4-33', '8-20', '15-9', '8-26', '20-2', '12-16', '4-35' ,
'18-12', '59-1', '22-8', '1-64', '12-33', '30-0', '0-92', '1-70', '1-78', '22-5' ,
'15-25', '15-28', '15-13', '18-8', '13-16', '8-24', '12-13', '4-45', '24-2', '0-90' ,
'4-54', '4-28', '20-24', '13-9', '18-18', '12-34', '8-13', '9-39', '12-30', '24-13' ,
'12-20', '8-34', '1-66', '9-36', '1-77', '24-8', '12-24', '12-18', '15-26', '8-44' ,

'4-36', '20-5', '1-76', '18-9', '4-42', '20-12', '4-52', '25-2', '45-1', '9-30' ,
'35-2', '8-30', '44-0', '56-4', '37-4', '4-46', '25-9', '28-1', '25-4', '4-59' ,
'24-9', '30-9', '13-25', '25-5', '46-1', '4-57', '39-0', '20-13', '4-55', '20-4' ,
'44-1', '4-53', '8-35', '22-12', '12-21', '8-28', '30-5', '13-8', '8-33', '39-2' ,
'22-13', '13-21', '35-0', '13-26', '15-18', '15-21', '13-18', '15-20', '8-45', '20-18' ,

'12-26', '15-24', '28-2', '41-1', '26-0', '8-36', '26-2', '30-1', '35-12', '39-1' ,
'25-12', '33-4', '22-9', '18-21', '37-0', '39-5', '46-2', '54-1', '18-24', '35-4' ,
'35-8', '12-25', '30-2', '13-24', '13-13', '26-8', '24-12', '18-13', '25-8', '35-9' ,
'35-5', '50-4', '35-1', '20-20', '45-0', '24-18', '12-36', '24-5', '4-60', '20-16' ,
'13-30', '34-0', '25-16', '45-4', '37-2', '13-20', '41-2', '41-0', '45-2', '12-35' ,

'33-12', '22-16', '54-2', '20-21', '18-20', '25-13', '37-8', '22-18', '28-4', '8-42' ,
'50-0', '39-8', '26-13', '26-16', '33-2', '34-2', '60-0', '59-2', '50-2', '41-5' ,
'60-1', '26-1', '46-4', '28-0', '34-8', '58-0', '33-9', '12-28', '34-5', '59-0' ,
'54-0', '28-12', '65-2', '30-4', '24-16', '34-4', '56-0', '41-4', '56-2', '28-5' ,
'44-4', '22-20', '26-5', '65-0', '51-0', '33-8', '28-8', '26-12', '54-4', '58-2' ,

'26-4', '34-12', '81-0', '46-5', '34-1', '39-4', '69-0', '26-9', '30-12', '45-5' ,
'37-9', '41-8', '34-9', '28-9', '39-9', '33-5', '51-1', '30-8', '37-5', '44-5' ,
'13-28', '50-1', '28-13', '75-0', '44-2', '51-5', '58-1', '64-2', '74-0', '85-0' ,
'64-0', '50-5', '79-0', '66-0', '70-0', '56-1', '51-2', '84-0', '65-1', '51-4' ,
'60-2', '28-16', '66-1', '69-1', '77-0', '74-1', '86-0', '64-1', '75-1', '70-1' ,
]

truc_model['cz_short_500_detune0']  = [

'0-0', '2-0', '5-0', '2-2', '0-2', '5-1', '5-2', '0-1', '2-1', '0-5' ,
'2-5', '1-0', '1-2', '9-0', '5-4', '0-4', '2-4', '5-5', '1-1', '0-9' ,
'2-9', '9-1', '4-0', '9-2', '1-5', '4-2', '0-12', '9-4', '2-12', '0-8' ,
'2-8', '4-1', '4-5', '9-5', '5-8', '0-16', '2-16', '1-9', '1-12', '8-0' ,
'5-9', '1-4', '0-13', '0-18', '2-13', '2-18', '5-12', '15-4', '9-8', '2-21' ,

'12-0', '4-4', '4-9', '0-21', '8-5', '2-20', '0-26', '18-0', '1-8', '5-26' ,
'2-26', '2-24', '25-0', '0-24', '15-0', '8-2', '5-16', '0-20', '2-45', '0-45' ,
'15-1', '2-39', '0-39', '8-12', '5-20', '8-1', '0-25', '12-5', '2-25', '5-13' ,
'0-33', '2-33', '1-18', '2-35', '0-34', '1-16', '5-33', '9-9', '2-34', '5-24' ,
'0-36', '12-2', '4-12', '2-36', '1-21', '5-34', '0-52', '2-52', '13-0', '2-46' ,

'5-21', '18-1', '2-59', '0-59', '2-65', '0-65', '8-9', '2-30', '0-30', '0-42' ,
'2-42', '15-2', '0-55', '2-55', '15-8', '5-18', '8-4', '5-35', '1-25', '4-8' ,
'0-35', '12-1', '0-57', '25-1', '9-12', '12-4', '1-30', '2-57', '2-53', '18-2' ,
'5-25', '2-68', '5-45', '9-24', '0-68', '1-20', '9-16', '5-30', '8-16', '9-34' ,
'12-9', '5-52', '1-13', '15-5', '0-54', '5-39', '4-16', '0-83', '1-26', '1-24' ,

'0-46', '18-5', '0-73', '2-54', '22-1', '4-18', '2-66', '9-13', '4-25', '20-0' ,
'5-53', '9-26', '0-53', '0-44', '4-13', '2-44', '9-33', '13-1', '0-64', '5-46' ,
'24-1', '2-64', '13-2', '9-20', '5-36', '1-33', '12-8', '20-2', '2-60', '0-76' ,
'4-44', '5-42', '15-16', '1-39', '18-4', '1-34', '5-44', '8-18', '4-39', '0-81' ,
'0-66', '0-82', '24-4', '22-2', '1-45', '22-4', '8-39', '13-16', '2-28', '1-35' ,

'22-0', '0-70', '0-28', '4-21', '8-8', '25-4', '0-60', '2-70', '0-77', '37-1' ,
'12-12', '15-25', '5-28', '0-78', '9-21', '33-0', '4-26', '9-35', '4-24', '46-0' ,
'15-12', '1-28', '8-21', '0-89', '20-9', '8-26', '9-18', '8-25', '1-52', '39-0' ,
'33-1', '13-5', '5-54', '4-20', '1-36', '1-60', '9-36', '12-30', '4-30', '1-65' ,
'24-0', '9-30', '1-54', '1-44', '20-1', '1-42', '12-18', '20-5', '15-9', '18-16' ,

'22-5', '9-28', '1-59', '1-55', '20-8', '13-12', '4-34', '8-20', '9-25', '15-13' ,
'12-16', '54-0', '8-13', '18-12', '1-46', '13-4', '22-8', '0-93', '59-1', '24-9' ,
'18-18', '4-35', '20-12', '12-20', '18-8', '8-24', '13-9', '24-2', '1-57', '4-42' ,
'1-53', '8-35', '1-68', '4-36', '13-26', '13-18', '0-86', '9-39', '4-33', '8-34' ,
'41-2', '24-13', '12-26', '18-9', '30-5', '15-26', '12-24', '12-13', '24-8', '1-64' ,

'12-33', '25-2', '0-92', '37-4', '30-0', '4-57', '44-1', '45-1', '1-73', '44-0' ,
'28-1', '4-46', '1-70', '37-0', '25-8', '13-8', '15-28', '0-90', '30-2', '41-0' ,
'46-1', '25-9', '20-24', '13-21', '25-5', '26-8', '30-9', '50-4', '13-25', '12-34' ,
'4-54', '35-0', '39-1', '22-12', '4-45', '4-53', '39-2', '1-66', '8-30', '8-44' ,
'34-0', '4-28', '4-52', '13-20', '15-20', '12-21', '20-4', '1-76', '35-2', '1-77' ,

'56-4', '8-45', '28-2', '18-20', '15-24', '12-35', '33-4', '46-2', '8-33', '22-18' ,
'28-4', '1-78', '18-24', '35-4', '20-18', '13-30', '41-1', '60-0', '4-59', '22-9' ,
'54-1', '15-18', '18-21', '12-25', '15-21', '35-8', '8-36', '20-16', '25-12', '22-13' ,
'20-13', '4-55', '13-13', '4-60', '13-24', '20-21', '24-5', '12-36', '51-0', '26-0' ,
'58-0', '24-18', '35-5', '35-12', '30-1', '28-0', '34-4', '8-42', '26-2', '39-4' ,

'37-2', '20-20', '37-8', '28-8', '22-16', '26-13', '54-2', '45-4', '35-1', '18-13' ,
'24-12', '59-2', '24-16', '8-28', '41-5', '26-12', '46-4', '35-9', '34-12', '41-4' ,
'39-8', '12-28', '45-0', '34-1', '25-16', '60-1', '33-12', '39-5', '34-2', '26-16' ,
'59-0', '25-13', '33-9', '45-2', '44-4', '30-12', '33-8', '44-5', '22-20', '28-12' ,
'28-5', '50-0', '28-13', '34-8', '51-1', '26-9', '33-2', '46-5', '30-4', '81-0' ,

'50-2', '26-1', '58-1', '75-0', '54-4', '58-2', '30-8', '34-5', '64-0', '65-2' ,
'37-9', '26-4', '26-5', '66-0', '28-9', '56-0', '33-5', '70-0', '56-2', '69-0' ,
'13-28', '45-5', '65-0', '37-5', '79-0', '84-0', '39-9', '41-8', '34-9', '66-1' ,
'50-1', '44-2', '64-2', '65-1', '51-5', '69-1', '28-16', '51-2', '64-1', '86-0' ,
'51-4', '74-0', '75-1', '85-0', '50-5', '70-1', '56-1', '77-0', '74-1', '60-2' ,
]

truc_model['hand_pick'] = [
'0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
'5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '0-9', '2-4', '9-0' ,
'1-5', '5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '0-13', '0-16' ,
'8-1', '4-4', '13-0', '15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5' ,
'0-18', '2-9', '1-12', '0-20', '0-21', '18-0', '9-2', '12-1', '5-5', '0-24' ,
'22-0', '2-12', '13-1', '24-0', '0-25' , 
] 

truc_model['cnot_all_path'] = [
'0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
'5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
'13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
'9-0', '20-2', '20-5', '12-5', '4-0', '15-5', '5-1', '18-2', '12-0', '24-2' ,
'26-5', '1-5', '15-0', '13-0', '25-2', '35-2', '8-1', '33-2', '9-5', '4-5' ,

'12-8', '9-8', '1-8', '37-2', '5-12', '25-0', '9-4', '1-25', '22-5', '56-2' ,
'18-4', '13-9', '4-4', '2-4', '15-4', '50-2', '13-5', '9-9', '5-9', '4-1' ,
'20-0', '25-5', '15-9', '18-5', '9-1', '30-5', '2-21', '26-9', '1-13', '20-9' ,
'45-2', '18-0', '5-16', '22-0', '34-5', '15-1', '5-4', '0-21', '20-4', '26-0' ,
'25-4', '25-1', '0-4', '9-12', '24-5', '37-5', '24-0', '1-18', '12-9', '44-2' ,

'25-8', '2-12', '41-2', '4-16', '4-12', '4-21', '8-12', '18-9', '18-1', '45-0' ,
'24-9', '26-8', '39-2', '4-8', '8-4', '34-0', '22-8', '35-4', '12-4', '30-0' ,
'12-1', '28-2', '18-8', '0-8', '35-5', '35-0', '9-16', '37-0', '2-8', '1-9' ,
'13-8', '46-2', '2-26', '46-0', '2-9', '41-0', '4-13', '39-0', '2-25', '0-12' ,
'22-9', '33-0', '33-4', '22-4', '54-2', '8-18', '28-1', '0-45', '50-0', '24-1' ,

'28-5', '30-1', '15-8', '34-1', '22-1', '20-1', '13-16', '12-18', '0-9', '24-8' ,
'13-1', '51-2', '33-5', '2-30', '1-21', '4-18', '59-0', '15-12', '0-25', '8-24' ,
'8-8', '9-18', '44-0', '26-1', '65-0', '56-0', '41-1', '41-4', '5-30', '39-4' ,
'30-4', '58-0', '5-13', '18-12', '0-30', '45-4', '0-39', '13-4', '46-1', '28-0' ,
'2-18', '0-24', '0-13', '9-13', '30-8', '69-0', '41-5', '33-1', '37-8', '4-24' ,
]

truc_model['cnot_short_path'] = [
'8-2', '0-0', '0-2', '2-0', '2-2', '12-2', '1-2', '1-0', '5-2', '5-0' ,
'4-9', '8-5', '2-1', '22-2', '9-2', '0-1', '5-8', '2-5', '4-2', '5-5' ,
'34-2', '8-0', '1-1', '0-5', '13-2', '8-9', '15-2', '26-2', '1-4', '9-0' ,
'20-2', '4-0', '30-2', '5-1', '12-5', '20-5', '12-0', '18-2', '15-5', '15-0' ,
'1-5', '13-0', '35-2', '33-2', '8-1', '24-2', '26-5', '4-5', '25-2', '4-1' ,

'12-8', '9-5', '9-8', '1-8', '5-12', '25-0', '12-9', '22-5', '56-2', '18-0' ,
'37-2', '1-25', '2-4', '4-4', '50-2', '18-4', '9-4', '5-9', '0-21', '15-4' ,
'20-0', '25-5', '13-5', '9-1', '46-2', '13-9', '30-5', '15-9', '9-9', '2-21' ,
'35-5', '5-4', '1-13', '26-9', '33-5', '20-9', '18-5', '22-0', '54-2', '34-5' ,
'26-0', '45-2', '20-4', '5-16', '0-4', '37-5', '15-1', '13-1', '24-5', '24-0' ,

'25-1', '4-8', '51-2', '12-1', '25-8', '25-4', '9-12', '13-12', '22-9', '4-21' ,
'45-0', '4-16', '41-2', '18-9', '1-18', '18-8', '44-2', '8-12', '8-4', '2-12' ,
'22-8', '26-8', '4-12', '0-8', '30-0', '41-0', '18-1', '4-25', '15-8', '24-9' ,
'37-0', '39-2', '35-4', '2-8', '1-9', '5-21', '35-0', '34-0', '33-0', '12-12' ,
'12-4', '46-0', '13-8', '2-9', '20-8', '28-2', '30-4', '9-16', '5-18', '39-0' ,

'0-12', '2-26', '12-20', '28-1', '0-45', '33-4', '8-18', '50-0', '8-21', '4-30' ,
'24-1', '0-9', '22-1', '2-25', '4-13', '22-4', '24-8', '30-1', '13-16', '34-1' ,
'1-21', '0-18', '41-5', '12-18', '8-16', '20-1', '59-0', '15-12', '25-9', '28-5' ,
'4-18', '39-5', '44-0', '65-0', '12-24', '56-0', '44-4', '8-24', '9-18', '30-9' ,
'41-4', '58-0', '39-4', '2-30', '28-4', '5-30', '8-8', '5-13', '12-13', '0-25' ,
]

truc_model['cnot_hand_pick'] = [
'0-0', '0-2', '2-0', '8-2', '2-2', '12-2', '4-9', '1-2', '1-0', '5-2' ,
'5-0', '8-5', '22-2', '9-2', '2-1', '5-8', '5-5', '0-1', '34-2', '2-5' ,
'13-2', '4-2', '8-0', '1-1', '0-5', '8-9', '15-2', '26-2', '30-2', '1-4' ,
'9-0', '20-2', '20-5', '12-5', '4-0', '15-5', '5-1', '18-2', '12-0', '24-2' ,
'26-5', '1-5', '15-0', '13-0', '25-2', '35-2', '8-1', '33-2', '9-5', '4-5' ,

'12-8', '9-8', '1-8', '37-2', '5-12', '25-0', '9-4', '1-25', '22-5', '56-2' ,
'50-2', '13-5', '9-9', '5-9', '4-1', '20-0', '25-5', '15-9', '18-5', '9-1' ,
'25-4', '25-1', '0-4', '9-12', '24-5', '37-5', '24-0', '1-18', '12-9', '44-2' ,
'24-9', '26-8', '39-2', '4-8', '8-4', '34-0', '22-8', '35-4', '12-4', '30-0' ,
'22-9', '33-0', '33-4', '22-4', '54-2', '8-18', '28-1', '0-45', '50-0', '24-1' ,
'13-1', '51-2', '33-5', '2-30', '1-21', '4-18', '59-0', '15-12', '0-25', '8-24' ,    
]

truc_model['cnot_short_1000'] = [

'0-0', '0-2', '2-2', '2-0', '1-2', '1-0', '5-2', '5-0', '8-2', '2-1' ,
'0-1', '2-5', '8-0', '0-5', '1-4', '15-0', '15-2', '4-2', '1-1', '9-2' ,
'9-0', '4-5', '5-5', '12-2', '4-0', '9-5', '5-1', '20-2', '12-0', '20-0' ,
'13-2', '9-1', '4-9', '18-2', '1-5', '13-0', '1-13', '4-1', '18-0', '0-21' ,
'5-4', '8-5', '26-0', '26-2', '8-1', '1-8', '22-2', '8-9', '5-9', '12-5' ,

'45-0', '4-8', '25-2', '8-4', '5-8', '12-1', '2-4', '25-0', '45-2', '4-4' ,
'34-2', '0-8', '35-2', '35-0', '41-0', '13-5', '1-9', '2-8', '39-0', '35-5' ,
'33-0', '33-2', '41-2', '39-2', '34-5', '50-0', '50-2', '13-1', '0-4', '30-2' ,
'22-0', '20-5', '9-8', '35-4', '59-0', '59-2', '24-2', '56-0', '56-2', '65-0' ,
'24-0', '15-5', '18-9', '28-1', '2-21', '65-2', '46-0', '28-5', '34-0', '26-5' ,

'25-4', '15-1', '69-2', '37-2', '34-1', '2-9', '30-0', '33-12', '69-0', '0-12' ,
'44-0', '37-0', '1-25', '0-18', '25-1', '51-5', '0-9', '85-0', '85-2', '1-20' ,
'9-4', '33-5', '18-5', '44-2', '12-9', '51-2', '2-12', '74-0', '74-2', '8-8' ,
'25-5', '24-1', '18-20', '4-13', '41-4', '9-9', '30-1', '26-1', '5-12', '20-1' ,
'33-4', '13-9', '22-1', '81-2', '58-2', '81-0', '51-1', '22-5', '0-13', '58-0' ,

'51-0', '18-1', '30-5', '46-5', '81-5', '66-2', '66-0', '0-16', '79-2', '12-8' ,
'1-21', '99-0', '99-2', '12-4', '33-1', '39-4', '46-2', '1-12', '2-16', '46-1' ,
'24-5', '56-4', '30-4', '5-16', '60-0', '79-0', '2-13', '60-2', '35-1', '77-0' ,
'77-2', '37-1', '9-12', '54-2', '18-4', '24-4', '37-5', '1-18', '102-0', '28-0' ,
'28-2', '2-30', '64-2', '15-4', '5-13', '102-2', '98-0', '86-2', '98-2', '86-0' ,

'46-4', '89-2', '45-5', '0-39', '54-0', '44-5', '13-4', '4-12', '30-8', '18-12' ,
'12-18', '60-1', '109-0', '70-2', '128-0', '4-16', '111-0', '89-0', '24-9', '64-0' ,
'15-9', '15-12', '8-44', '26-8', '35-9', '22-9', '127-0', '75-2', '18-8', '9-18' ,
'15-18', '34-4', '8-16', '50-5', '84-0', '22-4', '75-0', '26-13', '39-5', '28-4' ,
'1-16', '75-1', '8-21', '20-4', '39-9', '45-1', '34-9', '105-0', '12-24', '8-24' ,

'56-5', '20-9', '26-9', '5-25', '41-1', '25-9', '84-2', '12-20', '92-0', '13-12' ,
'15-8', '22-12', '60-5', '90-0', '2-25', '8-18', '74-5', '44-1', '70-0', '25-8' ,
'20-8', '25-12', '89-4', '79-1', '33-9', '93-2', '12-13', '54-4', '93-0', '0-24' ,
'81-1', '123-0', '92-2', '8-12', '64-5', '39-1', '56-1', '4-30', '4-21', '5-21' ,
'5-18', '65-5', '39-16', '115-0', '0-25', '30-9', '124-0', '90-2', '9-16', '12-12' ,

'4-25', '101-2', '70-5', '101-0', '90-1', '69-1', '37-4', '25-16', '50-1', '28-9' ,
'54-5', '44-4', '64-1', '4-20', '5-36', '13-8', '41-5', '8-13', '12-34', '121-0' ,
'116-0', '66-5', '110-0', '18-18', '2-18', '0-30', '26-4', '2-26', '86-1', '54-13' ,
'22-8', '84-5', '79-5', '9-13', '65-1', '58-5', '24-12', '0-36', '22-18', '39-8' ,
'59-5', '84-1', '75-5', '18-42', '20-24', '26-12', '74-1', '54-1', '9-25', '33-8' ,

'4-24', '69-5', '70-4', '51-9', '15-21', '99-1', '13-25', '34-12', '51-4', '81-4' ,
'106-0', '64-4', '100-2', '13-13', '100-0', '25-18', '35-8', '24-13', '69-4', '50-9' ,
'1-78', '20-12', '85-1', '66-1', '13-16', '9-28', '41-8', '59-1', '15-13', '60-8' ,
'4-18', '35-12', '5-35', '9-20', '1-42', '25-21', '0-20', '46-9', '30-16', '4-60' ,
'64-9', '22-13', '18-16', '77-5', '12-25', '12-21', '79-4', '24-16', '59-4', '45-4' ,

'60-9', '12-16', '35-18', '12-44', '70-1', '37-9', '24-8', '13-30', '93-1', '5-24' ,
'44-12', '50-12', '20-18', '60-4', '50-8', '54-12', '44-9', '58-1', '69-8', '86-4' ,
'25-35', '15-16', '58-4', '8-25', '0-45', '44-13', '30-12', '28-8', '8-59', '5-30' ,
'15-33', '0-33', '64-8', '77-1', '0-26', '34-8', '75-4', '85-5', '20-16', '22-20' ,
'1-33', '4-33', '33-20', '89-1', '15-34', '22-39', '54-9', '34-13', '37-12', '2-34' ,

'22-16', '46-8', '20-21', '22-36', '25-30', '22-24', '46-13', '33-18', '9-21', '75-8' ,
'106-1', '50-4', '58-9', '20-33', '90-4', '66-4', '56-8', '44-16', '25-13', '13-18' ,
'13-24', '18-21', '2-39', '45-8', '39-13', '2-35', '45-9', '50-18', '37-16', '28-24' ,
'41-16', '24-36', '37-8', '22-21', '1-59', '66-9', '8-52', '59-8', '26-18', '26-33' ,
'22-28', '12-35', '30-20', '8-39', '129-0', '15-25', '98-1', '1-24', '0-35', '41-9' ,

'74-4', '15-20', '1-30', '39-12', '74-8', '102-1', '28-30', '2-33', '1-39', '4-44' ,
'22-25', '101-1', '2-36', '8-34', '2-45', '51-12', '18-39', '26-16', '30-35', '24-30' ,
'65-9', '2-68', '20-13', '24-35', '5-46', '15-42', '59-12', '64-12', '18-25', '56-9' ,
'25-25', '30-24', '37-24', '41-26', '35-16', '4-34', '59-9', '18-34', '2-44', '84-4' ,
'65-8', '34-25', '1-34', '12-39', '1-28', '77-4', '26-34', '15-59', '30-13', '18-13' ,

'22-30', '41-18', '26-21', '22-44', '22-35', '2-54', '56-13', '22-34', '28-12', '65-4' ,
'2-24', '13-44', '24-28', '15-45', '37-20', '2-52', '24-21', '18-33', '100-1', '92-1' ,
'20-45', '22-26', '51-13', '28-25', '5-45', '37-13', '24-20', '20-39', '25-36', '13-36' ,
'8-42', '109-1', '25-24', '24-24', '20-34', '51-8', '58-8', '41-24', '0-34', '0-60' ,
'5-34', '46-12', '12-30', '70-8', '18-45', '39-18', '24-25', '1-98', '45-16', '20-26' ,

'15-39', '2-20', '13-35', '44-18', '35-21', '34-20', '34-16', '105-1', '28-18', '0-42' ,
'8-66', '0-52', '77-8', '20-20', '1-54', '8-73', '9-24', '33-16', '54-8', '33-24' ,
'35-26', '85-4', '9-46', '41-12', '13-33', '9-44', '9-54', '15-26', '8-30', '28-26' ,
'79-8', '44-8', '8-20', '4-53', '45-12', '35-25', '18-52', '58-13', '1-26', '1-44' ,
'45-13', '5-57', '20-28', '9-30', '20-30', '30-18', '0-59', '12-36', '13-34', '56-12' ,

'1-46', '4-36', '5-20', '28-13', '4-45', '45-18', '39-21', '66-8', '54-16', '50-16' ,
'2-28', '9-53', '25-20', '26-20', '12-33', '20-42', '13-21', '34-18', '28-21', '15-30' ,
'5-33', '4-46', '34-28', '30-30', '2-65', '41-13', '2-59', '2-55', '18-24', '25-33' ,
'9-34', '4-42', '9-36', '12-26', '5-54', '13-20', '0-28', '18-26', '0-65', '15-55' ,
'5-39', '0-55', '12-28', '34-30', '30-21', '15-24', '5-26', '0-53', '1-36', '34-21' ,

'9-39', '58-12', '9-42', '51-16', '8-45', '4-78', '8-57', '30-25', '0-54', '44-21' ,
'9-26', '0-44', '56-16', '34-24', '24-18', '35-13', '25-26', '33-13', '45-21', '0-78' ,
'2-53', '8-33', '2-42', '12-64', '15-28', '18-28', '46-16', '35-24', '18-35', '9-33' ,
'37-18', '2-78', '8-36', '15-36', '15-46', '4-35', '37-25', '20-25', '28-16', '51-18' ,
'15-52', '22-42', '60-12', '0-93', '4-26', '9-35', '5-59', '5-73', '33-26', '0-46' ,

'4-76', '12-53', '26-24', '35-20', '1-45', '5-42', '18-30', '37-21', '30-26', '12-46' ,
'39-20', '8-81', '13-26', '5-81', '46-18', '41-21', '8-55', '24-34', '25-39', '26-39' ,
'33-25', '12-59', '26-26', '44-20', '0-114', '41-20', '1-35', '28-34', '8-54', '28-20' ,
'9-45', '13-55', '0-68', '15-44', '1-60', '15-35', '18-44', '8-60', '8-26', '41-25' ,
'4-77', '18-36', '4-86', '5-65', '50-13', '9-52', '24-26', '2-73', '8-46', '20-36' ,

'24-39', '4-70', '12-45', '37-26', '13-46', '34-26', '20-44', '2-60', '4-39', '26-35' ,
'4-54', '0-89', '4-59', '9-70', '0-57', '0-73', '39-25', '25-34', '8-35', '4-28' ,
'13-59', '0-83', '22-33', '24-44', '26-30', '8-65', '12-52', '24-42', '12-54', '20-46' ,
'26-25', '20-35', '8-78', '4-64', '33-30', '15-54', '5-28', '35-28', '24-33', '35-30' ,
'8-28', '0-64', '8-64', '13-57', '9-66', '39-24', '5-44', '5-52', '1-53', '28-33' ,

'30-34', '13-42', '18-46', '9-60', '22-45', '1-68', '12-57', '13-45', '12-42', '8-68' ,
'5-68', '39-26', '28-35', '9-57', '13-39', '26-36', '13-53', '9-59', '2-57', '8-53' ,
'1-93', '15-53', '0-81', '18-53', '5-55', '1-81', '30-28', '2-86', '13-60', '5-60' ,
'25-28', '0-76', '2-83', '1-105', '1-57', '0-86', '13-52', '8-76', '12-76', '15-57' ,
'4-52', '1-65', '4-57', '5-53', '2-77', '26-28', '12-65', '33-28', '4-68', '0-98' ,

'4-55', '2-98', '12-55', '5-82', '9-65', '0-77', '9-64', '1-52', '1-89', '2-81' ,
'9-55', '9-68', '1-82', '8-77', '28-28', '1-83', '2-46', '1-55', '1-102', '2-89' ,
'0-82', '5-64', '4-93', '8-70', '2-100', '4-89', '2-82', '4-65', '2-66', '12-60' ,
'4-82', '4-66', '5-77', '1-114', '4-90', '5-78', '0-97', '1-109', '1-92', '1-100' ,
'0-112', '2-92', '4-92', '2-105', '0-109', '2-93', '0-125', '2-109', '0-70', '2-97' ,

'0-92', '0-102', '2-102', '13-28', '2-64', '28-36', '2-76', '5-70', '0-135', '13-54' ,
'4-73', '0-105', '1-73', '1-70', '0-66', '4-81', '5-66', '2-70', '1-86', '1-77' ,
'1-66', '4-83', '1-76', '0-124', '5-76', '5-83', '0-110', '1-64', '0-100', '1-97' ,
'5-86', '0-90', '0-103', '0-116', '2-103', '1-106', '0-123', '1-90', '0-121', '1-112' ,
'1-103', '2-106', '0-131', '0-106', '1-110', '2-90', '0-129', '0-132', '0-136', '0-128' ,
]

truc_model['cnot_short_500'] = [

'0-0', '2-0', '2-2', '0-2', '1-2', '1-0', '5-2', '5-0', '8-2', '2-1' ,
'0-1', '2-5', '8-0', '0-5', '1-4', '15-0', '15-2', '4-2', '1-1', '9-2' ,
'9-0', '5-5', '4-5', '12-2', '4-0', '5-1', '9-5', '20-2', '12-0', '13-2' ,
'4-9', '20-0', '1-5', '9-1', '13-0', '18-2', '1-13', '4-1', '18-0', '8-5' ,
'0-21', '5-4', '8-1', '26-0', '26-2', '1-8', '22-2', '8-9', '5-9', '12-5' ,

'25-2', '5-8', '45-0', '4-8', '2-4', '25-0', '8-4', '4-4', '12-1', '45-2' ,
'34-2', '0-8', '35-2', '35-0', '35-5', '41-0', '13-5', '1-9', '2-8', '39-0' ,
'33-0', '33-2', '41-2', '39-2', '34-5', '50-0', '50-2', '13-1', '0-4', '30-2' ,
'22-0', '20-5', '9-8', '24-2', '35-4', '59-0', '59-2', '24-0', '56-0', '56-2' ,
'65-0', '15-5', '46-0', '18-9', '28-1', '2-21', '65-2', '34-0', '26-5', '25-4' ,

'15-1', '28-5', '37-2', '2-9', '30-0', '0-12', '44-0', '37-0', '34-1', '33-12' ,
'69-0', '1-25', '0-18', '0-9', '25-1', '9-4', '51-5', '33-5', '18-5', '44-2' ,
'85-0', '12-9', '51-2', '1-20', '2-12', '8-8', '4-13', '74-0', '9-9', '25-5' ,
'26-1', '24-1', '18-20', '41-4', '20-1', '30-1', '5-12', '13-9', '33-4', '58-2' ,
'22-1', '81-0', '0-13', '58-0', '51-1', '51-0', '18-1', '12-8', '22-5', '66-0' ,

'30-5', '46-5', '1-21', '12-4', '33-1', '0-16', '46-2', '39-4', '30-4', '5-16' ,
'60-0', '1-12', '79-0', '2-13', '2-16', '60-2', '46-1', '24-5', '56-4', '35-1' ,
'18-4', '9-12', '77-0', '54-2', '37-1', '24-4', '15-4', '1-18', '28-0', '28-2' ,
'2-30', '64-2', '37-5', '86-0', '46-4', '45-5', '54-0', '13-4', '4-12', '5-13' ,
'0-39', '15-9', '44-5', '30-8', '18-12', '12-18', '60-1', '24-9', '64-0', '4-16' ,

'26-8', '15-12', '22-9', '8-44', '35-9', '18-8', '15-18', '34-4', '8-16', '20-4' ,
'50-5', '84-0', '22-4', '75-0', '39-5', '9-18', '28-4', '1-16', '8-21', '45-1' ,
'34-9', '20-9', '26-9', '26-13', '12-24', '75-1', '39-9', '41-1', '25-9', '12-20' ,
'8-24', '13-12', '5-25', '15-8', '22-12', '2-25', '8-18', '70-0', '20-8', '8-12' ,
'25-12', '44-1', '4-21', '25-8', '12-13', '54-4', '0-24', '33-9', '39-1', '56-1' ,

'4-30', '5-21', '5-18', '0-25', '9-16', '30-9', '12-12', '4-25', '69-1', '37-4' ,
'50-1', '28-9', '13-8', '44-4', '4-20', '25-16', '2-18', '41-5', '8-13', '26-4' ,
'64-1', '2-26', '5-36', '22-8', '18-18', '0-30', '12-34', '9-13', '65-1', '24-12' ,
'0-36', '39-8', '22-18', '74-1', '9-25', '33-8', '4-24', '20-24', '15-21', '13-25' ,
'26-12', '34-12', '51-4', '54-1', '13-13', '20-12', '35-8', '24-13', '13-16', '4-18' ,

'9-28', '41-8', '59-1', '9-20', '12-21', '1-42', '1-78', '66-1', '35-12', '0-20' ,
'24-16', '45-4', '4-60', '15-13', '22-13', '18-16', '5-35', '12-25', '24-8', '70-1' ,
'58-1', '37-9', '13-30', '20-18', '12-16', '0-45', '5-24', '28-8', '15-16', '34-8' ,
'30-12', '5-30', '8-25', '20-16', '22-16', '0-33', '22-20', '1-33', '0-26', '2-34' ,
'20-21', '4-33', '13-18', '9-21', '50-4', '25-13', '13-24', '37-8', '18-21', '2-35' ,

'15-25', '2-39', '1-59', '15-20', '1-30', '12-35', '8-39', '1-24', '0-35', '20-13' ,
'5-46', '1-39', '4-44', '2-36', '8-34', '26-16', '2-33', '2-45', '1-28', '2-68' ,
'2-44', '1-34', '4-34', '2-54', '18-13', '28-12', '2-24', '5-45', '2-52', '8-42' ,
'0-60', '5-34', '12-30', '20-20', '2-20', '0-34', '8-30', '9-24', '8-20', '0-42' ,
'0-52', '15-26', '1-54', '4-53', '5-20', '4-45', '1-26', '9-30', '12-33', '12-36' ,

'4-36', '28-13', '1-44', '2-28', '0-59', '5-26', '18-24', '5-33', '1-46', '9-34' ,
'12-26', '4-46', '13-21', '15-24', '9-36', '13-20', '0-28', '2-65', '1-36', '9-39' ,
'5-39', '2-59', '12-28', '2-55', '0-53', '4-42', '5-54', '0-65', '0-55', '8-45' ,
'0-54', '0-44', '15-28', '9-33', '0-78', '8-36', '2-53', '8-33', '9-26', '24-18' ,
'2-42', '28-16', '4-35', '0-93', '1-45', '9-35', '0-46', '4-26', '13-26', '5-42' ,

'1-35', '8-26', '0-68', '1-60', '4-28', '4-59', '2-60', '4-54', '0-57', '4-39' ,
'8-35', '0-89', '0-73', '0-83', '5-28', '8-28', '5-44', '1-53', '0-64', '5-52' ,
'1-68', '2-57', '0-81', '0-76', '0-86', '1-65', '1-57', '5-53', '4-57', '4-52' ,
'1-52', '4-55', '0-77', '2-46', '1-55', '0-82', '2-66', '0-70', '0-92', '13-28' ,
'2-64', '1-73', '0-66', '2-70', '1-70', '1-76', '1-77', '1-66', '1-64', '0-90' ,
]

truc_model['cnot_82'] = [
'0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
'5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '2-4', '9-0', '1-5' ,
'5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '8-1', '4-4', '13-0' ,
'15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5', '0-18', '2-9', '0-21' ,
'18-0', '9-2', '12-1', '5-5', '0-24', '4-8', '1-13', '20-0', '8-4', '22-0' ,

'2-12', '13-1', '24-0', '0-25', '15-1', '5-8', '4-9', '12-2', '9-4', '8-5' ,
'25-0', '13-2', '26-0', '1-18', '15-2', '5-9', '4-12', '0-30', '18-1', '12-4' ,
'9-5', '1-21', '8-8', '20-1', '4-13', '22-1', '4-16', '30-0', '13-4', '5-12' ,
'15-4', '24-1', '33-0', '18-2', '34-0', '2-21', '8-9', '35-0', '1-25', '9-8' ,
'12-5', '37-0', '20-2', '5-16', '22-2', '13-5', '25-1', '26-1', '24-2', '15-5' ,

'2-25', '39-0', '8-12', '2-26', '9-9', '18-4', '28-1', '12-8', '41-0', '0-45' ,
'4-21', '13-8', '25-2', '20-4', '22-4', '4-24', '26-2', '5-18', '15-8', '30-1' ,
'9-12', '44-0', '12-9', '28-2', '45-0', '33-1', '18-5', '2-30', '46-0', '34-1' ,
'13-9', '9-13', '9-16', '20-5', '22-5', '50-0', '30-2', '25-4', '15-9', '8-18' ,
'24-5', '12-12', '33-2', '18-8', '34-2', '35-2', '13-12', '41-1', '37-2', '22-8' ,

'56-0', '25-5', '26-5', '30-4', '18-9', '39-2', '13-16', '59-0', '33-4', '28-5' ,
'41-2', '35-4', '20-9', '22-9', '25-8', '65-0', '26-8', '24-9', '30-5', '12-18' ,
'44-2', '33-5', '45-2', '34-5', '35-5', '46-2', '12-20', '69-0', '37-5', '26-9' ,
'50-2', '51-2', '39-5', '54-2', '41-5', '56-2', '58-2', '34-9', '18-20', '59-2' ,
'44-5', '45-5', '60-2', '41-8', '46-5', '64-2', '65-2', '33-12', '50-5', '51-5' ,
]

truc_model['cnot_80'] = [
'0-0', '0-2', '2-0', '2-2', '8-0', '8-2', '12-0', '8-1', '1-2', '1-0' ,
'1-8', '5-2', '5-0', '1-4', '22-0', '20-1', '8-8', '30-0', '34-0', '2-1' ,
'4-4', '13-0', '9-0', '0-1', '2-5', '9-5', '0-5', '5-1', '4-5', '15-1' ,
'15-0', '15-2', '9-1', '26-1', '18-1', '24-0', '25-0', '13-4', '4-1', '12-4' ,
'2-4', '5-4', '9-4', '1-21', '37-0', '5-8', '0-13', '20-2', '24-4', '18-2' ,

'20-0', '18-0', '5-9', '41-2', '0-21', '1-13', '4-13', '4-12', '1-9', '22-4' ,
'4-9', '81-0', '8-16', '0-8', '2-8', '15-9', '20-5', '8-9', '0-36', '26-0' ,
'35-4', '12-2', '28-0', '8-5', '12-5', '4-8', '1-20', '12-1', '5-21', '60-0' ,
'8-4', '45-1', '0-35', '26-2', '9-12', '15-8', '4-2', '64-0', '44-0', '37-1' ,
'13-2', '50-1', '2-13', '13-5', '0-20', '1-1', '70-0', '45-0', '35-0', '22-2' ,

'30-4', '35-2', '4-24', '45-2', '9-2', '13-1', '1-5', '30-2', '0-18', '9-8' ,
'5-5', '34-2', '26-4', '75-0', '41-0', '25-2', '39-0', '4-0', '25-4', '0-9' ,
'58-0', '13-9', '33-0', '5-18', '24-2', '24-5', '2-21', '18-5', '39-2', '20-4' ,
'35-1', '35-5', '33-2', '51-0', '50-0', '46-0', '37-4', '0-30', '34-5', '50-2' ,
'56-0', '4-21', '28-4', '46-4', '34-4', '54-0', '44-4', '12-9', '1-25', '59-0' ,

'26-8', '41-5', '41-1', '33-8', '39-1', '56-2', '28-1', '26-5', '22-1', '65-0' ,
'33-1', '18-9', '25-1', '34-1', '12-8', '66-0', '5-12', '15-5', '44-1', '24-9' ,
'28-5', '15-18', '30-1', '59-1', '13-8', '9-28', '60-1', '56-1', '2-12', '58-1' ,
'24-1', '41-4', '0-24', '15-16', '74-0', '1-12', '1-26', '69-0', '0-16', '2-44' ,
'1-33', '34-8', '46-1', '1-16', '18-4', '9-9', '25-5', '2-16', '15-4', '18-8' ,
]

truc_model['cnot_14'] = [
'0-0', '0-2', '2-0', '1-4', '2-2', '12-0', '8-1', '1-2', '1-0', '5-2' ,
'5-0', '4-4', '30-0', '22-0', '2-1', '9-0', '0-1', '34-0', '2-5', '26-0' ,
'4-2', '1-1', '0-5', '1-8', '8-2', '15-0', '5-1', '4-8', '13-0', '9-2' ,
'12-1', '5-5', '20-0', '8-4', '4-0', '8-0', '13-2', '35-0', '1-13', '33-0' ,
'12-2', '18-0', '1-5', '24-0', '8-8', '15-2', '25-0', '2-4', '0-13', '25-2' ,

'4-5', '0-21', '4-9', '45-0', '18-1', '13-4', '22-1', '50-0', '5-4', '24-4' ,
'56-0', '8-5', '9-1', '37-0', '24-1', '20-2', '9-4', '34-1', '74-0', '12-4' ,
'4-1', '13-1', '18-2', '9-5', '5-8', '81-0', '22-4', '41-0', '22-2', '30-1' ,
'59-0', '26-4', '1-21', '65-0', '8-16', '39-0', '25-1', '15-4', '26-2', '34-4' ,
'24-2', '69-0', '0-4', '4-12', '20-4', '26-1', '44-0', '64-0', '1-12', '8-9' ,

'51-1', '12-8', '77-0', '5-12', '5-9', '12-5', '34-2', '13-8', '70-0', '45-1' ,
'30-2', '9-12', '41-4', '45-2', '15-1', '15-8', '37-2', '15-5', '35-2', '2-8' ,
'0-8', '60-1', '18-4', '28-1', '30-4', '1-9', '4-13', '37-1', '2-9', '28-0' ,
'13-5', '75-0', '33-2', '0-12', '41-2', '58-0', '33-4', '39-2', '37-4', '18-5' ,
'4-24', '8-13', '34-5', '50-2', '33-5', '20-1', '46-0', '0-9', '9-8', '12-9' ,

'22-8', '46-4', '33-1', '44-4', '20-5', '1-16', '8-12', '4-21', '24-8', '58-1' ,
'2-12', '41-5', '35-4', '34-8', '56-1', '5-18', '56-2', '35-5', '9-9', '2-13' ,
'26-5', '51-0', '33-8', '0-20', '54-0', '51-2', '46-2', '4-16', '35-1', '18-9' ,
'28-4', '15-9', '25-4', '26-8', '2-21', '59-1', '46-1', '9-28', '12-16', '13-9' ,
'28-5', '2-26', '44-2', '44-1', '28-2', '0-18', '15-12', '79-0', '66-0', '45-4' ,
]

truc_model['cnot_41'] = [
'0-0', '0-2', '2-0', '4-1', '2-2', '34-0', '4-4', '8-1', '1-2', '1-0' ,
'5-2', '5-0', '22-0', '13-0', '4-5', '15-1', '26-1', '2-1', '20-1', '0-1' ,
'1-4', '8-2', '2-5', '0-5', '8-0', '12-0', '9-0', '12-4', '15-0', '15-2' ,
'0-18', '24-0', '18-1', '8-8', '37-0', '56-0', '18-2', '1-8', '2-4', '5-8' ,
'20-2', '4-13', '18-0', '24-4', '20-0', '0-21', '5-4', '9-5', '1-21', '1-13' ,

'9-1', '13-4', '33-1', '0-20', '1-18', '8-9', '25-0', '26-0', '0-13', '12-5' ,
'4-8', '12-1', '4-9', '9-4', '25-2', '5-9', '34-4', '22-4', '8-4', '34-1' ,
'5-1', '8-16', '45-1', '15-13', '0-8', '50-1', '1-9', '2-8', '24-2', '26-2' ,
'5-5', '9-12', '4-12', '81-0', '45-0', '13-5', '2-12', '35-0', '39-1', '9-8' ,
'15-8', '64-0', '13-2', '13-1', '46-1', '35-2', '74-0', '45-2', '35-1', '2-21' ,

'59-0', '41-1', '4-2', '41-0', '60-0', '1-26', '39-0', '44-0', '1-1', '65-0' ,
'8-5', '34-2', '20-4', '41-2', '39-2', '37-4', '28-0', '70-0', '44-1', '4-24' ,
'33-2', '33-0', '22-1', '1-25', '2-18', '50-0', '58-0', '28-4', '60-1', '34-5' ,
'50-2', '46-4', '5-18', '12-2', '9-2', '5-12', '1-5', '2-13', '0-24', '35-4' ,
'30-4', '4-0', '41-5', '54-0', '44-4', '69-0', '75-0', '46-0', '56-2', '28-1' ,

'25-1', '1-16', '18-9', '26-4', '12-9', '77-0', '30-1', '33-4', '1-12', '28-5' ,
'0-36', '33-8', '4-16', '34-8', '2-9', '9-20', '22-2', '41-4', '30-0', '26-8' ,
'51-1', '1-20', '59-1', '0-9', '18-5', '24-1', '9-28', '18-8', '20-12', '15-4' ,
'25-5', '0-16', '30-2', '24-9', '37-1', '13-12', '2-44', '26-5', '2-16', '45-4' ,
'66-0', '15-16', '1-24', '0-39', '9-9', '24-5', '20-5', '22-5', '5-13', '15-5' ,
]


truc_model['cnot_45'] = [
'0-0', '0-2', '2-0', '4-5', '2-2', '8-5', '1-2', '1-0', '5-2', '5-0' ,
'12-2', '22-2', '2-1', '34-2', '12-5', '0-1', '13-2', '9-2', '2-5', '4-2' ,
'0-5', '26-5', '15-5', '56-2', '9-4', '1-1', '9-0', '5-5', '20-2', '22-5' ,
'13-0', '20-5', '26-2', '4-0', '5-1', '8-2', '9-8', '1-4', '15-2', '9-5' ,
'12-0', '1-5', '18-5', '15-0', '8-0', '34-5', '24-2', '8-9', '12-12', '9-9' ,
'45-2', '35-2', '4-9', '13-5', '4-1', '13-9', '18-2', '25-2', '1-18', '18-4' ,
'25-5', '30-2', '1-8', '33-2', '25-0', '37-2', '20-0', '5-16', '4-4', '2-4' ,
'18-1', '8-1', '18-0', '50-2', '1-13', '9-16', '12-9', '5-9', '9-12', '15-4' ,
'33-5', '30-5', '8-18', '0-21', '26-9', '5-12', '24-5', '5-8', '26-0', '22-0' ,
'25-4', '46-0', '13-4', '5-4', '20-9', '46-2', '15-1', '9-1', '2-9', '2-13' ,
'26-8', '13-8', '0-4', '0-30', '24-0', '4-12', '37-5', '1-25', '15-9', '8-12' ,
'2-30', '24-9', '2-25', '18-9', '45-0', '4-8', '8-4', '12-1', '26-12', '39-2' ,
'15-8', '30-0', '8-24', '22-4', '35-0', '13-16', '39-4', '22-8', '41-2', '12-8' ,
'12-18', '34-0', '2-21', '37-0', '0-39', '39-5', '18-12', '0-8', '2-26', '1-16' ,
'9-18', '41-0', '15-12', '30-4', '4-18', '39-0', '35-5', '2-8', '30-1', '18-8' ,
'33-0', '1-9', '20-4', '5-30', '35-4', '2-34', '0-25', '4-13', '50-0', '0-12' ,
'8-16', '1-21', '44-2', '13-1', '25-1', '8-8', '28-1', '30-8', '33-4', '20-1' ,
'24-8', '20-12', '0-9', '28-5', '22-1', '28-2', '25-8', '0-20', '2-12', '59-0' ,
'41-5', '0-24', '39-1', '65-0', '56-0', '24-16', '20-8', '45-4', '22-9', '24-4' ,
'4-16', '66-0', '12-16', '1-20', '0-18', '4-24', '51-2', '58-0', '5-13', '34-1' ,
]
















