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

def get_operator_two_zeropi(Ec0=1.0, truc1=30, truc_tot=50, charge_pick=False, n_cut=60, phi_cut=200):
    zp = scq.Circuit(zp_yml, from_file=False)
    zp.Ec0 = Ec0
    zp.configure(transformation_matrix=np.linalg.inv(transform_2zeropi))

    ##############################################################################################
    ### Construct subsystem, calculate eigenvalues
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [100, 100]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)

    zp.cutoff_ext_1, zp.cutoff_ext_5 = phi_cut, phi_cut
    zp.cutoff_n_2, zp.cutoff_n_6 = n_cut, n_cut

    ### the two-line code below takes time when truc1 is large
    eval0, eket0 = zp.subsystems[0].eigensys(evals_count=truc1)
    eval1, eket1 = zp.subsystems[1].eigensys(evals_count=truc1)

    sorted_idx0 = np.argsort(eval0)
    eval0 = eval0[sorted_idx0]
    eval0 = eval0 - eval0[0]
    eket0 = ssp.csr_matrix([eket0[:,idx] for idx in range(truc1)])

    sorted_idx1 = np.argsort(eval1)
    eval1 = eval1[sorted_idx1]
    eval1 = eval1 - eval1[0]
    eket1 = ssp.csr_matrix([eket1[:,idx] for idx in range(truc1)])

    # get the n-operator in qubit basis of single qubit
    n_theta0 = (eket0 @ zp.subsystems[0].n2_operator() @ eket0.conj().T).todense()
    n_theta1 = (eket1 @ zp.subsystems[1].n6_operator() @ eket1.conj().T).todense()
    ##############################################################################################
    ###  Truncate two qubits using charge matrix elements
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    thresh_matrix_element=1e-4
    if charge_pick:
        hspace_0 = [0, 2]
        hspace_1 = [0, 2]
        for s in hspace_0:
            for i in range(truc1):
                if np.abs(n_theta0[s, i]) > thresh_matrix_element and i not in hspace_0:
                    hspace_0.append(i)
        hspace_0.sort()
        for s in hspace_1:
            for i in range(truc1):
                if np.abs(n_theta1[s, i]) > thresh_matrix_element and i not in hspace_1:
                    hspace_1.append(i)
        hspace_1.sort()
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
    n2, n6 = symbols('n2 n6')
    g = float(zp.sym_interaction((1,0), return_expr=True).coeff(n2*n6) )
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
    def find_overlap(eket):
        overlaps = np.array([[np.abs( (eket @ bare_state[i][j].data).todense()[0,0] )
                            for j in range(len(eval1))]
                                for i in range(len(eval0))])
        flat_array = overlaps.flatten() # Flatten the 2D array
        # Find the indices of the top 3 largest values (in the flattened 1D array)
        top_indices_flat = np.argpartition(-flat_array, 10)[:10]
        # Convert the flat indices to 2D indices
        top_indices_2d = np.unravel_index(top_indices_flat, overlaps.shape)
        # Extract the values corresponding to the indices
        top_values = overlaps[top_indices_2d]
        # Sort the values in descending order
        sorted_indices = np.argsort(-top_values)  # Use a negative sign for descending order
        sorted_top_indices = [tuple(zip(top_indices_2d[0], top_indices_2d[1]))[i] for i in sorted_indices]
        sorted_top_values = top_values[sorted_indices]
        return sorted_top_indices, sorted_top_values

    result = Parallel(n_jobs=10, verbose=0)(delayed(find_overlap)(arg) for arg in eket_tot)
    top_index = [result[i][0] for i in range(eval_tot.shape[0])]
    top_overlap = [result[i][1] for i in range(eval_tot.shape[0])]
    # for i in range(truc_tot):
    #     print(i, top3_index[i], top3_overlap[i])

    ##############################################################################################
    ### Get the dressed states index
    index_array = [] # array index in each qubit (# in hspace_0, hspace_1)
    for i, index in enumerate(top_index):
        j=0
        while j < len(index):
            if index[j] not in index_array:
                index_array.append(index[j])
                break
            else:
                j+=1
            if j==10:
                index_array.append((0,0))
                print(i, 'need to further compare overlap')
    hspace_full = [(str(hspace_0[idx[0]])+'-'+str(hspace_1[idx[1]])) for idx in index_array] # actual state index in each qubit

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
    [tg, drive_amp, detune, 
     H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx, 
     option_ideal, option_noisy] = arg_all

    pulse_args = {'drive_amp_A': drive_amp,
                'drive_freq_A': W_target + 2*np.pi*detune,
                'gate_time': tg}
    tlist = np.linspace(0, tg, num=3*int(tg))  # total time

    propagator = get_propagator(H_qbt_drive, tlist, num_cpus, c_op_list, pulse_args, 
                                logi_idx, option_ideal, option_noisy)
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

def load_qubit_data_2q(truc_full, import_2000=False, truc1=300):
    """
    Loads the energy spectrum and matrix elements (n_theta, n_phi) for the 0-π qubit.
    The function "generate_data()" in sigmaX_fidelity_import_paras.py can generate the data
    """    
    if import_2000:
        folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2=2000_pick=False/'
    else:
        folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2=1000_pick=True/'
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()[:truc_full]
    eket_tot = ssp.csr_matrix(np.load(folder+ 'eket_tot.npy'))[:truc_full]
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()[:truc_full]
    n_theta0_dress = 2*np.pi* np.load(folder+'n_theta0_dress.npy')
    n_theta1_dress = 2*np.pi* np.load(folder+'n_theta1_dress.npy')
    n_theta0_dress = truncate_2(n_theta0_dress, np.arange(truc_full))
    n_theta1_dress = truncate_2(n_theta1_dress, np.arange(truc_full))

    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2=1000_pick=True/'
    hspace_0 = pd.read_csv(folder+ 'hspace_0.txt').to_numpy().flatten()
    hspace_1 = pd.read_csv(folder+ 'hspace_1.txt').to_numpy().flatten()
    dim_0 = len(hspace_0)
    dim_1 = len(hspace_1)    
    logi_state = ['0-0', '0-2', '2-0', '2-2']
    return hspace_full, eket_tot, eval_tot, n_theta0_dress, n_theta1_dress, dim_0, dim_1, logi_state


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

def construct_c_ops_xgate(n_hspace, drive_truc, Gamma_t1, gamma_dephase_new, tphi, hspace, state_idx_tphi):
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
    for i in range(1, n_hspace):
        for j in range(i): # only consider downwards deacy
            jump_t1.append(np.sqrt(gamma_decay_new[j, i]) * qt.basis(n_hspace, j) * qt.basis(n_hspace, i).dag())
    for i, state in enumerate(hspace):
        if state in list(state_idx_tphi):
            idx = list(state_idx_tphi).index(state) 
            jump_tphi.append(np.sqrt(2 * gamma_dephase_new[idx]) * qt.basis(n_hspace, i).proj())
        else:
            jump_tphi.append(qt.Qobj(np.zeros((n_hspace, n_hspace))))
    
    # print('np.shape(jump_t1)=',  np.shape(jump_t1), '; np.shape(jump_tphi)=',  np.shape(jump_tphi))
    return jump_t1 + jump_tphi

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
    tlist = np.linspace(0, tg, num= 3*int(tg))
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


def find_overlap(eket, *arg):
    bare_state, dim_0, dim_1 = arg
    overlaps = np.array([[np.abs( (eket @ bare_state[i][j].data).todense()[0,0] )
                        for j in range(dim_1)]
                            for i in range(dim_0)])
    flat_array = overlaps.flatten() # Flatten the 2D array
    # Find the indices of the top 3 largest values (in the flattened 1D array)
    top_indices_flat = np.argpartition(-flat_array, 10)[:10]
    # Convert the flat indices to 2D indices
    top_indices_2d = np.unravel_index(top_indices_flat, overlaps.shape)
    # Extract the values corresponding to the indices
    top_values = overlaps[top_indices_2d]
    # Sort the values in descending order
    sorted_indices = np.argsort(-top_values)  # Use a negative sign for descending order
    sorted_top_indices = [tuple(zip(top_indices_2d[0], top_indices_2d[1]))[i] for i in sorted_indices]
    sorted_top_values = top_values[sorted_indices]
    return sorted_top_indices, sorted_top_values


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

def get_jump_op_charge_pick(state_vec, *args):
    state_i, state_j = state_vec
    dim_0, dim_1, n_theta, gamma_dephase, eket_tot, qubit_a = args
    if qubit_a:
        # t_1
        ladder_ij = qt.basis(dim_0, state_i) * qt.basis(dim_0, state_j).dag()
        a_ij_I = qt.tensor(ladder_ij, qt.qeye(dim_1))
        jump_t1 = qt.Qobj( eket_tot @ ( np.sqrt(abs(n_theta[state_i, state_j]))* a_ij_I ).data @ eket_tot.conj().T )
        # t_phi
        proj_jj = qt.basis(dim_0, state_j).proj()
        a_jj_I = qt.tensor(proj_jj, qt.qeye(dim_1))
        jump_tphi = qt.Qobj( eket_tot @ ( np.sqrt(2*gamma_dephase[state_j] )* a_jj_I  ).data @ eket_tot.conj().T)
    else: # qubit_b
        # t_1
        ladder_ij = qt.basis(dim_1, state_i) * qt.basis(dim_1, state_j).dag()
        a_I_ij = qt.tensor(qt.qeye(dim_0), ladder_ij)
        jump_t1 = qt.Qobj( eket_tot @ ( np.sqrt(abs(n_theta[state_i, state_j]))* a_I_ij ).data @ eket_tot.conj().T )
        # t_phi
        proj_jj = qt.basis(dim_1, state_j).proj()
        a_I_jj = qt.tensor(qt.qeye(dim_0), proj_jj)
        jump_tphi = qt.Qobj( eket_tot @ ( np.sqrt(2*gamma_dephase[state_j] )* a_I_jj  ).data @ eket_tot.conj().T)
    return [jump_t1, jump_tphi]

def get_jump_op_decay(state_vec, *args):
    state_i, state_j = state_vec
    dim_0, dim_1, n_theta, eket_tot, qubit_a = args
    if qubit_a:
        ladder_ij = qt.basis(dim_0, state_i) * qt.basis(dim_0, state_j).dag()
        jump_t1 = np.sqrt(abs(n_theta[state_i, state_j])) * qt.tensor(ladder_ij, qt.qeye(dim_1))
    else: # qubit_b
        ladder_ij = qt.basis(dim_1, state_i) * qt.basis(dim_1, state_j).dag()
        jump_t1 = np.sqrt(abs(n_theta[state_i, state_j])) * qt.tensor(qt.qeye(dim_0), ladder_ij)
    jump_t1 = qt.Qobj( eket_tot @ ( np.sqrt(abs(n_theta[state_i, state_j])) * jump_t1 ).data @ eket_tot.conj().T )
    return jump_t1

def get_jump_op_dephase(state_j, *args):
    dim_0, dim_1, gamma_dephase, eket_tot, qubit_a = args
    if qubit_a:
        proj_jj = qt.basis(dim_0, state_j).proj()
        jump_tphi = np.sqrt( 2*gamma_dephase[state_j] ) * qt.tensor(proj_jj, qt.qeye(dim_1))
    else: # qubit_b
        proj_jj = qt.basis(dim_1, state_j).proj()
        jump_tphi = np.sqrt( 2*gamma_dephase[state_j] ) * qt.tensor(qt.qeye(dim_0), proj_jj)
    jump_tphi = qt.Qobj( eket_tot @ jump_tphi.data @ eket_tot.conj().T)
    return jump_tphi

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


def get_collapse_op(t1tphi_other, eket_tot_select, hspace_0_dim, hspace_1_dim):

    # tphi_logi = 100 # μs
    # gamma_decay_logi =  1 / 1600e3
    # gamma_dephase_logi = 1 / 1e3 / tphi_logi
    # gamma_decay_other =  1 / 1e3 / t1tphi_other
    # gamma_dephase_other = 1 / 1e3 / t1tphi_other
    # gamma_decay_old   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * 300
    # gamma_dephase_old = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * 300

    # if charge_pick == False:
    #     args = [truc1, gamma_decay_old, gamma_dephase_old, eket_tot]
    #     jump_op = Parallel(n_jobs=100)(delayed(ut.get_jump_op)(state, *args) for state in range(1,truc1))
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

    folder = f'../data/3ncut_two_zeropi/truc1=500/'
    gamma_q0 = pd.read_csv(folder+ 'data_gamma_qubit0.txt')
    gamma_q1 = pd.read_csv(folder+ 'data_gamma_qubit1.txt')
    gamma_decay_48_q0 = gamma_q0['t1_50us_48'].to_numpy() *50 /t1tphi_other
    gamma_decay_48_q1 = gamma_q1['t1_50us_48'].to_numpy() *50 /t1tphi_other
    gamma_dephase_02_q0 = gamma_q0['tphi_02'].to_numpy() *50 /t1tphi_other
    gamma_dephase_02_q1 = gamma_q1['tphi_02'].to_numpy() *50 /t1tphi_other

    qubit_a = True
    arg_a = [hspace_0_dim, hspace_1_dim, gamma_decay_48_q0, gamma_dephase_02_q0, eket_tot_select, qubit_a]
    jump_op_a = Parallel(n_jobs=100)(delayed(get_jump_op_charge_pick)(state, *arg_a) for state in range(1,hspace_0_dim))

    qubit_a = False
    arg_b = [hspace_0_dim, hspace_1_dim, gamma_decay_48_q1, gamma_dephase_02_q1, eket_tot_select, qubit_a]
    jump_op_b = Parallel(n_jobs=100)(delayed(get_jump_op_charge_pick)(state, *arg_b) for state in range(1,hspace_1_dim))
    jump_t1_list = np.array(jump_op_a)[:,0].tolist() + np.array(jump_op_b)[:,0].tolist()
    jump_tphi_list = np.array(jump_op_a)[:,1].tolist() + np.array(jump_op_b)[:,1].tolist()
    jump_t1_list = [qt.Qobj(matrix) for matrix in jump_t1_list]
    jump_tphi_list = [qt.Qobj(matrix) for matrix in jump_tphi_list]
    c_op_list = jump_t1_list + jump_tphi_list

    return c_op_list

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
            if num_digits != None and isinstance(data[0], float):
                print(', '.join(map(str, np.round(data[i:i + num_each_row], num_digits).tolist())), ',')
            else:
                print(', '.join(map(str, data[i:i + num_each_row])), ',')
    print('])')

def print_time():
    """Print the current time in Mountain Time (America/Denver)."""
    mountain_time = datetime.now(pytz.timezone('America/Denver'))
    print("\nCurrent Mountain Time:", mountain_time)    


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

cz_truc_model = {}
cz_truc_model['all_path'] = [
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

cz_truc_model['short_path'] = [
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

cz_truc_model['hand_pick'] = [
'0-0', '0-1', '1-0', '0-2', '2-0', '0-4', '4-0', '1-1', '0-5', '2-1' ,
'5-0', '1-2', '0-8', '2-2', '8-0', '1-4', '4-1', '0-9', '2-4', '9-0' ,
'1-5', '5-1', '4-2', '0-12', '2-5', '1-8', '12-0', '5-2', '0-13', '0-16' ,
'8-1', '4-4', '13-0', '15-0', '2-8', '1-9', '9-1', '8-2', '5-4', '4-5' ,
'0-18', '2-9', '1-12', '0-20', '0-21', '18-0', '9-2', '12-1', '5-5', '0-24' ,
'22-0', '2-12', '13-1', '24-0', '0-25' , 
] 

cnot_truc_model = {}
cnot_truc_model['all_path'] = [
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

cnot_truc_model['short_path'] = [
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

cnot_truc_model['hand_pick'] = [
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