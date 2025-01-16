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

def set_fig_font():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE, labelsize=SMALL_SIZE)     # fontsize of the axes
    plt.rc(['xtick', 'ytick' ], labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

def print_soln(xk, convergence=0):
    print("Best Soln:", xk)
    print("convergence", np.round(convergence,4))
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

def drive_gauss_A(t: float, args: dict) -> float:
    A = args.get('drive_amp_A', 0)
    wd = args.get('drive_freq_A', 0)
    tg = args.get('gate_time', 0)
    return A * (np.exp(-8 * t * (t - tg) / tg**2) - 1) * np.cos(wd * t) * (0<=t<=tg)

def drag_A(t: float, args: dict) -> float:
    A = args.get('drive_amp_A', 0)
    wd = args.get('drive_freq_A', 0)
    tg = args.get('gate_time', 0)
    alpha = args.get('alpha_A', 0)
    vg = A * (np.exp(-8 * t * (t - tg) / tg**2) - 1)
    return ( vg* np.cos(wd* t) + alpha* (vg+A)* (-8*(2*t-tg)/tg**2)* np.sin(wd* t) ) * (0<=t<=tg)



# Drive Coefficient on qubit B
def drive_cos_B(t: float, args: dict) -> float:
    A = args.get('drive_amp_B', 0)
    wd = args.get('drive_freq_B', 0)
    tg = args.get('gate_time', 0)
    return A * np.cos(wd * t) * (0<=t<=tg)

def drive_gauss_B(t: float, args: dict) -> float:
    A = args.get('drive_amp_B', 0)
    wd = args.get('drive_freq_B', 0)
    tg = args.get('gate_time', 0)
    return A * (np.exp(-8 * t * (t - tg) / tg**2) - 1) * np.cos(wd * t) * (0<=t<=tg)

def drag_B(t: float, args: dict) -> float:
    A = args.get('drive_amp_B', 0)
    wd = args.get('drive_freq_B', 0)
    tg = args.get('gate_time', 0)
    alpha = args.get('alpha_B', 0)
    vg = A * (np.exp(-8 * t * (t - tg) / tg**2) - 1)
    return ( vg* np.cos(wd* t) + alpha* (vg+A)* (-8*(2*t-tg)/tg**2)* np.sin(wd* t) ) * (0<=t<=tg)


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


# def get_operator_two_zeropi(Ec0=1.0, trunc_dim=10, get_eval=False):

#     zp = scq.Circuit(zp_yml, from_file=False)
#     zp.Ec0 = Ec0
#     zp.configure(transformation_matrix=np.linalg.inv(transform_2zeropi))

#     ##############################################################################################
#     ### 3. Construct subsystem, calculate eigenvalues
#     system_hierarchy = [[1,2],  [5,6]]
#     subsystem_trunc_dims = [trunc_dim, trunc_dim]
#     zp.configure(system_hierarchy=system_hierarchy,
#                 subsystem_trunc_dims=subsystem_trunc_dims)

#     zp.cutoff_ext_1 = 100
#     zp.cutoff_n_2 = 30
#     zp.cutoff_ext_5 = 100
#     zp.cutoff_n_6 = 30

#     eval0, eket0 = zp.subsystems[0].eigensys(evals_count=trunc_dim)
#     eval1, eket1 = zp.subsystems[1].eigensys(evals_count=trunc_dim)
#     eval0 = eval0 - eval0[0]
#     eval1 = eval1 - eval1[0]
#     eket0 = ssp.csr_matrix([eket0[:,idx] for idx in range(trunc_dim)])
#     eket1 = ssp.csr_matrix([eket1[:,idx] for idx in range(trunc_dim)])

    ##############################################################################################
    ### 4. Sort bare system eigenvalues, get dressed index
    # eval_bare_1d = []
    # order_1d = []
    # for idx, val0 in enumerate(eval0):
    #     for jdx, val1 in enumerate(eval1):
    #         eval_bare_1d.append(val0 + val1)
    #         order_1d.append(str(idx) + str(jdx))
    # sort_idx = np.argsort(eval_bare_1d)
    # order_sort = [order_1d[i] for i in sort_idx]

    # get the n-operator in qubit basis of single qubit
    # n_theta1 = (eket0 @ zp.subsystems[0].n2_operator() @ eket0.conj().T).todense()
    # n_theta2 = (eket1 @ zp.subsystems[1].n6_operator() @ eket1.conj().T).todense()

    # Hint = qt.tensor(qt.Qobj(n_theta1) , qt.Qobj(n_theta2))
    # n2, n6 = symbols('n2 n6')
    # g = zp.sym_interaction((1,0), return_expr=True).coeff(n2*n6)
    # H_bare = (  qt.tensor(qt.Qobj(np.diag(eval0)),  qt.identity(trunc_dim))
    #         +  qt.tensor(qt.identity(trunc_dim),  qt.Qobj(np.diag(eval1))) )


    # # Get the eigenvalues of coupled system
    # Htot = g* Hint + H_bare
    # eval_tot, eket_tot = Htot.eigenstates()
    # eval_tot = eval_tot - eval_tot[0]



    # bare_state = [[qt.tensor(qt.basis(trunc_dim, i), qt.basis(trunc_dim, j))
    #                     for j in range(trunc_dim)]
    #                         for i in range(trunc_dim)]
    # order_sort = []
    # for i in range(trunc_dim**2):
    #     overlaps = np.array([[np.abs(eket_tot[i].overlap(bare_state[j][k]))
    #                         for k in range(trunc_dim)]
    #                             for j in range(trunc_dim)])
    #     idx = np.unravel_index(np.argmax(overlaps), overlaps.shape)
    #     if (str(idx[0])+str(idx[1])) not in order_sort:
    #         order_sort.append(str(idx[0])+str(idx[1]))
    #     else:
    #         overlaps[idx[0], idx[1]] = 0
    #         idx2 = np.unravel_index(np.argmax(overlaps), overlaps.shape)
    #         order_sort.append(str(idx2[0])+str(idx2[1]))



    # # Get n_theta in dressed basis
    # N_a = qt.tensor(qt.Qobj(n_theta1), qt.identity(trunc_dim))
    # n_theta1_dress = np.zeros((trunc_dim**2, trunc_dim**2), dtype=complex)
    # N_b = qt.tensor(qt.identity(trunc_dim) , qt.Qobj(n_theta2))
    # n_theta2_dress = np.zeros((trunc_dim**2, trunc_dim**2), dtype=complex)
    # for idx, vec1 in enumerate(eket_tot):
    #     for jdx, vec2 in enumerate(eket_tot):
    #         n_theta1_dress[idx,jdx] = np.array(vec1.dag()* N_a* vec2).flatten()[0]
    #         n_theta2_dress[idx,jdx] = np.array(vec1.dag()* N_b* vec2).flatten()[0]

    # logi_index = [order_sort.index(x) for x in logi_space]
    # trunc_states = logi_space + ['50']

    # thresh_matrix_emt = 0.01
    # for i in logi_index:
    #     for j in range(n_theta1_dress.shape[0]):
    #         if np.abs(n_theta1_dress[i,j]) > thresh_matrix_emt or np.abs(n_theta2_dress[i,j]) > thresh_matrix_emt:
    #             j_state = order_sort[j]
    #             if j_state not in trunc_states:
    #                 trunc_states.append(j_state)
    # idxs = [order_sort.index(x) for x in trunc_states]

    # diag_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(eval_tot))
    # H0 = truncate_2(diag_hamiltonian, idxs)

    # n_theta1_truc = truncate_2(n_theta1_dress, idxs)
    # n_theta2_truc = truncate_2(n_theta2_dress, idxs)

    # truc = len(trunc_states)
    # state_tot = [qt.basis(truc, i) for i in range(truc)]
    # e_ops = [qt.basis(truc, i) * qt.basis(truc, i).dag()
    #         for i in range(truc)]

    # return [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    #         eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot ]


def get_operator_two_zeropi_v2(Ec0=1.0, truc1=30, truc_tot=50, thresh_matrix_element=1e-4, charge_pick=False):
    zp = scq.Circuit(zp_yml, from_file=False)
    zp.Ec0 = Ec0
    zp.configure(transformation_matrix=np.linalg.inv(transform_2zeropi))

    ##############################################################################################
    ### Construct subsystem, calculate eigenvalues
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [100, 100]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims)

    zp.cutoff_ext_1 = 100
    zp.cutoff_n_2 = 30
    zp.cutoff_ext_5 = 100
    zp.cutoff_n_6 = 30

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
    # n_theta0 = np.round(eket0 @ zp.subsystems[0].n2_operator() @ eket0.conj().T, 8).todense()
    # n_theta1 = np.round(eket1 @ zp.subsystems[1].n6_operator() @ eket1.conj().T, 8).todense()
    n_theta0 = (eket0 @ zp.subsystems[0].n2_operator() @ eket0.conj().T).todense()
    n_theta1 = (eket1 @ zp.subsystems[1].n6_operator() @ eket1.conj().T).todense()
    ##############################################################################################
    ###  Truncate two qubits using charge matrix elements
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
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

    return [hspace_0, hspace_1, top_index, top_overlap,
            n_theta0_dress, n_theta1_dress, eval_tot, hspace_full]

# def cz_fidelity(prop, state_logi):
#     ''' Compute fidelity to controlled Z gate (CZ) including the folloiwng steps:

#     1. remove global phase
#     2. perform two local Rz(phi) corrections

#     '''
#     Uz = remove_global_phase(qt.tensor( rz(dphi(prop, (2,2))),
#                                         rz(dphi(prop, (1,1)))))

#     Uc = qt.Qobj([ [prop.matrix_element(s1, s2) for s1 in state_logi]
#             for s2 in state_logi  ])
#     Uc_reshaped = qt.Qobj(Uc.data, dims=[[2, 2], [2, 2]])
#     Ucprime = remove_global_phase(Uz * Uc_reshaped)

#     #fidelity measure given on page 3 of Nesterov et al.
#     # fidelity = gate_fidelity( cz_gate(), Ucprime)
#     fidelity = qt.average_gate_fidelity(Ucprime, target=cz_gate())
#     return fidelity



def cz_fidelity_optimize(arg, *args):
    detune, drive_amp = arg # Independent arguments that can be optimized over
    [H_qbt_drive, W_target, state_logi, tg] = args # System arguments
    pulse_args = {'drive_amp_A': drive_amp,
                'drive_freq_A': W_target + 2*np.pi*detune,
                'gate_time': tg}
    tlist = np.linspace(0, tg, num=int(3*tg))  # total time
    prop = qt.propagator( H=H_qbt_drive,
                          t=tlist,
                          args=pulse_args,)[-1]  # get the propagator at the final time step
    fidelity = cz_fidelity( prop, state_logi )
    return np.log10(1-fidelity)


def cz_fidelity_parallel(arg, *args):
    tg, detune, drive_amp = arg # Independent arguments that can be optimized over
    [n_cpu, H_qbt_drive, W_target, state_logi] = args # System arguments

    pulse_args = {'drive_amp_A': drive_amp,
                'drive_freq_A': W_target + 2*np.pi*detune,
                'gate_time': tg}
    tlist = np.linspace(0, tg, num=int(3*tg))  # total time
    prop = qt.propagator( H=H_qbt_drive,
                        t=tlist,
                        args=pulse_args,
                        options=qt.Options( num_cpus=1 ),
                        num_cpus=n_cpu,
                        parallel=True,
                        )[-1]  # get the propagator at the final time step
    fidelity = cz_fidelity( prop, state_logi )
    return np.log10(1-fidelity)




# def get_fidelity_cz_dark(args_indep, *args):
#     detune, drive_amp, d_eta = args_indep # Independent arguments that can be optimized over
#     [H0, n_theta1_truc, n_theta2_truc, eta, W_target, tg, state_tot, drive_ab] = args # System arguments

#     drive_term = n_theta2_truc + (eta+d_eta) * n_theta1_truc  if drive_ab else n_theta2_truc
#     H_qbt_drive = [H0, [2*np.pi* drive_term, drive_gauss_A] ]

#     pulse_args = {'drive_amp_A': drive_amp,
#                 'drive_freq_A': W_target + 2*np.pi*detune,
#                 'gate_time': tg}
#     tlist = np.linspace(0, tg, num=int(tg))  # total time
#     prop = qt.propagator( H=H_qbt_drive,
#                           t=tlist,
#                           args=pulse_args,)[-1]  # get the propagator at the final time step
#     fidelity = cz_fidelity( prop, state_tot )
#     return np.log10(1-fidelity)


# def get_fidelity_cz_dark_print(args_indep, *args):
#     tg, detune, drive_amp, d_eta = args_indep # Independent arguments that can be optimized over
#     [H0, n_theta1_truc, n_theta2_truc, eta, W_target, state_tot, drive_ab] = args # System arguments

#     drive_term = n_theta2_truc + (eta+d_eta) * n_theta1_truc  if drive_ab else n_theta2_truc
#     # drive_term = n_theta1_truc + (eta+d_eta) * n_theta2_truc  if drive_ab else n_theta2_truc
#     H_qbt_drive = [H0, [2*np.pi* drive_term, drive_gauss_A] ]

#     pulse_args = {'drive_amp_A': drive_amp,
#                 'drive_freq_A': W_target + 2*np.pi*detune,
#                 'gate_time': tg}
#     tlist = np.linspace(0, tg, num=int(tg))  # total time
#     prop = qt.propagator( H=H_qbt_drive,
#                           t=tlist,
#                           args=pulse_args,)[-1]  # get the propagator at the final time step
#     fidelity = cz_fidelity( prop, state_tot )
#     return np.log10(1-fidelity)








        #  = args # System arguments
    # if one_drive == True:
    #     detune, drive_amp = args_indep # Independent arguments that can be optimized over
    # else:
    #     detune_A, drive_amp_A, detune_B, drive_amp_B = args_indep
    # if one_drive == True:
    #     drive_term = n_theta2_truc + n_theta1_truc if drive_ab==True else n_theta2_truc
    #     H_qbt_drive = [diag_hamiltonian_trunc, [2*np.pi* drive_term, cos_pulse] ]
    #     pulse_args = {'drive_amp': drive_amp, 'drive_freq': detune + trans_freq, 'gate_time': tg}
    # else:
    #     H_qbt_drive = [diag_hamiltonian_trunc,
    #                     [2*np.pi*n_theta1_truc, drive_coeff_A],
    #                     [2*np.pi*n_theta2_truc, drive_coeff_B]  ]
    #     gauss_env = lambda t: gaussian_envelope(t, tg)
    #     pulse_args = {'drive_amp_A': drive_amp_A, 'drive_freq_A': detune_A + trans_freq,
    #                 'drive_amp_B': drive_amp_B, 'drive_freq_B': detune_B + trans_freq, 'gate_time': tg}
    ###############################################
    # convert the product states to the closes eigenstates of the dressed system
    # product_states = ['00', '20', '02', '22',
    #                 '01', '21', '05', '25',
    #                 '10', '50', '12', '52' ]
    # trans_freq = np.abs(transition_frequency(order_sort.index('20') , order_sort.index('50'), eval_tot))
    # idxs = [order_sort.index(i) for i in product_states]
    # total_truncation = np.max(idxs) +1
    # states = [qt.basis(total_truncation, idx) for idx in idxs]
#####################################################
### CNOT Raman gate


def get_fidelity_cnot_1A0(arg_de, *args):
    drive_amp, tg, detune = arg_de

    (w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
    n_theta2_20_14, state_tot, H_qbt_drive, n1n1) = args

    if n1n1:
        pulse_args = {'drive_amp_A': drive_amp* (n_theta1_20_14/n_theta1_00_14) ,
                    'drive_freq_A': w_00_14 + 2*np.pi*detune,
                    'drive_amp_B': drive_amp ,
                    'drive_freq_B': w_20_14 + 2*np.pi*detune,
                    'gate_time': tg }
    else:
        pulse_args = {'drive_amp_A': drive_amp* (n_theta1_20_14/n_theta2_00_14) ,
                    'drive_freq_A': w_00_14 + 2*np.pi*detune,
                    'drive_amp_B': drive_amp ,
                    'drive_freq_B': w_20_14 + 2*np.pi*detune,
                    'gate_time': tg }
    tlist = np.linspace(0, tg, num=int(tg))  # total time
    prop = qt.propagator( H=H_qbt_drive,
                          t=tlist,
                          args=pulse_args,
                        #   c_op_list=c_op_list,
                          )[-1]  # get the propagator at the final time step
    fidelity = cnot_fidelity( prop, state_tot )
    return np.log10(1-fidelity)





def cnot_fidelity(prop, state_tot):
    state_logi = state_tot[:4]
    Uc = qt.Qobj([ [prop.matrix_element(s1, s2) for s1 in state_logi]
            for s2 in state_logi  ], dims=[[2, 2], [2, 2]])

    XI = qt.tensor(qt.sigmax(), qt.qeye(2))

    # Uc_prime = XI* swap()* Uc* swap()* XI # for |14> state
    Uc_prime = swap()* Uc* swap() # for |45> state

    phase = np.angle(Uc_prime)

    x1 = 0.5* (- phase[1,1] + phase[2,3] - phase[3,2] + phase[0,0])
    x2 = 0.5* (  phase[1,1] - phase[2,3] - phase[3,2] + phase[0,0])
    x3 = 0.5* (- phase[1,1] - phase[2,3] + phase[3,2] + phase[0,0])
    U_bef = qt.Qobj(np.diag([1, np.exp(1j* x1), np.exp(1j* x2), np.exp(1j* (x1+x2))])
            , dims=[[2, 2], [2, 2]])
    U_aft = qt.Qobj(np.diag([1, np.exp(1j* x3), 1, np.exp(1j* x3)])
            , dims=[[2, 2], [2, 2]])

    U_final = np.exp(-1j* phase[0,0])* U_bef* Uc_prime* U_aft

    # fidelity = gate_fidelity( cnot(), U_final)
    fidelity = qt.average_gate_fidelity(U_final, target=cnot())
    return fidelity



def get_transition_freq_cnot(args, trans_goal='02-45_22-45'):
# def get_transition_freq(args, trans_goal='00-14_20-14'):

    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot  ] = args

    transition_ik_jk = []
    w_ik = []
    w_jk = []
    n_theta1_ik = []
    n_theta2_ik = []
    n_theta1_jk = []
    n_theta2_jk = []
    thresh = 0.01
    for idx in range(len(logi_space)):
        for jdx in range(idx+1, len(logi_space)):
            state_i = logi_space[idx]
            state_j = logi_space[jdx]

            i = order_sort.index(state_i)
            j = order_sort.index(state_j)
            for k in np.arange(0, len(eval_tot)):

                if (( np.abs(n_theta1_dress[i,k]) > thresh or np.abs(n_theta2_dress[i,k]) > thresh ) and (
                    np.abs(n_theta1_dress[j,k]) > thresh or np.abs(n_theta2_dress[j,k]) > thresh ) ):
                    transition_ik_jk.append(f'{state_i}-{order_sort[k]}_{state_j}-{order_sort[k]}')

                    w_ik.append( 2*np.pi*( np.abs( eval_tot[i] - eval_tot[k] )))
                    w_jk.append( 2*np.pi*( np.abs( eval_tot[j] - eval_tot[k] )))

                    n_theta1_ik.append(np.abs(n_theta1_dress[i ,k]))
                    n_theta2_ik.append(np.abs(n_theta2_dress[i ,k]))
                    n_theta1_jk.append(np.abs(n_theta1_dress[j ,k]))
                    n_theta2_jk.append(np.abs(n_theta2_dress[j ,k]))

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

    idx = transition_ik_jk.index(trans_goal)
    # print(transition_ik_jk)
# Trans ;  w_ij;    n_theta1; n_theta2; sum
# 00-14 ;  45.925 ;  0.032 ;  0.006 ;  0.038
# 20-14 ;  24.289 ;  0.046 ;  0.001 ;  0.047

    w_00_14 = w_ik[idx]
    w_20_14 = w_jk[idx]
    n_theta1_00_14 = n_theta1_ik[idx]
    n_theta2_00_14 = n_theta2_ik[idx]
    n_theta1_20_14 = n_theta1_jk[idx]
    n_theta2_20_14 = n_theta2_jk[idx]
    return w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14, n_theta2_20_14


def get_pop(args, *argz, state_0='00'):

    drive_amp_A, drive_amp_B, tg, detune_A, detune_B = args
    (n_theta1_20_14, n_theta1_00_14, w_00_14, w_20_14,
     trunc_states, logi_space, H_qbt_drive, e_ops ) = argz

    pulse_args = {'drive_amp_A': drive_amp_A* (n_theta1_20_14/n_theta1_00_14) ,
                'drive_freq_A': w_00_14 + 2*np.pi*detune_A,
                'drive_amp_B': drive_amp_B ,
                'drive_freq_B': w_20_14 + 2*np.pi*detune_B,
                'gate_time': tg }

    tlist = np.linspace(0, tg, num=1000)  # total time
    options = qt.Options(nsteps=10000, store_states=True)
    result = qt.mesolve(
        H=H_qbt_drive,
        rho0=qt.basis( len(trunc_states), logi_space.index(state_0) ),
        tlist=tlist,
        e_ops=e_ops,
        args=pulse_args,
        options=options
    )
    return result.expect[0][-1]


def cnot_phase_correct(U_kraus):
    U_final = []
    for u in U_kraus:
        phase = np.angle(u)
        x2 = 0.5* (  phase[0,0] + phase[1,3] - phase[2,2] - phase[3,1])
        x3 = 0.5* (  phase[0,0] - phase[1,3] + phase[2,2] - phase[3,1])
        x4 = 0.5* (  phase[0,0] - phase[1,3] - phase[2,2] + phase[3,1])
        Ur = qt.Qobj( np.diag([ 1, np.exp(1j*x3), np.exp(1j*x4), np.exp(1j*(x3+x4)) ]))
        Ul = qt.Qobj( np.diag([ 1, 1, np.exp(1j*x2), np.exp(1j*x2) ]))
        U_final.append( Ul * u * Ur)
    return U_final


def cz_phase_correct(U_kraus):
    U_final = []
    for u in U_kraus:
        Uz = remove_global_phase(qt.tensor( rz(dphi(u, (2,2))),
                                            rz(dphi(u, (1,1)))))
        Uc_reshaped = qt.Qobj(u.data, dims=[[2, 2], [2, 2]])
        U_final.append(remove_global_phase(Uz * Uc_reshaped))
    return U_final







def xgate_fidelity_optimize(arg, *args):
    [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, tg, drag] = args
    alpha_B = 0
    if drag == 0:
        alpha_A = 0
        [drive_amp_A, drive_amp_B, detune_A, detune_B] = arg
    else:
        [drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = arg

    n_cpu = 1
    argz = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, tg,
            drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
    return xgate_fidelity_fast(argz)


def xgate_fidelity_parallel(arg, *args):
    [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, drag] = args

    if drag == 0:
        [alpha_A, alpha_B] = [0, 0]
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = arg
    elif drag == 1:
        alpha_B = 0
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = arg
    else:
        [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = arg

    argz = [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, tg,
            drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
    return xgate_fidelity_fast(argz)


def xgate_fidelity_fast(argz):
    [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, tg,
    drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = argz

    H0_truc = truncate_2(H0, hilbert_space)
    drive_truc = truncate_2(drive_term, hilbert_space)
    H_qbt_drive = [H0_truc, [drive_truc, drag_A],
                            [drive_truc, drag_B],]

    pulse_args = {'drive_amp_A': drive_amp_A ,
            'drive_freq_A': w_trans_1 + 2*np.pi*detune_A,
            'drive_amp_B': drive_amp_B ,
            'drive_freq_B': w_trans_2 + 2*np.pi*detune_B,
            'gate_time': tg,
             'alpha_A': alpha_A,
             'alpha_B': alpha_B }
    tlist = np.linspace(0, tg,  num=int(tg))  # total time

    sesolve_args = []
    # logical_states = [0, 1]
    logical_states = [0, 2]
    logical_idx = [hilbert_space.index(s) for s in logical_states]
    prop = np.zeros((len(hilbert_space), len(logical_states)), dtype=np.complex128)
    if n_cpu > 1:
        options = qt.Options(num_cpus=n_cpu, nsteps=1e4)
        for i in logical_idx:
            sesolve_args.append([H_qbt_drive, i, tlist, pulse_args, options])
        pool = Pool(processes=n_cpu)
        for i, res in pool.imap_unordered(sesolve_parallel, sesolve_args):
            prop[:, logical_idx.index(i)] = res.states[-1].full().flatten()
    else:
        options = qt.Options(num_cpus=5, nsteps=1e4)
        for i in logical_idx:
            sesolve_args.append([H_qbt_drive, i, tlist, pulse_args, options])
        for se_arg in sesolve_args:
            i, res = sesolve_parallel(se_arg)
            prop[:, logical_idx.index(i)] = res.states[-1].full().flatten()
    prop = qt.Qobj(prop)

    Uc = truncate_2(prop, logical_idx)
    # error_leak = 1 - 0.5*(Uc.dag()*Uc).tr()
    # fidelity = gate_fidelity(Utarg=qt.sigmax(), Ucand=Uc)
    fidelity = qt.average_gate_fidelity(Uc, target=qt.sigmax())
    return np.log10(1-fidelity)

def sesolve_parallel(argz):
    H, psi0, tlist, pulse_args, options = argz
    return psi0, qt.sesolve(H, qt.basis(H[0].shape[0], psi0), tlist, options=options, args=pulse_args)


def cnot_fidelity_fast(arg_de, *args):
    tg, drive_amp_A, drive_amp_B, detune_A, detune_B = arg_de

    (w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
    n_theta2_20_14, state_tot, H_qbt_drive, n1n1) = args

    pulse_args = {'drive_amp_A': drive_amp_A ,
                'drive_freq_A': w_00_14 + 2*np.pi*detune_A,
                'drive_amp_B': drive_amp_B ,
                'drive_freq_B': w_20_14 + 2*np.pi*detune_B,
                'gate_time': tg }
    tlist = np.linspace(0, tg, num=2*int(tg))  # total time

    sesolve_args = []
    logical_states = [0, 2]
    logical_idx = [hilbert_space.index(s) for s in logical_states]
    prop = np.zeros((len(hilbert_space), len(logical_states)), dtype=np.complex128)
    if n_cpu > 1:
        options = qt.Options(num_cpus=n_cpu, nsteps=1e4)
        for i in logical_idx:
            sesolve_args.append([H_qbt_drive, i, tlist, pulse_args, options])
        pool = Pool(processes=n_cpu)
        for i, res in pool.imap_unordered(sesolve_parallel, sesolve_args):
            prop[:, logical_idx.index(i)] = res.states[-1].full().flatten()
    else:
        options = qt.Options(num_cpus=5, nsteps=1e4)
        for i in logical_idx:
            sesolve_args.append([H_qbt_drive, i, tlist, pulse_args, options])
        for se_arg in sesolve_args:
            i, res = sesolve_parallel(se_arg)
            prop[:, logical_idx.index(i)] = res.states[-1].full().flatten()
    prop = qt.Qobj(prop)

    Uc = truncate_2(prop, logical_idx)
    # error_leak = 1 - 0.5*(Uc.dag()*Uc).tr()
    # fidelity = gate_fidelity(Utarg=qt.sigmax(), Ucand=Uc)
    fidelity = qt.average_gate_fidelity(Uc, target=qt.sigmax())


    prop = qt.propagator( H=H_qbt_drive,
                          t=tlist,
                          args=pulse_args,)[-1]  # get the propagator at the final time step
    fidelity = cnot_fidelity( prop, state_tot )



    return np.log10(1-fidelity)



def xgate_fidelity_noise(args_indep, *args):
    [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state] = args
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = args_indep
    pulse_args = {'drive_amp_A': drive_amp_A ,
            'drive_freq_A': w_trans_1 + 2*np.pi*detune_A,
            'drive_amp_B': drive_amp_B ,
            'drive_freq_B': w_trans_2 + 2*np.pi*detune_B,
            'gate_time': tg,
                'alpha_A': 0,
                'alpha_B': 0 }
    tlist = np.linspace(0, tg,  num=3*int(tg))  # total time
    options =qt.Options(max_step=1e-4, nsteps=1e4, num_cpus=n_cpu )
    p = qt.propagator( H=H_qbt_drive,
                            t=tlist,
                            c_op_list=c_op_list,
                            options=options,
                            args=pulse_args,
                            num_cpus=n_cpu,
                            parallel=True,
                            )[-1]  # get the propagator at the final time step
    p0_kraus = qt.to_kraus(qt.to_super(p))
    p0_kraus = [truncate_2(i, logi_state) for i in p0_kraus]
    p0_super_2 = qt.kraus_to_super(p0_kraus)
    f_noise = qt.metrics.average_gate_fidelity(p0_super_2, target=qt.sigmax())
    return np.log10(1-f_noise)


def get_fidelity_noise_cz(args_indep, *args):
    tg, f_zl, f_qt, detune, drive_amp, d_eta = args_indep # Independent arguments that can be optimized over
    [H0, n_theta1_truc, n_theta2_truc, eta, W_target, max_steps, n_cpu, c_op_list, drive_ab ] = args # System arguments

    drive_term = n_theta2_truc + (eta+d_eta) * n_theta1_truc  if drive_ab else n_theta2_truc
    H_qbt_drive = [H0, [2*np.pi* drive_term, drive_gauss_A] ]

    pulse_args = {'drive_amp_A': drive_amp,
                'drive_freq_A': W_target + 2*np.pi*detune,
                'gate_time': tg}
    tlist = np.linspace(0, tg, num=3*int(tg))  # total time
    options =qt.Options(max_step=max_steps, nsteps=1e4, num_cpus=1 )
    p = qt.propagator( H=H_qbt_drive,
                            t=tlist,
                            c_op_list=c_op_list,
                            options=options,
                            args=pulse_args,
                            num_cpus=n_cpu,
                            parallel=True,
                            )[-1]  # get the propagator at the final time step
    p0_kraus = qt.to_kraus(qt.to_super(p))
    p0_kraus = [truncate(i, 4) for i in p0_kraus]
    p0_kraus_zz = cz_phase_correct(p0_kraus)
    p0_super_2 = qt.kraus_to_super(p0_kraus_zz)
    f_noise = qt.metrics.average_gate_fidelity(p0_super_2, target=cz_gate())

    return np.log10(1-f_noise)



def cz_fidelity_optimize(arg, *args):
    [tg, drive_amp, detune] = arg # Independent arguments that can be optimized over
    [n_cpu, hspace, W_20_50, H_drive, logic_idx] = args # System arguments
    arg_all = [tg, drive_amp, detune, n_cpu, hspace, W_20_50, H_drive, logic_idx]
    return cz_fidelity(arg_all)


def cz_fidelity(arg_all):
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


def zero_pi_initialize(drive_phi, drive_theta, truncation=10, ncut=60, phi_cut=200):
    EL        = 0.377 # GHz
    EJ        = 6.013 # Soft Zero Pi (Gyenis)
    EC_phi    = 1.142
    EC_theta  = 0.092
    E_CJ = 2 * EC_phi
    E_C = 2./(1./EC_theta -1./EC_phi)
    phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, phi_cut)
    zero_pi = scq.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=E_CJ, EC = E_C, dEJ=0.,
                            ng=0., flux=0., ncut=ncut, truncated_dim=truncation)
    # print('zero_pi.ncut=',zero_pi.ncut, ', zero_pi.grid.pt_count=',zero_pi.grid.pt_count )
    n_Theta = zero_pi.matrixelement_table(operator='n_theta_operator', evals_count=truncation)
    n_Phi = zero_pi.matrixelement_table(operator='i_d_dphi_operator', evals_count=truncation)
    n_phi = 2*np.pi * qt.Qobj(n_Phi)
    n_theta = 2*np.pi * qt.Qobj(n_Theta)

    evals = 2*np.pi * zero_pi.eigenvals(evals_count=truncation)
    evals = evals - evals[0]
    H0 = qt.Qobj(np.diag(evals))

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
        drive_term = 0.976*n_phi+ 0.024*n_theta

    ## find hilbert space
    thresh = 0.01
    hspace_charge = [0, 2]
    for s in hspace_charge:
        for i in range(truncation):
            if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()

    return H0, drive_term, w_trans_1, w_trans_2, hspace_charge


# def zero_pi_initialize_peter(drive_phi, drive_theta, truncation=10):
#     EL   = 0.04 # GHz
#     EJ   = 10
#     E_CJ = 20
#     E_C  = 0.04
#     phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, 100)
#     zero_pi = scq.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=E_CJ, EC = E_C, dEJ=0.,
#                             ng=0., flux=0., ncut=30, truncated_dim=truncation)
#     n_Theta = zero_pi.matrixelement_table(operator='n_theta_operator', evals_count=truncation)
#     n_Phi = zero_pi.matrixelement_table(operator='i_d_dphi_operator', evals_count=truncation)
#     n_phi = 2*np.pi * qt.Qobj(n_Phi)
#     n_theta = 2*np.pi * qt.Qobj(n_Theta)

#     evals = 2*np.pi * zero_pi.eigenvals(evals_count=truncation)
#     evals = evals - evals[0]
#     H0 = qt.Qobj(np.diag(evals))

#     if drive_theta:
#         w_trans_1 = evals[10] - evals[0]
#         w_trans_2 = evals[10] - evals[1]
#         drive_term = n_theta

#     ## find hilbert space
#     thresh = 0.01
#     hspace_charge = [0, 1]
#     for s in hspace_charge:
#         for i in range(truncation):
#             if np.abs(drive_term[s, i]/(2*np.pi)) > thresh and i not in hspace_charge:
#                 hspace_charge.append(i)
#     hspace_charge.sort()

#     return H0, drive_term, w_trans_1, w_trans_2, hspace_charge



def _parallel_mesolve_fast(n, N, H, tlist, c_op_list, args, options, proj_idx, dims=None):
    row_idx, col_idx = proj_idx[n]
    rho0 = qt.states.projection(N, row_idx, col_idx)
    rho0.dims = dims
    output = qt.mesolve(
        H, rho0, tlist, c_ops=c_op_list, args=args, options=options,
        _safe_mode=False)
    return output

def get_propagator_noise_fast(H, tlist, num_cpus, parallel, c_op_list, args, options, logi_state):
    dimz = len(logi_state)
    proj_idx = [(logi_state[i],logi_state[j]) for j in range(dimz)
                for i in range(dimz)]
    if isinstance(H, list):
        H0 = H[0][0] if isinstance(H[0], list) else H[0]
    else:
        H0 = H
    N = H0.shape[0]
    u = np.zeros([N * N, dimz * dimz, len(tlist)], dtype=complex)
    if parallel:
        output = qt.parallel.parallel_map(_parallel_mesolve_fast, range(dimz * dimz),
                                task_args=(
                                    N, H, tlist, c_op_list, args, options, proj_idx),
                                task_kwargs={"dims": H0.dims},
                                num_cpus=num_cpus)
        for n in range(dimz * dimz):
            for k, t in enumerate(tlist):
                u[:, n, k] = qt.superoperator.mat2vec(output[n].states[k].full()).T
    else:
        for n, idx in enumerate(proj_idx):
            row_idx, col_idx = idx
            rho0 = qt.states.projection(N, row_idx, col_idx)
            rho0.dims = H0.dims
            output = qt.mesolve(
                H, rho0, tlist, c_ops=c_op_list, args=args,
                options=options, _safe_mode=False)
            for k, t in enumerate(tlist):
                u[:, n, k] = qt.superoperator.mat2vec(output.states[k].full()).T
    out = np.empty((len(tlist),), dtype=object)
    out[:] = [qt.Qobj(u[:, :, k], dims=[[[N], [N]], [[dimz], [dimz]]]) for k in range(len(tlist))]
    return out[-1]


def get_fidelity(s_op, keep_levels):
    p0_kraus = qt.to_kraus(qt.to_super(s_op))
    p0_kraus = [truncate_2(i, keep_levels) for i in p0_kraus]
    p0_super_2 = qt.kraus_to_super(p0_kraus)
    f_noise = qt.metrics.average_gate_fidelity(p0_super_2, target=qt.sigmax())
    return f_noise


def xgate_fidelity_noise_fast(args_indep, *args):
    [H_qbt_drive, w_trans_1, w_trans_2, n_cpu, c_op_list, logi_state, parallel] = args
    [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = args_indep
    pulse_args = {'drive_amp_A': drive_amp_A ,
            'drive_freq_A': w_trans_1 + 2*np.pi*detune_A,
            'drive_amp_B': drive_amp_B ,
            'drive_freq_B': w_trans_2 + 2*np.pi*detune_B,
            'gate_time': tg,
                'alpha_A': 0,
                'alpha_B': 0 }
    tlist = np.linspace(0, tg,  num=3*int(tg))  # total time
    options =qt.Options(max_step=1e-4, nsteps=1e4, num_cpus=n_cpu )
    p_simple_2_a = get_propagator_noise_fast(H_qbt_drive, tlist, n_cpu, parallel, c_op_list,
                                   args=pulse_args, options=options, logi_state=logi_state)
    f_noise = get_fidelity(p_simple_2_a, logi_state)
    return np.log10(1-f_noise)


def is_sparse(matrix):
    non_zero = np.count_nonzero(matrix)  # 非零元素数量
    total_elements = matrix.size  # 总元素数量
    sparsity = 1 - (non_zero / total_elements)  # 稀疏度
    return sparsity  # 如果稀疏度 > 0.7，认为是稀疏矩阵
# 判断是否为稀疏矩阵
# print(is_sparse(np.round(Hint, 8)))  # 输出: True

# def xgate_fidelity(argz):
    # [H0, drive_term, w_trans_1, w_trans_2, hilbert_space, n_cpu, tg,
    # drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = argz

    # states = [qt.basis(len(hilbert_space), i) for i in range(len(hilbert_space))]
    # H0_truc = truncate_2(H0, hilbert_space)
    # drive_truc = truncate_2(drive_term, hilbert_space)
    # H_qbt_drive = [H0_truc, [drive_truc, drag_A],
    #                         [drive_truc, drag_B],]

    # pulse_args = {'drive_amp_A': drive_amp_A ,
    #         'drive_freq_A': w_trans_1 + 2*np.pi*detune_A,
    #         'drive_amp_B': drive_amp_B ,
    #         'drive_freq_B': w_trans_2 + 2*np.pi*detune_B,
    #         'gate_time': tg,
    #          'alpha_A': alpha_A,
    #          'alpha_B': alpha_B }
    # tlist = np.linspace(0, tg,  num=3* int(np.max([tg, len(hilbert_space) ]) ) )  # total time
    # if n_cpu==1:
    #     prop = qt.propagator( H=H_qbt_drive,
    #                         t=tlist,
    #                         args=pulse_args,
    #                         )[-1]  # get the propagator at the final time step

    #     # num=100* int(np.max([tg, len(hilbert_space) ]))
    #     # options =qt.Options( nsteps=100*num)
    #     # prop = qt.propagator( H=H_qbt_drive,
    #     #                         t=tg,
    #     #                         args=pulse_args,
    #     #                         options=options,
    #     #                         # num_cpus=n_cpu,
    #     #                         # parallel=True,
    #     #                         )  # get the propagator at the final time step
    # else:
    #     options =qt.Options( num_cpus=1 )
    #     prop = qt.propagator( H=H_qbt_drive,
    #                         t=tlist,
    #                         args=pulse_args,
    #                         options=options,
    #                         num_cpus=n_cpu,
    #                         parallel=True,
    #                         )[-1]  # get the propagator at the final time step
    # index_2 = hilbert_space.index(2)
    # state_logi = [states[0], states[index_2]]
    # Uc = qt.Qobj([ [prop.matrix_element(s1, s2) for s1 in state_logi]
    #         for s2 in state_logi  ])
    # fidelity = qt.average_gate_fidelity(Uc, target=qt.sigmax())
    # return np.log10(1-fidelity)



    # def get_fidelity_x(arg_de, *args):
#     drive_amp_A, drive_amp_B, detune_A, detune_B = arg_de
#     (tg, w_trans_1, w_trans_2, states, H_qbt_drive) = args

#     pulse_args = {'drive_amp_A': drive_amp_A ,
#             'drive_freq_A': w_trans_1 + 2*np.pi*detune_A,
#             'drive_amp_B': drive_amp_B ,
#             'drive_freq_B': w_trans_2 + 2*np.pi*detune_B,
#             'gate_time': tg }
#     tlist = np.linspace(0, tg, num=2*int(np.max([tg, len(states) ]) ) )  # total time
#     prop = qt.propagator( H=H_qbt_drive,
#                         t=tlist,
#                         args=pulse_args,
#                         )[-1]  # get the propagator at the final time step
#     state_logi = [states[0], states[2]]
#     Uc = qt.Qobj([ [prop.matrix_element(s1, s2) for s1 in state_logi]
#             for s2 in state_logi  ])
#     fidelity = qt.average_gate_fidelity(Uc, target=qt.sigmax())
#     return np.log10(1-fidelity)

# def get_fidelity_given_hspace(hilbert_space, argz):
#     [H0, drive_term, tg, w_trans_1, w_trans_2, drive_amp_A, drive_amp_B, detune_A, detune_B] = argz
#     states = [qt.basis(len(hilbert_space), i) for i in range(len(hilbert_space))]
#     H0_truc = truncate_2(H0, hilbert_space)
#     drive_truc = truncate_2(drive_term, hilbert_space)
#     H_qbt_drive = [H0_truc, [drive_truc, drive_gauss_A],
#                             [drive_truc, drive_gauss_B],]
#     args = [tg, w_trans_1, w_trans_2, states, H_qbt_drive]
#     arg_de = [drive_amp_A, drive_amp_B, detune_A, detune_B]
#     fidelity = get_fidelity_x(arg_de, *args)
#     return fidelity