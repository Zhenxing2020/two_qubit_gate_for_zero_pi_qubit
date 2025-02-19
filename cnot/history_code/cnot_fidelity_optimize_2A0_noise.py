import sys
sys.path.append('../')

import qutip as qt
import numpy as np
from qutip.qip.operations import rz, cz_gate
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
from joblib import Parallel, delayed
import scipy.sparse as ssp
from sympy import symbols
import scipy as sp
import time
import utils_2Q_gate_zp as ut
import os
from datetime import datetime
import pytz

def fidelity_noise_sweep(param, *args):
    drive_amp_A, drive_amp_B, tg, detune_A, detune_B, _ = param
    (max_steps, w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14,
        n_theta1_20_14, n_theta2_20_14, state_tot, H_qbt_drive, n1n1, num_cpus) = args

    pulse_args = {'drive_amp_A': drive_amp_A* (n_theta1_20_14/n_theta1_00_14) ,
                'drive_freq_A': w_00_14 + 2*np.pi*detune_A,
                'drive_amp_B': drive_amp_B ,
                'drive_freq_B': w_20_14 + 2*np.pi*detune_B,
                'gate_time': tg }
    tlist = np.linspace(0, tg, num=3*int(tg))  # total time
    options =qt.Options(max_step=max_steps, nsteps=1e4, num_cpus=1 )
    p = qt.propagator( H=H_qbt_drive,
                            t=tlist,
                            options=options,
                            args=pulse_args,
                            num_cpus=num_cpus,
                            parallel=True,
                            )[-1]  # get the propagator at the final time step
    f_zl = ut.cnot_fidelity( p, state_tot )

    p_super = qt.propagator(H=H_qbt_drive,
                        t=tlist,
                        args=pulse_args,
                        c_op_list=jump_t1 + jump_tphi,
                        options=options,
                        num_cpus=num_cpus,
                        parallel=True,
                        )[-1]  # get the propagator at the final time step

    p0_kraus = qt.to_kraus(qt.to_super(p))
    p0_kraus = [ut.truncate(i, 4) for i in p0_kraus]
    p0_kraus_zz = ut.cnot_phase_correct(p0_kraus)
    p0_super_2 = qt.kraus_to_super(p0_kraus_zz)
    f_ideal = qt.metrics.average_gate_fidelity(p0_super_2, target=U_cnot1)

    p1_kraus = qt.to_kraus(p_super)
    p1_kraus = [ut.truncate(i, 4) for i in p1_kraus]
    p1_kraus_zz = ut.cnot_phase_correct(p1_kraus)
    p1_super_2 = qt.kraus_to_super(p1_kraus_zz)
    f_noise = qt.metrics.average_gate_fidelity(p1_super_2, target=U_cnot1)

    return f_zl, f_ideal, f_noise


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))
    n_cpu = 3
    n_job = 15
    max_steps = 1e-4
    gamma_2 =  1 / 1600e3
    gamma_p2 = 1 / 100e3
    ratio = 10
    gamma_3 =  gamma_2 * ratio
    gamma_p3 = gamma_p2 * ratio

    args_all = ut.get_operator_two_zeropi()
    [n_theta1, n_theta2, n_theta1_dress, n_theta2_dress, n_theta1_truc, n_theta2_truc,
    eval_tot, order_sort, H0, trunc_states, eket0, eket1, eket_tot ] = args_all
    idxs = [order_sort.index(i) for i in trunc_states]

    trunc_dim = eket1.shape[0]
    truc = len(trunc_states)
    eket_truc = [eket_tot[i] for i in idxs]
    eket_truc = np.reshape(eket_truc, (truc, trunc_dim**2))
    jump_t1   = []
    jump_tphi = []
    gamma_t1   = [0, gamma_3,  gamma_2]  + [gamma_3]  * (trunc_dim-3)
    gamma_tphi = [0, gamma_p3, gamma_p2] + [gamma_p3] * (trunc_dim-3)
    for i in range(1,trunc_dim):
        ladder_0i = qt.basis(trunc_dim,0) * qt.basis(trunc_dim,i).dag()
        a_0i_I = qt.tensor(ladder_0i, qt.qeye(trunc_dim))
        a_I_0i = qt.tensor(qt.qeye(trunc_dim), ladder_0i)
        jump_t1.append(qt.Qobj( eket_truc @ ( np.sqrt(gamma_t1[i])* a_0i_I ).data @ eket_truc.conj().T ))
        jump_t1.append(qt.Qobj( eket_truc @ ( np.sqrt(gamma_t1[i])* a_I_0i ).data @ eket_truc.conj().T ))

        # t_phi
        proj_ii = qt.basis(trunc_dim,i).proj()
        a_ii_I = qt.tensor(proj_ii, qt.qeye(trunc_dim))
        a_I_ii = qt.tensor(qt.qeye(trunc_dim), proj_ii)
        jump_tphi.append(qt.Qobj( eket_truc @ ( np.sqrt(2*gamma_tphi[i])* a_ii_I  ).data @ eket_truc.conj().T) )
        jump_tphi.append(qt.Qobj( eket_truc @ ( np.sqrt(2*gamma_tphi[i])* a_I_ii  ).data @ eket_truc.conj().T) )


    state_tot = [qt.basis(truc, i) for i in range(truc)]
    [w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14, n_theta1_20_14,
    n_theta2_20_14] = ut.get_transition_freq_cnot(args_all)
    n1n1 = True
    if n1n1:
        H_qbt_drive = [ H0, [2*np.pi*n_theta1_truc, ut.drive_gauss_A],
                            [2*np.pi*n_theta1_truc, ut.drive_gauss_B]  ]
    else:
        H_qbt_drive = [ H0, [2*np.pi*n_theta2_truc, ut.drive_gauss_A],
                            [2*np.pi*n_theta1_truc, ut.drive_gauss_B]  ]
    U_cnot1 =   qt.Qobj([   [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]])


    print("n_cpu = ", n_cpu, ";   n_job = ", n_job)
    print("gamma_2 = ", gamma_2, "gamma_p2 = ", gamma_p2,
          ";  gamma_3 / gamma_2 =", ratio,";   max_steps = ", max_steps)
    if gamma_2 != 0:
        print(f"T1_2 = {1/gamma_2} ns")
    if gamma_p2 != 0:
        print(f"Tphi_2 = {1/gamma_p2} ns")

    params = np.array([
[ 2.93782707e-01,  2.07163345e-01,  2.00000000e+01,-2.99853463e-01, -2.81048967e-01, -3.27570812e-01],
[ 1.99867633e-01,  1.65429125e-01,  2.64300000e+01,-2.99951100e-01, -2.85620118e-01, -3.85077565e-01],
[ 1.02410895e-01,  6.65714640e-02,  3.28571429e+01,-2.46259456e-01, -2.42666790e-01, -4.06779848e-01],
[ 9.84521169e-02,  6.35170012e-02,  3.92871429e+01,-2.45635387e-01, -2.42616658e-01, -5.25405502e-01],
[ 9.28765245e-02,  6.18442825e-02,  4.57142857e+01,-2.45198180e-01, -2.42282881e-01, -6.55342504e-01],
[ 8.50073482e-02,  5.85607676e-02,  5.21442857e+01,-2.41482954e-01, -2.38807023e-01, -7.74712283e-01],
[ 8.19633839e-02,  5.83129845e-02,  5.85714286e+01,-2.42999483e-01, -2.39996871e-01, -8.68408959e-01],
[ 0.09085433,  0.06723939, 65.00142837, -0.26168943, -0.2577288, -0.9844214061746501 ],
[ 8.77105890e-02,  6.68604305e-02,  7.14285714e+01,-2.62456488e-01, -2.58722899e-01, -1.15048185e+00],
[ 8.26896087e-02,  6.48637726e-02,  7.78585714e+01,-2.59804829e-01, -2.55845581e-01, -1.32948344e+00],
[ 8.01041189e-02,  6.29575207e-02,  8.42857143e+01,-2.57971464e-01, -2.54357292e-01, -1.51032781e+00],
[ 7.71941252e-02,  6.14696625e-02,  9.07157143e+01,-2.56441326e-01, -2.52954399e-01, -1.66800102e+00],
[0.075351643, 0.06135683, 97.14285, -0.25725, -0.25353, -1.79079],
[ 7.60520623e-02,  6.39432799e-02,  1.03572857e+02,-2.63253918e-01, -2.59046309e-01, -1.93110218e+00],
[ 7.45778117e-02,  6.34654436e-02,  1.10000000e+02,-2.63462937e-01, -2.59269155e-01, -2.11694638e+00],
[ 7.30722502e-02,  6.20440074e-02,  1.16430000e+02,-2.62115084e-01, -2.58149714e-01, -2.25041164e+00],
[ 7.16774353e-02,  6.10992882e-02,  1.22857143e+02,-2.61474000e-01, -2.57563815e-01, -2.28375390e+00],
[ 1.39375677e-01,  4.12452337e-02,  1.29287143e+02,-2.60842903e-01, -2.79434355e-01, -2.26002033e+00],
[ 1.36380478e-01,  3.94416719e-02,  1.35714286e+02, -2.59402399e-01, -2.76635053e-01, -2.50436],
[ 1.33690914e-01,  3.79693451e-02 , 1.42140000e+02 ,-2.58306995e-01, -2.74236850e-01, -2.68068],
[ 1.33143163e-01,  3.69228390e-02,  1.48571428e+02, -2.58921599e-01, -2.74046847e-01,  -2.73726],
[ 1.40342948e-01,  3.71991110e-02,  1.55001429e+02, -2.66083127e-01, -2.81628325e-01, -2.88338727e+00],
[ 1.39949303e-01,  3.61571071e-02,  1.61428573e+02, -2.66656925e-01, -2.81472098e-01, -3.11159],
[ 1.39496859e-01,  3.55201590e-02,  1.67858570e+02, -2.67606025e-01, -2.81677946e-01,   -3.21012],
[ 1.36599461e-01,  3.43961750e-02,  1.74285714e+02,-2.65980086e-01, -2.79133255e-01, -3.32061902e+00],
[ 1.85754469e-01,  3.80918860e-02,  1.80715714e+02, -2.95110129e-01, -3.28798105e-01,  -3.15373],
[ 1.86997685e-01,  3.74284980e-02,  1.87142856e+02, -2.98005591e-01, -3.30703646e-01,  -3.24536],
[ 1.32295234e-01,  3.22886233e-02,  1.93572855e+02, -2.65458734e-01, -2.76565235e-01, -3.51013],
[ 2.32894448e-01,  4.20098643e-02,  1.99999999e+02, -3.33118430e-01, -3.92974542e-01,  -3.46532]
       ])
    for para in params:
        print(para.tolist(), ',')

    args =  (max_steps, w_00_14, w_20_14, n_theta1_00_14, n_theta2_00_14,
        n_theta1_20_14, n_theta2_20_14, state_tot, H_qbt_drive, n1n1, n_cpu)
    fidelity = Parallel(n_jobs=n_job, verbose=20)(delayed(fidelity_noise_sweep)(param, *args)
                                                for param in params)
    fidelity = np.reshape(fidelity, (len(params), 3))

    # print('f_zl = ',    np.array(fidelity[:,0]).tolist())
    # print('f_ideal = ', np.array(fidelity[:,1]).tolist())
    # print('f_noise = ', np.array(fidelity[:,2]).tolist())

    print('\nf_ideal = [')
    for i in range(0, len(fidelity[:,1]), 4):
        print(', '.join(map(str, fidelity[:,1][i:i+4])), ',')
    print(']')

    print('\nf_noise = [')
    for i in range(0, len(fidelity[:,2]), 4):
        print(', '.join(map(str, fidelity[:,2][i:i+4])), ',')
    print(']')

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))




    # eket0 = eket0.todense()
    # eket1 = eket1.todense()
    # jump_op_t1 = np.zeros((truc, truc), dtype=complex)
    # jump_op_tphi = np.zeros((truc, truc), dtype=complex)
    # for state in trunc_states[1:]:
    #     a0i = eket0[0].conj().T @ eket0[int(state[0])]
    #     a0i = eket0 @ a0i @ eket0.conj().T
    #     b0j = eket1[0].conj().T @ eket1[int(state[1])]
    #     b0j = eket1 @ b0j @ eket1.conj().T
    #     eket_truc = np.reshape([eket_tot[i] for i in idxs], (truc, eket1.shape[0]**2))
    #     jump_op_t1 += (eket_truc @ np.kron(a0i, b0j) @ eket_truc.conj().T)

    #     aii = eket0[int(state[0])].conj().T @ eket0[int(state[0])]
    #     aii = eket0 @ aii @ eket0.conj().T
    #     bjj = eket1[int(state[1])].conj().T @ eket1[int(state[1])]
    #     bjj = eket1 @ bjj @ eket1.conj().T
    #     eket_truc = np.reshape([eket_tot[i] for i in idxs], (truc, eket1.shape[0]**2))
    #     jump_op_tphi += (eket_truc @ np.kron(aii, bjj) @ eket_truc.conj().T)
    # jump_op_t1 = qt.Qobj(jump_op_t1) * np.sqrt(gamma_2)
    # jump_op_tphi = qt.Qobj(jump_op_tphi) * np.sqrt(2*gamma_p2)