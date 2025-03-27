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
import utils_2Q_gate_zp as ut
import scipy.sparse as ssp
# Update scqubits settings
settings.OVERLAP_THRESHOLD = 0.3

def import_para_noise():
    """
    Imports parameters and computes gate fidelities for noisy systems.
    """
    # drive_phi, drive_theta, truc = True, False, 300
    drive_phi, drive_theta, truc = False, True, 150
    drive_0 = True
    folder = 'data_xgate_theta_3ncut.txt' if drive_theta else 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/'+folder)
    params = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
                      ]].to_numpy()#[[-5], :]
    # [[ 1,2, 4,5, 7,8, 10,11, 13,14, 16,17,18],:]  , 0,3,9,12,17,
    #
    num_cpus, n_job = 4, 1*len(params)
    logi_state = [0, 2]

    truncation=1000
    folder = '../../data/3ncut_one_zeropi/'
    if drive_0:
        evals = 2*np.pi* scq.read(folder + f'zeropi_0_specdata_truc={truncation}_3ncut.h5').energy_table
        n_theta = 2*np.pi* scq.read(folder + f'zeropi_0_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_phi = 2*np.pi* scq.read(folder + f'zeropi_0_n_phi_truc={truncation}_3ncut.h5').matrixelem_table
    else:
        evals = 2*np.pi* scq.read(folder + f'zeropi_1_specdata_truc={truncation}_3ncut.h5').energy_table
        n_theta = 2*np.pi* scq.read(folder + f'zeropi_1_n_theta_truc={truncation}_3ncut.h5').matrixelem_table
        n_phi = 2*np.pi* scq.read(folder + f'zeropi_1_n_phi_truc={truncation}_3ncut.h5').matrixelem_table

    evals = evals - evals[0]
    gate_target = qt.sigmax()
    H0 = qt.Qobj(np.diag(evals))
    if drive_phi:
        w_trans_1 = evals[9] - evals[0]
        w_trans_2 = evals[9] - evals[2]
        drive_term = n_phi
    if drive_theta:
        w_trans_1 = evals[7] - evals[0]
        w_trans_2 = evals[7] - evals[2]
        drive_term = n_theta

    hspace_charge = [0, 2]  # Start with the ground and first excited states
    for s in hspace_charge:
        for i in range(truc):
            if np.abs(drive_term[s, i] / (2 * np.pi)) > 0.01 and i not in hspace_charge:
                hspace_charge.append(i)
    hspace_charge.sort()
    # hspace_charge = np.arange(truc).tolist()

    t1_other = 30 # μs

    tphi_logi = 100 # μs
    gamma_decay_logi =  1 / 1600e3
    gamma_dephase_logi = 1 / 1e3 / tphi_logi
    gamma_decay_other =  1 / 1e3 / t1_other
    gamma_dephase_other = 1 / 1e3 / t1_other
    idx_2 = 2 if drive_theta else 1

    hspace_len = len(hspace_charge)
    if drive_theta:
        gamma_decay_old   = [0, gamma_decay_other,  gamma_decay_logi]  + [gamma_decay_other]  * (hspace_len-3)
        gamma_dephase_old = [0, gamma_dephase_other, gamma_dephase_logi] + [gamma_dephase_other] * (hspace_len-3)
    else:
        gamma_decay_old   = [0,  gamma_decay_logi]  + [gamma_decay_other]  * (hspace_len-2)
        gamma_dephase_old = [0,  gamma_dephase_logi] + [gamma_dephase_other] * (hspace_len-2)

    folder_1 = 'data/data_gamma_'
    folder_2 = 'theta.txt' if drive_theta else 'phi.txt'
    gamma_new = pd.read_csv(folder_1 + folder_2)

    ### 't1_50us_47', 't1_50us_27', 't1_50us_07'
    ### 'tphi_50us_02', 'tphi_50us_07', 'tphi_1e6'
    gamma_decay_new = gamma_new['t1_50us_47'].to_numpy()
    gamma_dephase_new = gamma_new['tphi_50us_02'].to_numpy()
    # gamma_dephase_new = gamma_new['tphi_50us_02'].to_numpy()

    print("gamma_decay_new[2] = ", gamma_decay_new[2], ", gamma_dephase_new[2] = ", gamma_dephase_new[2])
    gamma_decay_new = gamma_decay_new *50 /t1_other
    gamma_dephase_new = gamma_dephase_new *50 /t1_other
    # gamma_dephase_new[idx_2] = gamma_dephase_logi
    # gamma_decay_new[idx_2] = gamma_decay_logi

    jump_t1   = []
    jump_tphi = []
    for i in range(1,hspace_len):
        jump_t1.append( np.sqrt(gamma_decay_new[i]) * qt.basis(hspace_len,0) * qt.basis(hspace_len,i).dag() )
        jump_tphi.append( np.sqrt(2*gamma_dephase_new[i]) * qt.basis(hspace_len,i).proj() )

    logi_idx = [hspace_charge.index(s) for s in logi_state]
    H0_truc = ut.truncate_2(H0, hspace_charge)
    drive_truc = ut.truncate_2(drive_term, hspace_charge)
    H_qbt_drive = [H0_truc, [drive_truc, ut.drive_gauss_A],
                            [drive_truc, ut.drive_gauss_B],]
    print('drive_phi=', drive_phi, '; drive_theta = ', drive_theta, '; truc = ', truc)
    print('hspace_len=', hspace_len)
    print('params =')
    for para in params:
        print(para.tolist(), ',')
    print("num_cpus = ", num_cpus, ";   n_job = ", n_job)
    print("gamma_decay_logi = ", gamma_decay_logi, ", gamma_dephase_logi = ", gamma_dephase_logi)
    print("gamma_decay_other = ", gamma_decay_other, ", gamma_dephase_other = ", gamma_dephase_other)
    print(f"T1_logi = {1/gamma_decay_logi} ns") if gamma_decay_logi != 0 else None
    print(f"Tphi_logi = {1/gamma_dephase_logi} ns") if gamma_dephase_logi != 0 else None
    print(f"T1_other = {1/gamma_decay_other} ns") if gamma_decay_other != 0 else None
    print(f"Tphi_other = {1/gamma_dephase_other} ns") if gamma_dephase_other != 0 else None

    ############################################################
    # c_op_list = [qt.Qobj(np.zeros((truc, truc)))]
    # c_op_list = []
    # args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx, gate_target]
    # f_ideal = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args)
    #                                             for args_indep in params)
    # print('\nf_ideal = [')
    # for i in range(0, len(f_ideal), 4):
    #     print(', '.join(map(str, f_ideal[i:i+4])), ',')
    # print(']')
    # print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    ############################################################
    c_op_list = jump_t1 + jump_tphi
    args = [H_qbt_drive, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx, gate_target]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_noise = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, f_noise[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))






def import_select_phi():
    folder = 'data_xgate_phi_3ncut.txt'
    f_xgate = pd.read_csv('data/'+folder)
    x0_vec = f_xgate[['tg', 'drive_amp_1', 'drive_amp_2', 'detune_1', 'detune_2'
                      ]].to_numpy()[ 9: ,:]
    # [[-1], :]   [ 0::5 ,:]   [ [8, 15,16,17,18 ] ,:]   [ [6,7,8, 13 ] ,:]
    folder = '../../data/3ncut_one_zeropi/'
    evals = 2*np.pi* scq.read(folder + f'zeropi_0_specdata_truc=1000_3ncut.h5').energy_table
    n_phi = 2*np.pi* scq.read(folder + f'zeropi_0_n_phi_truc=1000_3ncut.h5').matrixelem_table
    evals = evals - evals[0]
    w_trans_1 = evals[9] - evals[0]
    w_trans_2 = evals[9] - evals[2]
    drive_term = n_phi

    truc = 500
    ### hspace_select
    # hspace_select = np.arange(truc).tolist()
    hspace_select =    [
#         ### charge_300
# 0, 2, 3, 4, 8, 9, 10, 11, 15, 16 ,
# 18, 19, 20, 23, 24, 25, 28, 31, 33, 35 ,
# 37, 39, 40, 42, 43, 46, 47, 49, 51, 53 ,
# 55, 56, 57, 60, 62, 64, 66, 67, 69, 70 ,
# 73, 76, 77, 79, 81, 83, 85, 86, 88, 91 ,

# 92, 95, 96, 98, 100, 101, 102, 104, 106, 108 ,
# 110, 112, 113, 116, 118, 120, 121, 123, 126, 128 ,
# 130, 132, 133, 136, 137, 139, 141, 143, 145, 146 ,
# 148, 151, 153, 154, 156, 158, 159, 162, 164, 166 ,
# 167, 169, 170, 173, 174, 176, 179, 181, 183, 185 ,

# 186, 188, 190, 192, 194, 196, 198, 200, 202, 205 ,
# 206, 208, 209, 212, 213, 216, 218, 220, 222, 223 ,
# 226, 227, 229, 230, 232, 234, 237, 239, 241, 242 ,
# 245, 246, 248, 250, 252, 253, 255, 257, 259, 262 ,
# 264, 265, 267, 269, 271, 273, 275, 277, 279, 281 ,
# 282, 283, 286, 288, 290, 291, 293, 294, 297, 299 ,

### all_500
0, 9, 2, 33, 10, 19, 25, 37, 31, 23 ,
16, 69, 18, 40, 46, 11, 15, 28, 57, 47 ,
42, 20, 62, 64, 55, 70, 53, 39, 49, 35 ,
51, 73, 56, 83, 24, 77, 76, 86, 66, 43 ,
60, 95, 92, 98, 79, 91, 101, 100, 85, 106 ,

3, 81, 108, 67, 116, 154, 112, 130, 88, 104 ,
121, 133, 96, 110, 113, 158, 146, 143, 123, 132 ,
139, 153, 102, 186, 120, 118, 166, 136, 162, 206 ,
176, 151, 145, 170, 194, 192, 126, 148, 232, 209 ,
141, 196, 220, 128, 253, 137, 159, 164, 173, 188 ,

185, 248, 218, 205, 167, 241, 229, 213, 281, 198 ,
156, 246, 181, 179, 174, 230, 169, 200, 226, 202 ,
265, 305, 190, 264, 259, 255, 223, 216, 250, 306 ,
282, 279, 262, 183, 297, 212, 237, 269, 257, 275 ,
273, 234, 322, 252, 291, 267, 319, 328, 288, 349 , # yes

227, 294, 336, 242, 239, 208, 315, 286, 320, 333 ,
358, 363, 370, 348, 384, 8, 397, 222, 366, 303 ,
# 337, 302, 418, 414, 357, 398, 326, 271, 310, 451 ,  #### no
438, 352, 346, 277, 245, 474, 283, 445, 313, 407 , # yes
# 391, 299, 290, 372, 341, 382, 409, 293, 361, 343 ,  # no
    ]

    ### common part
    logi_state = [0, 2]
    hspace_full = np.arange(truc).tolist()
    logi_idx_select = [hspace_select.index(i) for i in logi_state]
    index_select = [hspace_full.index(i) for i in hspace_select]
    len_select = len(hspace_select)
    H0_full = qt.Qobj(np.diag(evals))
    H0_select = ut.truncate_2( H0_full, index_select)
    drive_select = ut.truncate_2(drive_term, index_select)
    H_drive_select = [H0_select, [drive_select, ut.drive_gauss_A],
                                 [drive_select, ut.drive_gauss_B],]

    num_cpus, n_job = 4, 100
    print('truc = ', truc)
    print("num_cpus = ", num_cpus, ";   n_job = ", n_job)
    print('params =')
    for para in x0_vec:
        print(para.tolist(), ',')

    states_all_index = [hspace_full.index(i) for i in hspace_select]
    data = states_all_index
    print(f'\nhspace_select_index (len={len_select}) = [')
    for i in range(0, len(data), 10):  # Step size of 10
        print(", ".join(f"{x}" for x in data[i:i + 10]), ',')
    print(']')

    ############################################################
    # c_op_list = [qt.Qobj(np.zeros((truc, truc)))]
    c_op_list = []
    args = [H_drive_select, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx_select]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in x0_vec)
    print('\nf_ideal = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, f_ideal[i:i+4])), ',')
    print(']')

    ###################################################################
    ### get collapse operators
    t1_tphi_other = 170 # μs

    print(f"t1_tphi_other = {t1_tphi_other} μs", )
    hspace_charge_dim = len(hspace_select)
    gamma_new = pd.read_csv('data/data_gamma_phi_truc200.txt')
    gamma_decay_new = gamma_new['t1_50us_49'].to_numpy()
    gamma_dephase_new = gamma_new['tphi_50us_02'].to_numpy()
    gamma_decay_new = gamma_decay_new *50 /t1_tphi_other
    gamma_dephase_new = gamma_dephase_new *50 /t1_tphi_other
    jump_t1   = []
    jump_tphi = []
    for i in range(1,hspace_charge_dim):
        jump_t1.append( np.sqrt(gamma_decay_new[i]) * qt.basis(hspace_charge_dim,0) * qt.basis(hspace_charge_dim,i).dag() )
        jump_tphi.append( np.sqrt(2*gamma_dephase_new[i]) * qt.basis(hspace_charge_dim,i).proj() )

    c_op_list = jump_t1 + jump_tphi
    args = [H_drive_select, w_trans_1, w_trans_2, num_cpus, c_op_list, logi_idx_select]
    f_noise = Parallel(n_jobs=n_job)(delayed(ut.xgate_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in x0_vec)
    print('\nf_noise = [')
    for i in range(0, len(f_noise), 4):
        print(', '.join(map(str, f_noise[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


if __name__ == '__main__':
    print(os.path.basename(__file__))  # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # import_para()
    # import_para_noise()
    import_select_phi()



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
    n_phi = scq.read(folder + f'zeropi_n_phi_truc={truncation}_3ncut.h5')
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