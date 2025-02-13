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
from multiprocessing import Pool
import scqubits as scq
from sympy import symbols
from joblib import Parallel, delayed
import scipy.sparse as ssp


def generate_eval_tot_3ncut():
    truc_import = 500
    truc1, truc_tot, charge_pick = 300, 500, False

    folder = f'../../data/3ncut_two_zeropi/truc1={truc_import}/'
    eval0 = pd.read_csv(folder+ 'eval0.txt').to_numpy().flatten()
    eval1 = pd.read_csv(folder+ 'eval1.txt').to_numpy().flatten()
    n_theta0 = pd.read_csv(folder+ 'n_theta0.txt').map(complex).to_numpy()
    n_theta1 = pd.read_csv(folder+ 'n_theta1.txt').map(complex).to_numpy()

    eval0 = eval0[:truc1]
    eval1 = eval1[:truc1]
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    n_theta0 = qt.Qobj(n_theta0[np.ix_(hspace_0, hspace_0)])
    n_theta1 = qt.Qobj(n_theta1[np.ix_(hspace_1, hspace_1)])

    ##############################################################################################
    zp = scq.Circuit(ut.zp_yml, from_file=False)
    zp.configure(transformation_matrix=np.linalg.inv(ut.transform_2zeropi))
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [100, 100]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims
                )
    zp.cutoff_ext_1, zp.cutoff_ext_5 = 300, 300
    zp.cutoff_n_2, zp.cutoff_n_6 = 90, 90

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
        n_theta0 = ut.truncate_2(n_theta0, hspace_0)
        n_theta1 = ut.truncate_2(n_theta1, hspace_1)
        eval0 = eval0[hspace_0]
        eval1 = eval1[hspace_1]

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

    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_charge={charge_pick}/'
    os.mkdir(folder_save)
    pd.DataFrame(eval_tot).to_csv(folder_save+ 'eval_tot.txt', sep=',', index=False, header=True)
    pd.DataFrame(eket_tot.toarray()).to_csv(folder_save+ 'eket_tot.txt', sep=',', index=False, header=True)


def generate_data_3ncut():
    truc_import = 500
    truc1, truc_tot, charge_pick = 300, 1000, False
    # truc1, truc_tot, charge_pick = 10, 11, False

    folder = f'../../data/3ncut_two_zeropi/truc1={truc_import}/'
    eval0 = pd.read_csv(folder+ 'eval0.txt').to_numpy().flatten()
    eval1 = pd.read_csv(folder+ 'eval1.txt').to_numpy().flatten()
    n_theta0 = pd.read_csv(folder+ 'n_theta0.txt').map(complex).to_numpy()
    n_theta1 = pd.read_csv(folder+ 'n_theta1.txt').map(complex).to_numpy()

    eval0 = eval0[:truc1]
    eval1 = eval1[:truc1]
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    n_theta0 = qt.Qobj(n_theta0[np.ix_(hspace_0, hspace_0)])
    n_theta1 = qt.Qobj(n_theta1[np.ix_(hspace_1, hspace_1)])

    ##############################################################################################
    zp = scq.Circuit(ut.zp_yml, from_file=False)
    zp.configure(transformation_matrix=np.linalg.inv(ut.transform_2zeropi))
    system_hierarchy = [[1,2],  [5,6]]
    subsystem_trunc_dims = [100, 100]
    zp.configure(system_hierarchy=system_hierarchy,
                subsystem_trunc_dims=subsystem_trunc_dims
                )
    zp.cutoff_ext_1, zp.cutoff_ext_5 = 300, 300
    zp.cutoff_n_2, zp.cutoff_n_6 = 90, 90
    # zp.cutoff_ext_1, zp.cutoff_ext_5 = 50, 50
    # zp.cutoff_n_2, zp.cutoff_n_6 = 20, 20
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
        n_theta0 = ut.truncate_2(n_theta0, hspace_0)
        n_theta1 = ut.truncate_2(n_theta1, hspace_1)
        eval0 = eval0[hspace_0]
        eval1 = eval1[hspace_1]

    ##############################################################################################
    ###  Compute eigenvalues and eigenvectors for coupling H
    n2, n6 = symbols('n2 n6')
    g = float(zp.sym_interaction((1,0), return_expr=True).coeff(n2*n6) )
    Hint = qt.tensor(qt.Qobj(n_theta0) , qt.Qobj(n_theta1))
    H_bare = (  qt.tensor(qt.Qobj(np.diag(eval0)),  qt.identity(len(hspace_1)))
            +  qt.tensor(qt.identity(len(hspace_0)),  qt.Qobj(np.diag(eval1))) )
    # Htot = (g* Hint + H_bare).tidyup(atol=1e-8)
    Htot = g* Hint + H_bare
    print("Htot = g* Hint + H_bare.... Time:", datetime.now(pytz.timezone('America/Denver')))

    ### the one-line code below takes time when truc1 is large
    k = Htot.shape[0] - 1
    if truc_tot != None:
        k = truc_tot
    eval_tot, eket_tot = ssp.linalg.eigsh(Htot.data, k=k, which='SA', tol=1.e-10)
    print("eval_tot, eket_tot..... Time:", datetime.now(pytz.timezone('America/Denver')))

    sorted_idx_tot = np.argsort(eval_tot)
    eval_tot = eval_tot[sorted_idx_tot]
    eval_tot = eval_tot - eval_tot[0]
    eket_tot = ssp.csr_matrix([eket_tot[:,idx] for idx in sorted_idx_tot])

    print("folder_save =..... Time:", datetime.now(pytz.timezone('America/Denver')))
    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    os.mkdir(folder_save)
    pd.DataFrame(eval_tot).to_csv(folder_save+ 'eval_tot.txt', sep=',', index=False, header=True)
    pd.DataFrame(eket_tot.toarray()).to_csv(folder_save+ 'eket_tot.txt', sep=',', index=False, header=True)

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
    pd.DataFrame(top_index).to_csv(folder_save+ 'top_index.txt', sep=',', index=False, header=True)
    pd.DataFrame(top_overlap).to_csv(folder_save+ 'top_overlap.txt', sep=',', index=False, header=True)
    print("pd.DataFrame(top_overlap)..... Time:", datetime.now(pytz.timezone('America/Denver')))

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
    n_theta1_dress = ssp.kron(ssp.identity(len(hspace_0)), n_theta1)
    n_theta0_dress = (eket_tot @ n_theta0_dress @ eket_tot.conj().T).todense()
    n_theta1_dress = (eket_tot @ n_theta1_dress @ eket_tot.conj().T).todense()
    print("folder_save.... Time:", datetime.now(pytz.timezone('America/Denver')))

    pd.DataFrame(n_theta0_dress).to_csv(folder_save+ 'n_theta0_dress.txt', sep=',', index=False, header=True)
    pd.DataFrame(n_theta1_dress).to_csv(folder_save+ 'n_theta1_dress.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_full).to_csv(folder_save+ 'hspace_full.txt', sep=',', index=False, header=True)


def generate_data():
    truc1, truc_tot, charge_pick = 300, 300, True
    n_cut, phi_cut = 90, 300
    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    args_all = ut.get_operator_two_zeropi_v2(truc1=truc1, truc_tot=truc_tot, charge_pick=charge_pick,
                                             n_cut=n_cut, phi_cut=phi_cut)
    [hspace_0, hspace_1, eval_tot, eket_tot] = args_all

    folder = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    os.mkdir(folder)
    pd.DataFrame(eval_tot).to_csv(folder+ 'eval_tot.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_0).to_csv(folder+ 'hspace_0.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_1).to_csv(folder+ 'hspace_1.txt', sep=',', index=False, header=True)
    pd.DataFrame(eket_tot.toarray()).to_csv(folder+ 'eket_tot.txt', sep=',', index=False, header=True)


def cz_fidelity_pool(args):
    [tg, _, _, _, _, _, _, _] = args
    return tg, ut.cz_fidelity(args)

def import_para():
    cz300_se_3ncut= pd.read_csv('data/data_cz_1ncut_truc1=80.txt')
    params = cz300_se_3ncut[['tg', 'drive_amp', 'detune']].to_numpy()#[[0, -1],:]#[[0, 5, 10, 15, 20, 25],:]  #

    truc1, truc_tot, charge_pick = 300, 1000, True
    truc_tot_2 = 1000
    num_cpus, n_job = 1, len(params)
    folder = f'../../data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    n_theta0_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
    n_theta1_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta1_dress.txt').to_numpy()

    eval_tot = eval_tot[:truc_tot_2]
    hspace_full = hspace_full[:truc_tot_2]
    hspace_dress = np.arange(truc_tot_2)
    n_theta0_dress = qt.Qobj(n_theta0_dress[np.ix_(hspace_dress, hspace_dress)])
    n_theta1_dress = qt.Qobj(n_theta1_dress[np.ix_(hspace_dress, hspace_dress)])
    print('truc1=', truc1, '; truc_tot = ', truc_tot, '; charge_pick = ', charge_pick)
    print("truc_tot_2 = ", truc_tot_2)
    print("num_cpus = ", num_cpus, ";   n_job = ", n_job)
    for para in params:
        print(para.tolist(), ',')

    logi_state = ['0-0', '0-2', '2-0', '2-2']
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    H0 = qt.Qobj(np.diag(eval_tot))
    logi_idx = [hspace_full.index(state) for state in logi_state]
    H_qbt_drive = [H0, [n_theta1_dress, ut.drive_gauss_A] ]
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    # c_op_list = [qt.Qobj(np.zeros((truc_tot_2, truc_tot_2)))]
    c_op_list = []
    args = [H_qbt_drive, W_20_50, num_cpus, c_op_list, logi_idx ]
    f_ideal = Parallel(n_jobs=n_job)(delayed(ut.cz_fidelity_log_noise)(args_indep, *args)
                                                for args_indep in params)
    print('\nf_ideal = [')
    for i in range(0, len(f_ideal), 4):
        print(', '.join(map(str, f_ideal[i:i+4])), ',')
    print(']')
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))


def generate_data_3ncut_nop():
    truc_import = 500
    truc1, truc_tot, charge_pick = 300, 1000, False
    # truc1, truc_tot, charge_pick = 10, 11, False

    folder = f'../../data/3ncut_two_zeropi/truc1={truc_import}/'
    # eval0 = pd.read_csv(folder+ 'eval0.txt').to_numpy().flatten()
    # eval1 = pd.read_csv(folder+ 'eval1.txt').to_numpy().flatten()
    n_theta0 = pd.read_csv(folder+ 'n_theta0.txt').map(complex).to_numpy()
    n_theta1 = pd.read_csv(folder+ 'n_theta1.txt').map(complex).to_numpy()

    # eval0 = eval0[:truc1]
    # eval1 = eval1[:truc1]
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    n_theta0 = qt.Qobj(n_theta0[np.ix_(hspace_0, hspace_0)])
    n_theta1 = qt.Qobj(n_theta1[np.ix_(hspace_1, hspace_1)])

    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = pd.read_csv(folder_save+ 'eval_tot.txt').to_numpy().flatten()
    eket_tot = ssp.csr_matrix(pd.read_csv(folder_save+ 'eket_tot.txt').map(complex).to_numpy())



    ###  Get wavefunction overlap for the truncated dressed states
    bare_state = [[qt.tensor(qt.basis(truc1, i), qt.basis(truc1, j))
                            for j in range(truc1)]
                                for i in range(truc1)]
    def find_overlap(eket):
        overlaps = np.array([[np.abs( (eket @ bare_state[i][j].data).todense()[0,0] )
                            for j in range(truc1)]
                                for i in range(truc1)])
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
    pd.DataFrame(top_index).to_csv(folder_save+ 'top_index.txt', sep=',', index=False, header=True)
    pd.DataFrame(top_overlap).to_csv(folder_save+ 'top_overlap.txt', sep=',', index=False, header=True)
    print("pd.DataFrame(top_overlap)..... Time:", datetime.now(pytz.timezone('America/Denver')))

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
    hspace_full = [(str(idx[0])+'-'+str(idx[1])) for idx in index_array] # actual state index in each qubit

    n_theta0_dress = ssp.kron(n_theta0, ssp.identity(len(hspace_1)))
    n_theta1_dress = ssp.kron(ssp.identity(truc1), n_theta1)
    n_theta0_dress = (eket_tot @ n_theta0_dress @ eket_tot.conj().T).todense()
    n_theta1_dress = (eket_tot @ n_theta1_dress @ eket_tot.conj().T).todense()
    print("folder_save.... Time:", datetime.now(pytz.timezone('America/Denver')))

    pd.DataFrame(n_theta0_dress).to_csv(folder_save+ 'n_theta0_dress.txt', sep=',', index=False, header=True)
    pd.DataFrame(n_theta1_dress).to_csv(folder_save+ 'n_theta1_dress.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_full).to_csv(folder_save+ 'hspace_full.txt', sep=',', index=False, header=True)


def reduce_eket():
    truc1, truc_tot, charge_pick = 300, 1000, False
    folder_input = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = pd.read_csv(folder_input+ 'eval_tot.txt').to_numpy().flatten()
    eket_tot = pd.read_csv(folder_input+ 'eket_tot.txt').map(complex).to_numpy()

    print("\n eket_tot = pd.read..... Time:", datetime.now(pytz.timezone('America/Denver')))
    truc_output = 500
    eval_tot = eval_tot[:truc_output]
    eket_tot = eket_tot[:truc_output]

    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_output}_pick={charge_pick}/'
    # os.mkdir(folder_save)
    pd.DataFrame(eval_tot).to_csv(folder_save+ 'eval_tot.txt', sep=',', index=False, header=True)
    pd.DataFrame(eket_tot).to_csv(folder_save+ 'eket_tot.txt', sep=',', index=False, header=True)



if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    import_para()
    # generate_data()
    # generate_data_3ncut()
    # generate_data_3ncut_nop()
    # reduce_eket()

    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















