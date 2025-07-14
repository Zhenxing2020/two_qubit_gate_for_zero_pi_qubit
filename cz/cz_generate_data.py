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


def generate_ekettot_3ncut(test=True):
    # zp = scq.Circuit(ut.zp_yml, from_file=False)
    # zp.configure(transformation_matrix=np.linalg.inv(ut.transform_2zeropi))
    # system_hierarchy = [[1,2],  [5,6]]
    # subsystem_trunc_dims = [100, 100]
    # zp.configure(system_hierarchy=system_hierarchy,
    #             subsystem_trunc_dims=subsystem_trunc_dims)    
    # if test:
    #     zp.cutoff_ext_1, zp.cutoff_ext_5 = 50, 50
    #     zp.cutoff_n_2, zp.cutoff_n_6 = 20, 20        
    # else:
    #     zp.cutoff_ext_1, zp.cutoff_ext_5 = 300, 300
    #     zp.cutoff_n_2, zp.cutoff_n_6 = 90, 90
    # n2, n6 = symbols('n2 n6')
    # g = float(zp.sym_interaction((1,0), return_expr=True).coeff(n2*n6) )
    # print(f'zp.cutoff_ext_1, zp.cutoff_n_2 = {zp.cutoff_ext_1}, {zp.cutoff_n_2}')

    if test:
        truc1, truc_tot, charge_pick = 11, 15, True       
    else:
        truc1, truc_tot, charge_pick = 150, 2000, False   
    g=0.030961990356445312
    folder = f'../../data/3ncut_two_zeropi/truc1=500/'
    eval0 = pd.read_csv(folder+ 'eval0.txt').to_numpy().flatten()
    eval1 = pd.read_csv(folder+ 'eval1.txt').to_numpy().flatten()
    n_theta0 = np.load(folder+'n_theta0.npy')
    n_theta1 = np.load(folder+'n_theta1.npy')

    eval0 = eval0[:truc1]
    eval1 = eval1[:truc1]
    hspace_0 = np.arange(truc1)
    hspace_1 = np.arange(truc1)
    n_theta0 = ut.truncate_2(n_theta0, hspace_0)
    n_theta1 = ut.truncate_2(n_theta1, hspace_1)

    ##############################################################################################
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
    print(f'len(hspace_0)={len(hspace_0)}, len(hspace_1)={len(hspace_1)}')
    print(f'truc1, truc_tot, charge_pick = {truc1}, {truc_tot}, {charge_pick}')
    print(f'g={g}')
    Hint = qt.tensor(qt.Qobj(n_theta0) , qt.Qobj(n_theta1))
    H_bare = (  qt.tensor(qt.Qobj(np.diag(eval0)),  qt.identity(len(hspace_1)))
            +  qt.tensor(qt.identity(len(hspace_0)),  qt.Qobj(np.diag(eval1))) )
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

    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    os.mkdir(folder_save)
    pd.DataFrame(eval_tot).to_csv(folder_save+ 'eval_tot.txt', sep=',', index=False, header=True)
    np.save(folder_save+'eket_tot.npy', eket_tot.toarray())
    print(f'np.shape(eket_tot)={np.shape(eket_tot)}')


def generate_nop_3ncut():
    truc1, truc_tot, charge_pick = 150, 1000, True
    # truc1, truc_tot, charge_pick = 11, 15, False

    if charge_pick:
        folder = f'../../data/3ncut_two_zeropi/truc1=150_truc2=1000_pick=True/'
        hspace_0 = pd.read_csv(folder+ 'hspace_0.txt').to_numpy().flatten()
        hspace_1 = pd.read_csv(folder+ 'hspace_1.txt').to_numpy().flatten()
    else:
        hspace_0 = np.arange(truc1)
        hspace_1 = np.arange(truc1)

    n0 = len(hspace_0)
    n1 = len(hspace_1)

    folder = f'../../data/3ncut_two_zeropi/truc1=500/'
    n_theta0 = np.load(folder+'n_theta0.npy')
    n_theta1 = np.load(folder+'n_theta1.npy')

    n_theta0 = ut.truncate_2(n_theta0, hspace_0)
    n_theta1 = ut.truncate_2(n_theta1, hspace_1)

    folder_save = f'data/3ncut_two_zeropi/truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = pd.read_csv(folder_save+ 'eval_tot.txt').to_numpy().flatten()
    eket_tot = ssp.csr_matrix(np.load(folder_save+ 'eket_tot.npy'))

    ###  Get wavefunction overlap for the truncated dressed states
    bare_state = [[qt.tensor(qt.basis(n0, i), qt.basis(n1, j))
                            for j in range(n1)]
                                for i in range(n0)]
    arg = [bare_state, n0, n1]

    print("ut.find_overlap..... Time:", datetime.now(pytz.timezone('America/Denver')))
    result = Parallel(n_jobs=100)(delayed(ut.find_overlap)(i, *arg) for i in eket_tot)
    top_index = [result[i][0] for i in range(eval_tot.shape[0])]
    top_overlap = [result[i][1] for i in range(eval_tot.shape[0])]
    np.save(folder_save+f'top_index.npy', top_index)
    np.save(folder_save+f'top_overlap.npy', top_overlap)
    print("pd.DataFrame(top_overlap)..... Time:", datetime.now(pytz.timezone('America/Denver')))

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

    n_theta0_dress = ssp.kron(n_theta0, ssp.identity(n1))
    n_theta1_dress = ssp.kron(ssp.identity(n0), n_theta1)
    n_theta0_dress = (eket_tot @ n_theta0_dress @ eket_tot.conj().T).todense()
    n_theta1_dress = (eket_tot @ n_theta1_dress @ eket_tot.conj().T).todense()
    print("folder_save.... Time:", datetime.now(pytz.timezone('America/Denver')))

    pd.DataFrame(hspace_full).to_csv(folder_save+ f'hspace_full.txt', sep=',', index=False, header=True)
    np.save(folder_save+f'n_theta0_dress.npy', n_theta0_dress)
    np.save(folder_save+f'n_theta1_dress.npy', n_theta1_dress)
    print(f'len(hspace_full) = {len(hspace_full)}')
    print(f'np.shape(n_theta0_dress) = {np.shape(n_theta0_dress)}')
    print(f'np.shape(n_theta1_dress) = {np.shape(n_theta1_dress)}')

if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("NUMEXPR_NUM_THREADS =", os.environ.get('NUMEXPR_NUM_THREADS'))
    print("MKL_NUM_THREADS =", os.environ.get('MKL_NUM_THREADS'))
    ut.print_time()

    # import_para()
    # generate_data()
    # generate_data_3ncut()
    # reduce_eket()
    generate_ekettot_3ncut(test=False)
    # generate_nop_3ncut()
    # generate_eval()

    ut.print_time()



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

















