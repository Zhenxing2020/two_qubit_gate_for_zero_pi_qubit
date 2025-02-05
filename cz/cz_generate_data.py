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


def generate_data_3ncut():
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
    n_cpu_full, n_cpu_pool = 1, 100
    # truc1, truc_tot, charge_pick = 300, 2000, True
    truc1, truc_tot, charge_pick = 300, 1000, False
    truc_tot_2 = 500
    cz = pd.read_csv('data/data_cz_3ncut_truc1=300_select.txt')
    para_tot = cz[['tg', 'drive_amp', 'detune']].to_numpy()[[0, 15, 30],:]

    print('n_cpu_full=', n_cpu_full, ', n_cpu_pool=', n_cpu_pool,)
    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_tot_2)
    folder = f'../../data/two_qubit_data_truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    # n_theta0_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
    n_theta1_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta1_dress.txt').to_numpy()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()

    # the full Hilbert space
    logic_states = ['0-0', '0-2', '2-0', '2-2']
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    drive_term = n_theta1_dress
    H0_full = qt.Qobj(np.diag(eval_tot))
    # H_drive_full = [H0_full, [2*np.pi* qt.Qobj(drive_term), ut.drive_gauss_A] ]
    # logic_idx_full = [hspace_full.index(i) for i in logic_states]

    # Truncate the imported full Hilbert space
    truc_index = np.arange(truc_tot_2)
    hspace_truc = hspace_full[:truc_tot_2]
    H0_truc = ut.truncate_2( H0_full, truc_index)
    drive_truc = ut.truncate_2(drive_term, truc_index)
    H_drive_truc = [H0_truc, [ drive_truc, ut.drive_gauss_A] ]
    logic_idx_truc = [hspace_truc.index(i) for i in logic_states]

    pool = Pool(processes=n_cpu_pool)
    sesolve_args = []
    for i in range(len(para_tot)):
        sesolve_args.append([para_tot[i][0], para_tot[i][1], para_tot[i][2],
                            n_cpu_full, hspace_truc, W_20_50, H_drive_truc, logic_idx_truc])
    result = []
    for tg, fidelity in pool.imap_unordered(cz_fidelity_pool, sesolve_args):
        result.append([tg, fidelity])
    result = pd.DataFrame(result, columns=['tg', 'fidelity']).sort_values("tg", ascending=True).to_numpy().tolist()
    print(f'fidelity (truc={truc_tot_2}) = ')
    for i in result:
        print(i, ',')

if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Start Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    import_para()
    # generate_data()
    # generate_data_3ncut()

    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















