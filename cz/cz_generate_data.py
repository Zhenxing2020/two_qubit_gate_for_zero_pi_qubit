import sys
sys.path.append('../')

import pandas as pd
import datetime
import pytz
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import utils_2Q_gate_zp as ut
from datetime import datetime
import pytz
import os
import numpy as np
import qutip as qt
from multiprocessing import Pool

def generate_data():
    truc1, truc_tot, charge_pick = 400, 500, True
    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    args_all = ut.get_operator_two_zeropi_v2(truc1=truc1, truc_tot=truc_tot, charge_pick=charge_pick)
    [hspace_0, hspace_1, top_index, top_overlap,
    n_theta0_dress, n_theta1_dress, eval_tot, hspace_full] = args_all

    folder = f'../two_qubit_data_truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    os.mkdir(folder)
    pd.DataFrame(n_theta0_dress).to_csv(folder+ 'n_theta0_dress.txt', sep=',', index=False, header=True)
    pd.DataFrame(n_theta1_dress).to_csv(folder+ 'n_theta1_dress.txt', sep=',', index=False, header=True)
    pd.DataFrame(eval_tot).to_csv(folder+ 'eval_tot.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_full).to_csv(folder+ 'hspace_full.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_0).to_csv(folder+ 'hspace_0.txt', sep=',', index=False, header=True)
    pd.DataFrame(hspace_1).to_csv(folder+ 'hspace_1.txt', sep=',', index=False, header=True)
    pd.DataFrame(top_index).to_csv(folder+ 'top_index.txt', sep=',', index=False, header=True)
    pd.DataFrame(top_overlap).to_csv(folder+ 'top_overlap.txt', sep=',', index=False, header=True)


def cz_fidelity_pool(args):
    [tg, _, _, _, _, _, _, _] = args
    return tg, ut.cz_fidelity(args)

def import_para():
    n_cpu_full, n_cpu_pool = 1, 100
    # truc1, truc_tot, charge_pick = 300, 2000, True
    truc1, truc_tot, charge_pick = 400, 500, True
    truc_tot_2 = 500
    cz = pd.read_csv('data/data_cz_sesolve_truc1=300_select_v2.txt')
    para_tot = cz[['tg', 'drive_amp', 'detune']].to_numpy().tolist()

    print('n_cpu_full=', n_cpu_full, ', n_cpu_pool=', n_cpu_pool,)
    print('\ntruc1=', truc1, ', truc_tot=', truc_tot, ', charge_pick=', charge_pick)
    print('truc_tot_2=', truc_tot_2)
    folder = f'../two_qubit_data_truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = 2*np.pi* pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    n_theta0_dress = 2*np.pi* pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
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

    print("\nCurrent Mountain Time:", datetime.now(pytz.timezone('America/Denver')))





















