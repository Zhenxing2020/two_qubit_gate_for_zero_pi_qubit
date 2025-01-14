import sys
sys.path.append('../')
import scqubits as scq
import qutip as qt
import numpy as np
from tqdm import tqdm
import datetime
import pytz
import scqubits.settings as settings
settings.OVERLAP_THRESHOLD = 0.3
import scipy as sp
import utils_2Q_gate_zp as ut
import os
from datetime import datetime
import pytz



###################################################################
## Optimize fidelity with differential evolution and sweep
###################################################################
def fidelity_de():
    fidelity = []
    drive_param = []
    fidelity_full = []
    fidelity_300 = []
    # hspace_4 = [0,2,7,25]
    n_cpu = 1
    args = [H0, drive_term, w_trans_1, w_trans_2, hspace_charge, n_cpu, drag]
    for jdx, tg in tqdm(enumerate(tg_vec)):
    # for jdx, tg in tqdm(enumerate(x0_vec[:,0])):
        tg_bounds = (tg+tg_bound[0], tg+tg_bound[1])
        bounds = (tg_bounds, amp1_bounds, amp2_bounds, detune1_bounds, detune2_bounds)
        res = sp.optimize.differential_evolution(
            func=ut.xgate_fidelity_parallel,
            bounds=bounds,
            args=args,
            disp=True,
            callback=ut.print_soln,
            init="sobol",
            workers=workers,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            # x0=x0_vec[jdx],
            polish=False, # 'True' will make the for-loop break
            )
        fidelity.append(res.fun)
        drive_param.append(res.x)

        ### print fidelity for truc=44
        print(res, '\n')
        print('\ntg = ', np.array(tg_vec[:jdx+1]).tolist())
        # print('\ntg = ', np.array((x0_vec[:,0])[:jdx+1]).tolist())

        print(f'\nlog of gate error (truc1={len(hspace_charge)}) = ')
        for i in range(0, len(fidelity), 4):
            print(', '.join(map(str, np.round(fidelity[i:i+4], 8))), ',')

        # fidelity_real = 1-10**np.array(fidelity)
        # print('\nfidelity (truc1) = ')
        # for i in range(0, len(fidelity), 4):
        #     print(', '.join(map(str, np.round(fidelity_real[i:i+4], 8))), ',')

        print(f'\ndrive_param (truc1={len(hspace_charge)}) = ')
        for i in drive_param:
            print(np.round(i,6).tolist(),',')

        ## get fidelity for truc=157
        if drag == 0:
            [alpha_A, alpha_B] = [0, 0]
            [tg, drive_amp_A, drive_amp_B, detune_A, detune_B] = drive_param[jdx]
        elif drag == 1:
            alpha_B = 0
            [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A] = drive_param[jdx]
        else:
            [tg, drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B] = drive_param[jdx]

        n_cpu2 = 30
        argz = [H0_full, drive_full, w_trans_1, w_trans_2, hspace_full, n_cpu2, tg,
                drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
        fidelity_full.append(ut.xgate_fidelity_fast(argz))
        print(f'\nlog of gate error (truc_full={truc_full}) = ')
        for i in range(0, len(fidelity_full), 4):
            print(', '.join(map(str, np.round(fidelity_full[i:i+4], 8))), ',')

        truc3 = 300
        hspace_300 = np.arange(truc3).tolist()
        argz = [H0_full, drive_full, w_trans_1, w_trans_2, hspace_300, n_cpu2, tg,
                drive_amp_A, drive_amp_B, detune_A, detune_B, alpha_A, alpha_B]
        fidelity_300.append(ut.xgate_fidelity_fast(argz))
        print(f'\nlog of gate error (truc_full={truc3}) = ')
        for i in range(0, len(fidelity_300), 4):
            print(', '.join(map(str, np.round(fidelity_300[i:i+4], 8))), ',')


        print("\n***Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')), '\n')
    print('amp1_bounds=',amp1_bounds, ', amp2_bounds=',amp2_bounds,)
    print('detune1_bounds=',detune1_bounds, ', detune2_bounds=', detune2_bounds)


if __name__ == '__main__':
    print(os.path.basename(__file__)) # Print the name of the current Python file
    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))

    drive_phi, drive_theta, drag = False, True,  0
    # amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0, 0.1), (0, 0.04), (0, 0.005), (0, 0.005)] # theta
    amp1_bounds, amp2_bounds, detune1_bounds,  detune2_bounds = [(0.1, 0.2), (0.02, 0.04), (-0.1, 0.005), (-0.005, 0.005)] # phi
    tg_bound = (-0.1, 0.1)
    # tg_vec = [400, 1000] + np.arange(500, 1000, step=100).tolist()
    tg_vec = [30, 35]
    # params = np.array([
# [110, 0.134365, 0.261486, 0.212833, 0.206275] ,
# [120, 0.1772, 0.109632, 0.356868, 0.351128] ,
# [130, 0.078378, 0.193147, -0.206274, -0.211425] ,
# [140, 0.29241, 0.071924, 0.3484, 0.486637] ,
# [40.0,-3.301470,-3.302673,0.105147,0.028068,0.001898,0.00438] ,
# [50.0,-3.77945,-2.803664,0.10746657,0.10323223,-0.30225871,-0.25967603] ,
# [60.0,-3.409958,-3.348597, 0.081606,0.106285,-0.299703,-0.247238] ,
# [70.0,-3.442062,-2.453763,0.069316, 0.109747, -0.311983, -0.252449] ,
# [20, 0.20939, 0.23914, 0.464705, 0.468903] ,
# [30, 0.220554, 0.202162, 0.388947, 0.424067] ,
# [40, 0.200606, 0.16824, 0.38813, 0.390683] ,
# [50, 0.215363, 0.150406, 0.353257, 0.37547] ,
# [60, 0.213582, 0.143426, 0.35721, 0.376478] ,
# [70, 0.208837, 0.134152, 0.352139, 0.36885] ,
# [80, 0.220351, 0.12167, 0.347189, 0.374815] ,
# [90, 0.205536, 0.116973, 0.346827, 0.360525] ,
# [100, 0.202994, 0.113049, 0.343961, 0.356533] ,
    # ])
    # x0_vec = params # extract col=0 and col=3,4,5,6
    # x0_vec = params[:,[0,3,4,5,6]] # extract col=0 and col=3,4,5,6

    workers, popsize = 100, 10
    recombination, tol, mutation = [0.7, 0.01, (0.5, 1.0)]
    truc1, truc_full = 300, 500 # theta
    # truc1, truc_full = 300, 500 # phi
    print('drive_phi=', drive_phi, ', drive_theta = ', drive_theta, ', Drag=', drag)
    print('amp1_bounds=',amp1_bounds, ', amp2_bounds=',amp2_bounds, ', tg_bound=', tg_bound)
    print('detune1_bounds=',detune1_bounds, ', detune2_bounds=', detune2_bounds)
    print('workers=',workers, ', popsize=',popsize)
    print('recombination=',recombination, ', tol=',tol, ', mutation=',mutation)
    if 'params' in globals():
        print('params = ')
        for i in params:
            print(np.round(i,6).tolist(),',')
    if 'tg_vec' in globals():
        print('tg_vec = ')
        for i in range(0, len(tg_vec), 4):
            print(', '.join(map(str, tg_vec[i:i+4])), ',')


    [H0, drive_term, w_trans_1, w_trans_2, hspace_charge] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc1,)
    [H0_full, drive_full, _, _, _] = ut.zero_pi_initialize(drive_phi, drive_theta, truncation=truc_full,)
    hspace_full = np.arange(truc_full).tolist()
    print('truc1 =', truc1, ', truc2 (in optimization) =', len(hspace_charge))
    print('truc1_full =', truc_full, ', truc2_full =', len(hspace_full))

    ### optimize
    fidelity_de()

    print("Current Mountain Time:", datetime.now(pytz.timezone('America/Denver')))



