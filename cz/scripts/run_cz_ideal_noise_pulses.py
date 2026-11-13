"""All selected CZ pulses using run_2q_fidelity.py; edit CONFIG, no CLI."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/cz-ideal-all-mpl')
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import json
import hashlib
import fcntl
import numpy as np
import pandas as pd
from joblib import parallel_config
import run_2q_fidelity as original

PROJECT=Path(__file__).resolve().parents[2]
# ================= CENTRAL CONFIGURATION =================
CONFIG=dict(
    n_truc_list=[60,200,1000],
    pulse_file=str(PROJECT/'figure/data/pulse_params_noise_scaling.txt'),
    output_dir=str(PROJECT/'cz/data/ideal_cz_noise_pulses_new_phase'),
    n_job=30,  # pulse-level workers; no nested workers
    max_step_ideal=1e-3,
    reduced_model='charge_pick',
)
# Waveform/model: original.run_one_truncation -> ut.build_hamiltonian_2q,
# ut.drive_gauss_A. Phase correction: current shared cz_phase_correct().

def main():
    cfg=original.get_config()
    cfg.update(CONFIG)
    cfg.update(cz_run=True,calculate_ideal=True,calculate_noise=False,
               num_cpus_ideal=1,num_cpus_noisy=1,use_qt_fidelity=False)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'run.pid').write_text(str(os.getpid()))
    params=original._load_cz_pulse_params(cfg['pulse_file'])
    cfg['params']=params;cfg['tg_list']=list(range(len(params)))
    cfg['ideal_checkpoint_dir']=str(out/'points')
    provenance=dict(config=CONFIG,params=params.tolist(),phase_correction='cz_phase_optimize_channel',
                    fidelity='trace_decreasing',num_cpus_ideal=1)
    provenance['source_hashes']={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (PROJECT/'utils_2Q_gate_zp.py',Path(original.__file__),Path(__file__))}
    for p in (PROJECT/'data/Two_qubit_data_Sorted_Truc/two_qubit_data.npz',):
        st=p.stat();provenance[str(p)]=[st.st_size,st.st_mtime_ns]
    cfg['checkpoint_signature']=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    provenance['signature']=cfg['checkpoint_signature']
    (out/'config.json').write_text(json.dumps(provenance,indent=2))
    print(f'{len(params)} pulses x {len(cfg["n_truc_list"])} truncations; {cfg["n_job"]} workers',flush=True)
    data=original.load_system_data(cfg);gate=original.select_gate_quantities(cfg,data)
    ideal_opt,noisy_opt=original.ut.get_qutip_options(cfg['max_step_ideal'],cfg['max_step_noisy'])
    rows=[]
    for n in cfg['n_truc_list']:
        print(f'Starting n_truc={n}',flush=True)
        with parallel_config(backend='loky',inner_max_num_threads=1):
            values=original.run_one_truncation(n,cfg,data,gate,ideal_opt,noisy_opt)['ideal']
        for i,(p,v) in enumerate(zip(params,values)):
            rows.append(dict(n_truc=n,pulse_index=i,tg_ns=p[0],drive_amp=p[1],detune_ghz=p[2],
                             fidelity=1-10.**v,gate_error=10.**v,log10_error=v))
        pd.DataFrame(rows).to_csv(out/'all_results.csv',index=False)
    (out/'status.json').write_text(json.dumps(dict(state='complete',points=len(rows))))
    print('COMPLETE:',out,flush=True)
if __name__=='__main__':
    main()
