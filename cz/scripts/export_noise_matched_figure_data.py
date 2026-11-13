"""Export ideal/noisy curves on identical recovered noise-source pulses."""
from pathlib import Path
import json,re,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figure/data/cz_new_phase_20261112'
IDEAL=ROOT/'cz/data/ideal_cz_noise_pulses_new_phase'
SENS=ROOT/'cz/data/coherent_pulse_sensitivity_noise_pulses_n200_tg182'
PULSES=ROOT/'figure/data/pulse_params_noise_scaling.txt'
NOISE=ROOT/'figure/data/noise_scaling_readable.csv'

def main():
    OUT.mkdir(exist_ok=True)
    p=pd.read_csv(PULSES);n=pd.read_csv(NOISE);ideal=pd.read_csv(IDEAL/'all_results.csv')
    np.testing.assert_allclose(p.tg,n.gate_time_ns,rtol=0,atol=1e-12)
    for source, dest in [('amplitude','drive_amp'),('detune_ghz','detune')]:
        if source in n:
            np.testing.assert_allclose(n[source],p[dest],rtol=0,atol=1e-12)
    assert len(ideal)==90 and len(p)==30
    result=p.copy()
    for size in (60,200,1000):
        d=ideal[ideal.n_truc==size].sort_values('pulse_index')
        np.testing.assert_allclose(d[['tg_ns','drive_amp','detune_ghz']],p,rtol=0,atol=1e-12)
        result[f'f_{size}']=d.log10_error.to_numpy()
    mapping={}
    for c in [name for name in n.columns if name.startswith('error__')]:
        m=re.fullmatch(r'error__T1_Tphi_(\d+)us__(nominal_1x|decay_x10|dephasing_x10|both_x10)',c)
        if not m:raise ValueError(c)
        t,mode=m.groups();name=f'f_60_{t}us'+('' if mode=='nominal_1x' else '_'+mode)
        rates=n[c].to_numpy(float);assert np.all((rates>0)&(rates<=1))
        result[name]=np.log10(rates);mapping[c]=name
    # Preserve the previous pulse-family export once, but remove obsolete columns
    # from the current table rather than mixing historic results or placeholders.
    archive=OUT/'previous_pulse_export'
    if not archive.exists():
        archive.mkdir()
        for f in OUT.iterdir():
            if f.is_file():shutil.copy2(f,archive/f.name)
    target=OUT/'data_cz_fidelity_npz_select.txt'
    tmp=target.with_suffix('.tmp');result.to_csv(tmp,index=False,float_format='%.16g');tmp.replace(target)
    # Remove the now-obsolete strict old-schema compatibility table from CURRENT exports.
    (OUT/'data_cz_fidelity_npz_select_legacy_schema.txt').unlink(missing_ok=True)
    sens=pd.read_csv(SENS/'all_results.csv');meta=json.loads((SENS/'config.json').read_text())
    base=np.asarray(meta['base_params'])
    np.testing.assert_allclose(base,p.iloc[(p.tg-182).abs().argmin()],rtol=0,atol=1e-12)
    a=sens[sens.scan=='amplitude'].sort_values('deviation');d=sens[sens.scan=='detuning'].sort_values('deviation')
    assert len(a)==len(d)==101
    with np.load(ROOT/'figure/data/cz_sensitivity.npz') as old:
        keys=old.files
        const={k:old[k].copy() for k in ('nop','detune_over_2pi_mhz','detune_ang_ghz',
             'threshold_fidelity','threshold_log_error','freq_half_width_mhz','amp_percent_half_width')}
    for mode,name in [('optimized_z','cz_sensitivity.npz'),('fixed_z','cz_sensitivity_fixed_z.npz')]:
        center=a.loc[a.deviation.abs().idxmin()]
        v=dict(source_log=np.array(str((SENS/'all_results.csv').relative_to(ROOT))),base_params=base,
            tg_ns=np.array([base[0]]),tg_log_errors=np.array([np.log10(center[f'gate_error_{mode}'])]),
            amp=a.amplitude.to_numpy(),amp_dimensionless=a.amplitude.to_numpy()*(np.e**2-1)*float(const['nop'])/float(const['detune_ang_ghz']),
            amp_log_errors=np.log10(a[f'gate_error_{mode}'].to_numpy()),
            detune_ghz=d.detune_ghz.to_numpy(),detune_mhz=d.detune_ghz.to_numpy()*1000,
            detune_log_errors=np.log10(d[f'gate_error_{mode}'].to_numpy()),
            opt_amp=np.array(base[1]),opt_detune_ghz=np.array(base[2]),
            freq_mask=np.asarray(abs(d.deviation)<=.5+1e-12),amp_mask=np.asarray(abs(a.deviation)<=.02+1e-12),**const)
        assert set(v)==set(keys)
        np.savez(OUT/name,**{k:v[k] for k in keys})
    (OUT/'noise_column_mapping.json').write_text(json.dumps(mapping,indent=2))
    (OUT/'README.md').write_text('''# Noise-matched CZ pulses: current new-phase results

All rows use the pulse family in figure/data/noise_scaling_readable.csv.
Pulse amplitude and detuning were recovered from cz/data/all_results.csv,
then checked against the amplitude/detune columns now present in the CSV.
All 30 complete pulse tuples match to absolute tolerance 1e-12. The recovered pulse table is ../pulse_params_noise_scaling.txt.

## data_cz_fidelity_npz_select.txt
Columns: tg (ns), drive_amp, detune (GHz); f_60/f_200/f_1000 (newly recomputed
ideal, charge-picked, optimized common-channel local Z); followed by all 16
noise curves from noise_scaling_readable.csv. EVERY f_* column is log10(1-F).
Noise column names are f_60_{T}us for nominal and f_60_{T}us_{condition} for
nonlogical decay_x10, dephasing_x10, both_x10. T = 3,30,300,3000 us.
Noise values are taken as supplied (not recomputed in this export).
noise_column_mapping.json records the exact source-column mapping.

Only newly computed ideal columns and supplied noise columns are retained.
All legacy-calibration columns and NaN placeholders are removed.
The previous, different pulse-family export is archived in previous_pulse_export/;
no old-phase/old-family results are mixed into the CURRENT table.

## cz_sensitivity.npz / cz_sensitivity_fixed_z.npz
Exact keys/order and units as the original figure/data/cz_sensitivity.npz.
Sensitivity uses ONLY the recovered pulse near 182 ns:
[182.014628 ns, amplitude 0.00788503, detune 0.01312045 GHz].
n_truc=200; 50 workers; 101 detuning points over ±1 MHz and 101 amplitude
points over ±5%. Nominal local Z is re-optimized at each point in the main file;
the second file fixes the nominal local Z correction. *_log_errors are log10(1-F).
No gate-time sweep was requested: tg_ns/tg_log_errors have one nominal point.
amp_dimensionless retains the historical axis-conversion constants from the
original NPZ (nop=0.537, detune/(2*pi)=33.45 MHz), not re-estimated quantities.

Notebook plots updated: cz/plot_cz_v1.ipynb cell 2 and the final section of
cz/sensitivity_pulse_params.ipynb. Raw recomputation outputs:
- cz/data/ideal_cz_noise_pulses_new_phase/
- cz/data/coherent_pulse_sensitivity_noise_pulses_n200_tg182/
''')
    assert len(mapping)==16
    loaded=pd.read_csv(target)
    assert len(loaded)==30 and not loaded.isna().any().any()
    for c,name in mapping.items():np.testing.assert_allclose(10**loaded[name],n[c],rtol=1e-13)
    print('PASS: clean table, all 90 ideal and 16 noisy curves aligned; NPZ updated:',OUT,flush=True)
if __name__=='__main__':main()
