# Noise-matched CZ pulses: current new-phase results

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
