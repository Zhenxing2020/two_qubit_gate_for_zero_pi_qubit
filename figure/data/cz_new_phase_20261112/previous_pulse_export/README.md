# Current CZ results, new local-Z phase correction

Original files in figure/data are unchanged. Export script:
`cz/scripts/export_new_phase_figure_data.py`.

## cz_sensitivity.npz
Exact same keys, key order, array ranks and dtype kinds as ../cz_sensitivity.npz.
Current n_truc=200; nominal pulse [182.010007 ns, 0.007880, 0.013127 GHz].
Errors use a separately optimized common local-Z correction at every point.
- amp / amp_log_errors / amp_mask: 101 amplitude points (±5% relative).
- detune_ghz / detune_mhz / detune_log_errors / freq_mask: 101 frequency points (±1 MHz deviation).
- tg_ns / tg_log_errors: ONE nominal point only. No gate-time sensitivity sweep was computed.
- All *_log_errors store log10(1-F), not F or raw error.
- opt_amp and opt_detune_ghz are the supplied nominal optimized pulse, NOT minima fitted to the new scans.
- freq_mask and amp_mask select the sampled ±500 kHz and ±2% windows.
- amp_dimensionless uses the original plotting convention:
  amp*(exp(2)-1)*nop/detune_ang_ghz; nop=0.537 and detune/(2*pi)=33.45 MHz
  are inherited axis-conversion constants, not newly calculated circuit quantities.

cz_sensitivity_fixed_z.npz has the same schema but uses fixed nominal local-Z angles.
See source CSV for both fidelity definitions and raw deviations.

## data_cz_fidelity_npz_select.txt
Comma-separated text, same 30 pulse rows and original 34 columns in the same order,
plus f_1000 appended. f_60, f_charge_60 and f_200 are newly calculated; f_1000
contains the NEW charge-picked 1000-state results. All f_* values are log10(1-F).
Unused legacy truncations, lowest-state models, and noisy columns are NaN:
they were NOT recomputed and are NOT copied from the old phase-correction results.
In particular f_900 is NOT relabeled as f_1000, and f_low1000_1e3 is NOT overwritten
with a charge-picked result.

`data_cz_fidelity_npz_select_legacy_schema.txt` has EXACTLY the original 34 columns;
use it if strict column equality is required. It necessarily excludes f_1000.

Sources:
- cz/data/ideal_cz_selected_new_phase/all_results.csv (90 ideal points, 60/200/1000 states)
- cz/data/coherent_pulse_sensitivity_n200_tg182/all_results.csv (202 scan rows)
