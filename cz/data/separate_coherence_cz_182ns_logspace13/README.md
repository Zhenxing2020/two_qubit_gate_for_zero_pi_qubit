# Extended independent coherence sweep

Both variable scales use `3 * np.logspace(0, 3, 13)` microseconds (3 to 3000 us):

1. Vary logical scale; auxiliary scale fixed at 30 us.
2. Vary auxiliary scale; logical scale fixed at 300 us.

This gives 26 plotted rows / 25 distinct parameter pairs. The 5 original pairs
(80 ME columns) are reused after explicit configuration/shape validation;
20 additional pairs require 320 new ME solves. Settings remain 60 dressed
states, CZ t_g=182.010007 ns, and at most 80 workers (one BLAS thread each).

The original `cz/data/separate_coherence_cz_182ns/` results are unchanged.
The extended script is `cz/scripts/run_2q_fidelity_separate_coherence.py`.

## Start outside the Codex sandbox

In a regular server terminal:

```bash
bash /home/zlqed/two_qubit_gate/zeropi/two_qubit_gate_clean/cz/scripts/start_separate_coherence_sweep_background.sh
```

This uses nohup, stdin=/dev/null, and logs to this directory's run.log.
The regular-server-shell job can continue after that shell or Codex exits.
The current Codex tool sandbox does not permit connecting to existing tmux
sockets and cleans up background child processes when its command exits;
therefore invoking that launcher inside the tool sandbox is NOT equivalent.

## Resume / monitor

Run the same launcher after an interruption. Validated final column checkpoints
are atomic and are not recomputed. Physics changes invalidate signatures;
changing sweep ranges or worker counts does not. A file lock rejects duplicate
concurrent runs. Historical original-run imports predate signatures, so they
require matching original physical settings, selected pulse and dressed basis.

`status.json` reports completed pairs/scan rows. `all_results.csv` and the PNG/PDF
plots update after each whole parameter pair completes. Partial results/plots
are NOT evidence that all 26 scan rows are finished; check state='complete'.
Each completed ME column is saved immediately as `channel_L*_A*_col*.npz`.

```bash
tail -f cz/data/separate_coherence_cz_182ns_logspace13/run.log
cat cz/data/separate_coherence_cz_182ns_logspace13/status.json
```
