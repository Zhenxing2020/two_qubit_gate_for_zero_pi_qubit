# CZ independent coherence sweep: 3–300 us, 13 points

- Fixed T_logical = 300 us; sweep T_nonlogical = 3*np.logspace(0,2,13) us.
- Fixed T_nonlogical = 3 us; sweep T_logical = 3*np.logspace(0,2,13) us.
- CZ gate time 182.010007 ns; charge-picked 60 states; up to 80 workers.
- 26 scan rows, 25 distinct parameter pairs; (300,3) is shared.
- Original completed data remain unchanged in ../separate_coherence_cz_182ns/.
- Compatible original checkpoints are reused; other completed checkpoints in
  this output directory are validated by model signature on restart.

Run from a regular server terminal, outside the Codex tool sandbox:

```bash
bash /home/zlqed/two_qubit_gate/zeropi/two_qubit_gate_clean/cz/scripts/start_separate_coherence_sweep_background.sh
```

Progress: run.log and status.json. CSV and plots update as whole pairs finish.
Check status.json state=complete to confirm that all 26 scan rows are finished.
