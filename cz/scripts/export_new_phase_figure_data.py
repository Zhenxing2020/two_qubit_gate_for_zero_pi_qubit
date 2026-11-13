"""Compatibility entry point: export ONLY the current noise-matched pulse family.

Configuration and validation live in export_noise_matched_figure_data.py.
The previous different-pulse export is retained under previous_pulse_export/.
"""
from export_noise_matched_figure_data import main

if __name__ == '__main__':
    main()
