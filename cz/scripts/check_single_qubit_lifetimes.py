"""Audit the bare single-qubit noise model used by run_2q_fidelity.py.

No gate evolution or large dressed-eigenvector loading is required.
Times are model predictions, not independent physical T1/Tphi measurements.
"""
from pathlib import Path
import numpy as np
import csv

# ==================== Configuration ====================
PROJECT_DIR = Path(__file__).resolve().parents[2]
SYSTEM_PATH = PROJECT_DIR / 'data/Two_qubit_data_Sorted_Truc/two_qubit_data.npz'
DEPHASE_PATH = PROJECT_DIR / 'data/flux_derivative_truc500/gamma_phi_2Q_truc500.npz'
OUTPUT_DIR = PROJECT_DIR / 'cz/data/single_qubit_lifetimes'
N_LEVELS = 20
T1_TPHI_OTHER_US = 170.0
FILTER_RATIO = 0.3
LOW_STATES = 50
DECAY_ENLARGE = 1.0  # Collapse AMPLITUDE scaling, hence rate scales as its square.

# ==================== Noise-model audit ====================
def inverse_rate_us(rate):
    """Convert ns^-1 to microseconds; zero rate means infinite lifetime."""
    return np.inf if rate == 0 else 1.0 / (1000.0 * rate)


def calculate_tables(system_path=SYSTEM_PATH, dephase_path=DEPHASE_PATH):
    levels, transitions = [], []
    with np.load(system_path, allow_pickle=True) as data, np.load(dephase_path) as noise:
        for q in range(2):
            matrix = data[f'n_theta{q + 1}']  # RAW matrix, no 2*pi.
            hspace = data[f'hspace_{q + 1}_charge'].astype(int)
            if np.any(np.diff(hspace) <= 0):
                raise ValueError('Expected increasing charge-selected level indices.')
            reference = abs(matrix[4, 8]) ** 2
            if reference == 0:
                raise ValueError('Zero reference matrix element for 8 -> 4.')
            prefactor = 1.0 / (1000 * T1_TPHI_OTHER_US * reference)
            low = hspace[hspace < LOW_STATES]
            n_max = np.abs(matrix[np.ix_(low, low)]).max()
            selected_position = {int(s): i for i, s in enumerate(hspace)}
            gamma = np.abs(noise[f'q{q}'] / T1_TPHI_OTHER_US)
            for j in range(min(N_LEVELS, len(matrix))):
                rates = prefactor * np.abs(matrix[:j, j]) ** 2 * DECAY_ENLARGE**2
                full_rate = float(rates.sum())
                included = j in selected_position
                cutoff = n_max * FILTER_RATIO * (1 if j < LOW_STATES else 10)
                filtered_rate = 0.0
                for i, rate in enumerate(rates):
                    retained = included and i in selected_position and abs(matrix[i, j]) > cutoff
                    if retained:
                        filtered_rate += float(rate)
                    transitions.append(dict(qubit=q, source=j, target=i,
                                            rate_per_ns=float(rate), retained=retained))
                # Ground projector is omitted by construct_c_ops_2q. For an
                # independent sqrt(2*gamma_j)|j><j| channel, rho_0j decays
                # at gamma_j; rho_ij (i,j>0) at gamma_i + gamma_j.
                mapped_gamma = float(gamma[j]) if j else 0.0
                actual_gamma = (float(gamma[selected_position[j]]) if included and j else
                                (0.0 if j == 0 else np.nan))
                levels.append(dict(
                    qubit=q, level=j, in_charge_basis=included,
                    charge_position=selected_position.get(j, np.nan),
                    decay_full_per_ns=full_rate,
                    T1_full_us=inverse_rate_us(full_rate),
                    decay_filtered_per_ns=filtered_rate if included else np.nan,
                    T1_filtered_us=inverse_rate_us(filtered_rate) if included else np.nan,
                    gamma_phi_mapped_per_ns=mapped_gamma,
                    Tphi_mapped_us=inverse_rate_us(mapped_gamma),
                    gamma_phi_current_per_ns=actual_gamma,
                    Tphi_current_us=inverse_rate_us(actual_gamma),
                ))
    return levels, transitions


def main():
    levels, transitions = calculate_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [('levels', levels), ('transitions', transitions)]:
        with (OUTPUT_DIR / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'T1/Tphi reference = {T1_TPHI_OTHER_US} us; filter_ratio = {FILTER_RATIO}')
    print('Full: all downward channels. Filtered: charge-selected channels used by the gate script.')
    print('Tphi_mapped: gamma indexed by physical level; Tphi_current: current compressed indexing.')
    print('Tphi refers to coherence with level 0, not population decay. Ground Tphi=inf is a convention.')
    print('NaN: level absent from charge-selected basis; inf: zero model rate.')
    columns = ['qubit', 'level', 'T1_full_us', 'T1_filtered_us', 'Tphi_mapped_us', 'Tphi_current_us']
    print(' '.join(f'{key:>18}' for key in columns))
    for row in levels:
        print(' '.join(f'{row[key]:18.6g}' for key in columns))
    print(f'\nSaved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
