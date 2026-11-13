"""Copy of run_2q_fidelity.py: independent logical/auxiliary noise references.

Run with the project's qutip4 or zp2q Python environment. Edit CONFIG below;
no CLI. T values are reference scales, not equal lifetimes for every level.
The original Hamiltonian, Gaussian waveform and common-local-Z optimized,
leakage-inclusive trace-decreasing average gate fidelity are preserved.
"""
import os
# Avoid BLAS oversubscription when running the 80 independent ME solves.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-separate-coherence')
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

GATE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = GATE_DIR.parent
sys.path.append(str(PROJECT_DIR))
os.chdir(GATE_DIR)

import scqubits.settings as settings
import utils_2Q_gate_zp as ut
import ham_data as hd

settings.OVERLAP_THRESHOLD = 0.3


# ============================================================
# CENTRAL CONFIGURATION -- edit here, no command-line arguments
# ============================================================
CONFIG = dict(
    cz_run=True,
    n_truc_list=[60],
    reduced_model='charge_pick',
    folder_load=str(PROJECT_DIR / 'data/Two_qubit_data_Sorted_Truc'),
    noise_dephase_path=str(PROJECT_DIR / 'data/flux_derivative_truc500/gamma_phi_2Q_truc500.npz'),
    pulse_file=str(PROJECT_DIR / 'figure/data/data_cz_fidelity_npz_select.txt'),
    target_tg_ns=182.0,
    logical_levels=(0, 2),
    logical_sweep_us=(3 * np.logspace(0, 2, 13)).tolist(),
    fixed_nonlogical_us=3.0,
    nonlogical_sweep_us=(3 * np.logspace(0, 2, 13)).tolist(),
    fixed_logical_us=300.0,
    apply_decay=True,
    apply_dephase=True,
    filter_ratio=0.3,
    retain_weak_logical_decay=True,
    decay_enlarge=1.0,
    max_step_ideal=1e-3,  # ns, original script setting
    max_step_noisy=1e-3,  # ns, original script setting
    atol=1e-10,
    rtol=1e-10,
    nsteps=1000000,
    max_workers=80,  # flat worker pool; always capped at 100
    output_dir=str(GATE_DIR / 'data/separate_coherence_cz_182ns_3to300_aux3'),
    seed_dir=str(GATE_DIR / 'data/separate_coherence_cz_182ns'),
    resume=True,  # reuse validated columns, including the completed original run
)


def get_config():
    return dict(CONFIG)


# ============================================================
# Load system data
# ============================================================
def load_system_data(cfg):
    """
    Load two-qubit Hilbert space and operators.
    """
    data = {}
    (
        data["hspace_full"],
        data["eket_tot"],
        data["eval_tot"],
        data["n_theta0"],
        data["n_theta1"],
        data["n_theta0_dress"],
        data["n_theta1_dress"],
        data["hspace_0"],
        data["hspace_1"],
        data["hspace_n_theta1"],
        data["hspace_n_theta2"],
        data["logi_state"],
    ) = hd.load_two_qubit_data(cfg["folder_load"], return_full=False)

    data["dim_0"] = len(data["hspace_0"])
    data["dim_1"] = len(data["hspace_1"])
    return data


# ============================================================
# Gate-dependent quantities
# ============================================================
def select_gate_quantities(cfg, data):
    """
    Return gate-dependent quantities.
    """
    gate = {}

    if cfg["cz_run"]:
        gate["drive_term"] = data["n_theta1_dress"]
        gate["W_20_50"] = (
            data["eval_tot"][data["hspace_full"].index("5-0")]
            - data["eval_tot"][data["hspace_full"].index("2-0")]
        )
    else:
        gate["drive_term"] = data["n_theta0_dress"]
        gate["mid_state"] = "8-2"

        idx0 = data["hspace_full"].index("0-2")
        idx1 = data["hspace_full"].index("2-2")
        idxm = data["hspace_full"].index(gate["mid_state"])

        gate["W_0_2"] = data["eval_tot"][idxm] - data["eval_tot"][idx0]
        gate["W_1_2"] = data["eval_tot"][idxm] - data["eval_tot"][idx1]

    return gate


# ============================================================
# NOISE MODEL EDIT AREA
# ============================================================
def build_collapse_ops(cfg, data, eket_truc):
    """Scale by each bare single-qubit source level, BEFORE dressing.

    gamma_ij = |n_ij/n_48|^2 / (1000*T_source_us) [ns^-1].
    gamma_phi_j = abs(saved_q[physical_level_j]) / T_source_us.
    L_phi_j = sqrt(2*gamma_phi_j) |j><j|; ground projector omitted,
    as in the original model. Independent projectors imply that rho_ij
    (i,j>0) dephases at gamma_i+gamma_j, not a uniform 1/T.
    Only auxiliary-source decay is threshold filtered. Logical-source
    decay includes every nonzero downward matrix element in the bare basis.
    No renormalization of physical lifetimes or projected channels is done.
    """
    ops, audit = [], []
    logical = set(cfg['logical_levels'])
    with np.load(cfg['noise_dephase_path']) as noise:
        for q in range(2):
            levels = list(map(int, data[f'hspace_{q}']))
            matrix = data[f'n_theta{q}'] / (2*np.pi)
            transitions, trunc = ut.get_transitions_for_collapse(
                levels, matrix, filter_ratio=cfg['filter_ratio'])
            kept = set(transitions)
            if cfg['retain_weak_logical_decay']:
                kept.update((i, j) for j, sj in enumerate(levels) if sj in logical
                            for i, si in enumerate(levels)
                            if si < sj and abs(trunc[i, j]) > 0)
            raw_gamma = np.abs(noise[f'q{q}'])
            reference = float(abs(matrix[4, 8])**2)
            if reference <= 0:
                raise ValueError('Zero reference matrix element for 8 -> 4')
            for j, sj in enumerate(levels):
                scale = cfg['T_logical_us'] if sj in logical else cfg['T_nonlogical_us']
                if scale <= 0:
                    raise ValueError('Coherence scales must be positive')
                prefactor = 1.0 / (1000.0 * scale * reference)
                phi = float(raw_gamma[sj] / scale) if sj else 0.0
                rate_sum = 0.0
                for i, jj in sorted(kept):
                    if jj != j:
                        continue
                    rate_sum += prefactor * float(abs(trunc[i, j])**2) * cfg['decay_enlarge']**2
                    if cfg['apply_decay']:
                        ops.append(cfg['decay_enlarge'] * ut.get_jump_op_decay(
                            (i, j), data['dim_0'], data['dim_1'], trunc,
                            eket_truc, q == 0, prefactor))
                if cfg['apply_dephase'] and sj != 0:
                    # Helpers expect compressed indexing; map physical levels explicitly.
                    gamma = np.zeros(len(levels))
                    gamma[j] = phi
                    ops.append(ut.get_jump_op_dephase(j, data['dim_0'], data['dim_1'],
                                                      gamma, eket_truc, q == 0))
                audit.append(dict(qubit=q, level=sj, manifold='logical' if sj in logical else 'nonlogical',
                                  reference_us=scale, decay_rate_per_ns=rate_sum,
                                  dephase_rate_per_ns=phi))
    return ops, audit


def _load_cz_pulse_params(folder):
    """Load either a CSV pulse table or a printed ``np.array`` table."""
    try:
        table = pd.read_csv(folder)
        return table[["tg", "drive_amp", "detune"]].to_numpy()
    except (pd.errors.ParserError, KeyError):
        rows = []
        with open(folder, encoding="utf-8") as stream:
            for line in stream:
                match = re.match(
                    r"\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,"
                    r"\s*([-+0-9.eE]+)\s*\]",
                    line,
                )
                if match:
                    rows.append([float(value) for value in match.groups()])
                elif rows:
                    break
        if not rows:
            raise ValueError(f"No [tg, drive_amp, detune] rows found in {folder}")
        return np.asarray(rows)


# ============================================================
# WAVEFORM / HAMILTONIAN EDIT AREA
# Uses ut.build_hamiltonian_2q and ut.drive_gauss_A, as in original.
# ============================================================
def prepare_model(cfg):
    table = _load_cz_pulse_params(cfg['pulse_file'])
    k = int(np.argmin(abs(table[:, 0] - cfg['target_tg_ns'])))
    pulse = table[k]
    if abs(pulse[0] - cfg['target_tg_ns']) > 3:
        raise ValueError('No pulse close to requested gate time')
    data = load_system_data(cfg)
    gate = select_gate_quantities(cfg, data)
    indices = data['hspace_n_theta1'][:cfg['n_truc_list'][0]]
    states = [data['hspace_full'][i] for i in indices]
    logical_idx = [states.index(s) for s in data['logi_state']]
    H, eket = ut.build_hamiltonian_2q(True, indices, data['eval_tot'],
                                     data['eket_tot'], gate['drive_term'])
    args = dict(drive_amp_A=float(pulse[1]),
                drive_freq_A=float(gate['W_20_50'] + 2*np.pi*pulse[2]),
                gate_time=float(pulse[0]))
    return data, H, eket, logical_idx, pulse, args, states


def evolve_column(pair, column, H, ops, logical_idx, pulse_args, cfg):
    """Independent ME solve; save only final state, no nested workers."""
    import qutip as qt
    N = H[0].shape[0]
    i, j = column % 4, column // 4  # column-major vec convention
    rho = qt.basis(N, logical_idx[i]) * qt.basis(N, logical_idx[j]).dag()
    options = qt.Options(max_step=cfg['max_step_noisy'], nsteps=cfg['nsteps'],
                         atol=cfg['atol'], rtol=cfg['rtol'], num_cpus=1,
                         store_states=True, normalize_output=False)
    start = time.monotonic()
    result = qt.mesolve(H, rho, [0.0, pulse_args['gate_time']], ops,
                        args=pulse_args, options=options)
    # Keep full final output for survival/CP diagnostics and original metric.
    final = result.states[-1].full()
    out = channel_path(cfg, pair, column)
    atomic_npz(out, final=final, seconds=time.monotonic()-start,
               model_signature=cfg['model_signature'], pair=pair, column=column)
    return pair, column


def summarize_channel(pair, H, logical_idx, cfg):
    import qutip as qt
    N = H[0].shape[0]
    files = [channel_path(cfg, pair, c) for c in range(16)]
    full = np.column_stack([np.load(f)['final'].reshape(-1, order='F') for f in files])
    prop = qt.Qobj(full, dims=[[[N], [N]], [[4], [4]]])
    # Identical fidelity definition and common-Z correction to original script.
    fidelity = ut.get_fidelity_super_operator(prop, logical_idx, ut.cz_gate(), [True],
                                               use_qt_fidelity=False)
    projected_rows = [i + N*j for j in logical_idx for i in logical_idx]
    logical_super = qt.Qobj(full[projected_rows], dims=[[[2,2],[2,2]], [[2,2],[2,2]]],
                           superrep='super')
    choi_min = float(np.linalg.eigvalsh(qt.to_choi(logical_super).full()).min())
    survival = float(np.real(sum(np.trace(full[:,c].reshape(N,N,order='F'))
                                for c in (0,5,10,15)) / 4))
    logical_survival = float(np.real(sum(np.trace(full[:,c].reshape(N,N,order='F')
                                                [np.ix_(logical_idx,logical_idx)])
                                        for c in (0,5,10,15)) / 4))
    return dict(T_logical_us=pair[0], T_nonlogical_us=pair[1], fidelity=fidelity,
                gate_error=1-fidelity, mean_full_trace=survival,
                mean_logical_survival=logical_survival, logical_choi_min_eigenvalue=choi_min)


def plot_results(rows, ideal_error, cfg):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for ax, scan, xkey, fixed in [
        (axes[0], 'logical_scan', 'T_logical_us', f"T auxiliary = {cfg['fixed_nonlogical_us']:g} us"),
        (axes[1], 'nonlogical_scan', 'T_nonlogical_us', f"T logical = {cfg['fixed_logical_us']:g} us")]:
        selected = sorted([r for r in rows if r['scan'] == scan], key=lambda r:r[xkey])
        ax.loglog([r[xkey] for r in selected], [r['gate_error'] for r in selected], 'o-')
        ax.axhline(ideal_error, color='gray', ls='--', label='Noiseless')
        ax.set(xlabel=xkey.replace('_us', '') + ' (us)', ylabel='Gate error 1 - F', title=fixed)
        ax.grid(True, which='both', alpha=.3)
        ax.legend()
    fig.suptitle('CZ, charge-picked 60 states; independent reference coherence scales')
    for ext in ('png', 'pdf'):
        fig.savefig(Path(cfg['output_dir']) / f'gate_error_scans.{ext}', dpi=180)
    plt.close(fig)


# ============================================================
# CHECKPOINTS -- signature validates physics, not sweep range/worker count
# ============================================================
PHYSICS_KEYS = (
    'cz_run', 'n_truc_list', 'reduced_model', 'folder_load', 'noise_dephase_path',
    'logical_levels', 'apply_decay', 'apply_dephase', 'filter_ratio',
    'retain_weak_logical_decay', 'decay_enlarge', 'max_step_ideal',
    'max_step_noisy', 'atol', 'rtol', 'nsteps',
)


def channel_path(cfg, pair, column):
    return Path(cfg['output_dir']) / f'channel_L{pair[0]:.12g}_A{pair[1]:.12g}_col{column:02d}.npz'


def atomic_npz(path, **arrays):
    import tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def valid_column(path, pair, column, N, signature=None):
    try:
        with np.load(path) as saved:
            if saved['final'].shape != (N, N) or not np.all(np.isfinite(saved['final'])):
                return False
            if signature is not None:
                return (str(saved['model_signature'].item()) == signature
                        and np.array_equal(saved['pair'], pair)
                        and int(saved['column']) == column)
            return True
    except (OSError, ValueError, KeyError, EOFError):
        return False


def model_signature(cfg, pulse, states):
    import hashlib
    import json
    import inspect
    physics = {k:cfg[k] for k in PHYSICS_KEYS}
    physics.update(selected_pulse=pulse.tolist(), selected_states=states)
    # Include data file identities and shared code hashes. Changes invalidate resume.
    physics['input_files'] = {}
    for file in (Path(cfg['folder_load']) / 'two_qubit_data.npz',
                 Path(cfg['noise_dephase_path']), PROJECT_DIR / 'utils_2Q_gate_zp.py',
                 PROJECT_DIR / 'ham_data.py'):
        stat = file.stat()
        physics['input_files'][str(file)] = [stat.st_size, stat.st_mtime_ns]
    physics['noise_code'] = inspect.getsource(build_collapse_ops)
    physics['hamiltonian_code'] = inspect.getsource(prepare_model)
    return hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()


def seed_columns(cfg, pairs, N, pulse, states):
    """Explicitly import the known completed original run, without changing it.

    That run predates signatures: require identical physical configuration,
    selected pulse/states, and finite correctly-shaped column arrays.
    New checkpoints always carry the model signature.
    """
    import json
    seed = Path(cfg['seed_dir'])
    if not cfg['resume'] or not (seed / 'config.json').exists():
        return 0
    old = json.loads((seed / 'config.json').read_text())
    comparable = {k:cfg[k] for k in PHYSICS_KEYS}
    comparable.update(selected_pulse=pulse.tolist(), selected_states=states)
    comparable = json.loads(json.dumps(comparable))  # normalize tuples to JSON lists
    for key, value in comparable.items():
        if old.get(key) != value:
            print(f'Seed not used: physical configuration differs at {key}', flush=True)
            return 0
    manifest = Path(cfg['output_dir']) / 'legacy_seed_signature.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['model_signature'] != cfg['model_signature']:
            print('Seed not used: model changed since first original-run import', flush=True)
            return 0
    else:
        manifest.write_text(json.dumps(dict(model_signature=cfg['model_signature'],
                                            source=str(seed)), indent=2))
    copied = 0
    for pair in pairs:
        for c in range(16):
            dest = channel_path(cfg, pair, c)
            if valid_column(dest, pair, c, N, cfg['model_signature']):
                continue
            source = seed / f'channel_L{pair[0]:g}_A{pair[1]:g}_col{c:02d}.npz'
            # Original filename precision must represent the same numeric pair.
            if any(float(f'{v:g}') != v for v in pair):
                continue
            if valid_column(source, pair, c, N):
                with np.load(source) as saved:
                    atomic_npz(dest, final=saved['final'], seconds=saved['seconds'],
                               model_signature=cfg['model_signature'], pair=pair, column=c)
                copied += 1
    return copied


def main():
    import json
    import qutip as qt
    import fcntl
    from joblib import parallel_config
    cfg = get_config()
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    # Refuse overlapping launches into the same output directory.
    lock = open(out / 'run.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another sweep is already running in this output directory')
    (out / 'run.pid').write_text(str(os.getpid()) + '\n')
    (out / 'status.json').write_text(json.dumps(dict(state='preparing', pid=os.getpid())))
    data, H, eket, logical_idx, pulse, pulse_args, states = prepare_model(cfg)
    cfg['model_signature'] = model_signature(cfg, pulse, states)
    (out / 'config.json').write_text(json.dumps(dict(cfg, selected_pulse=pulse.tolist(),
                                                  selected_states=states), indent=2))
    print('Selected [tg_ns, drive_amp, detune_GHz]:', pulse, flush=True)
    # Canonicalize roundoff at common points such as 30 and 300 us.
    number = lambda t: float(f'{t:.12g}')
    scans = [('logical_scan', number(t), cfg['fixed_nonlogical_us']) for t in cfg['logical_sweep_us']]
    scans += [('nonlogical_scan', cfg['fixed_logical_us'], number(t)) for t in cfg['nonlogical_sweep_us']]
    pairs = list(dict.fromkeys((a,b) for _,a,b in scans))
    copied = seed_columns(cfg, pairs, H[0].shape[0], pulse, states)
    tasks = [(pair,c) for pair in pairs for c in range(16)
             if not cfg['resume'] or not valid_column(channel_path(cfg,pair,c),pair,c,
                                                       H[0].shape[0],cfg['model_signature'])]
    print(f'{len(scans)} scan rows; {len(pairs)} unique pairs; '
          f'{copied} original columns imported; {len(tasks)} new ME solves', flush=True)
    pending_pairs = set(pair for pair,c in tasks)
    channels = {}
    for pair in pairs:
        if pair not in pending_pairs:
            continue
        local = dict(cfg, T_logical_us=pair[0], T_nonlogical_us=pair[1])
        channels[pair], audit = build_collapse_ops(local, data, eket)
        pd.DataFrame(audit).to_csv(out / f'noise_audit_L{pair[0]:.12g}_A{pair[1]:.12g}.csv', index=False)
        print(f'Pair {pair}: {len(channels[pair])} collapse operators', flush=True)
    del data, eket
    options = qt.Options(max_step=cfg['max_step_ideal'], nsteps=cfg['nsteps'],
                         atol=cfg['atol'], rtol=cfg['rtol'])
    ideal_prop = ut.get_propagator(H, [0, float(pulse[0])], 1, [], pulse_args,
                                   logical_idx, options, options)
    ideal = ut.get_fidelity_super_operator(ideal_prop, logical_idx, ut.cz_gate(), [],
                                          use_qt_fidelity=False)
    (out / 'ideal.json').write_text(json.dumps(dict(fidelity=ideal, gate_error=1-ideal), indent=2))
    print(f'Noiseless fidelity={ideal:.12g}, error={1-ideal:.12g}', flush=True)
    cache = {}
    def checkpoint():
        for pair in pairs:
            if pair not in cache and all(valid_column(channel_path(cfg,pair,c),pair,c,
                                          H[0].shape[0],cfg['model_signature']) for c in range(16)):
                cache[pair] = summarize_channel(pair,H,logical_idx,cfg)
                print(f'Completed pair {pair}: error={cache[pair]["gate_error"]:.12g}', flush=True)
        rows = [dict(scan=name, tg_ns=float(pulse[0]), n_truc=H[0].shape[0],
                     ideal_fidelity=ideal, ideal_error=1-ideal, **cache[(a,b)])
                for name,a,b in scans if (a,b) in cache]
        if rows:
            tmp = out / 'all_results.csv.tmp'
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, out / 'all_results.csv')
            plot_results(rows, 1-ideal, cfg)
        (out / 'status.json').write_text(json.dumps(dict(
            state='complete' if len(cache)==len(pairs) else 'running', pid=os.getpid(),
            completed_pairs=len(cache), total_pairs=len(pairs), completed_scan_rows=len(rows),
            total_scan_rows=len(scans)), indent=2))
    checkpoint()
    if tasks:
        workers = min(100, cfg['max_workers'], len(tasks))
        print(f'Starting {len(tasks)} ME solves using {workers} workers', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            completed = Parallel(n_jobs=workers, verbose=10, return_as='generator_unordered')(
                delayed(evolve_column)(pair,c,H,channels[pair],logical_idx,pulse_args,cfg)
                for pair,c in tasks)
            # Summarize only after all pending columns of a pair arrive.
            remaining = {pair:sum(p==pair for p,c in tasks) for pair in pending_pairs}
            for pair,c in completed:
                remaining[pair] -= 1
                if remaining[pair] == 0:
                    checkpoint()
    checkpoint()
    print('Saved results and plots:', out, flush=True)


if __name__ == '__main__':
    main()
