"""
Cleaned and documented compatibility wrapper for `utils_2Q_gate_zp`.

This module re-exports the full public API of `utils_2Q_gate_zp` while
adding module-level documentation and enhanced docstrings for commonly
used functions. It does not modify runtime behavior: all functions are
delegated to the original implementation.

Usage
-----
- Drop-in replacement: `import utils_2Q_gate_zp_clean as ut`
- Or switch imports in existing notebooks/scripts to this module to
  get improved help() and IDE hover docs without changing behavior.

Notes
-----
- We import the original module and re-bind its public attributes into
  this namespace. For selected key functions, we enrich their
  `__doc__` dynamically to improve discoverability.
- If the upstream module changes, this wrapper will reflect those
  changes automatically on next import.
"""

from __future__ import annotations

import types
import typing as _t

_orig = __import__("utils_2Q_gate_zp", globals(), locals(), ["*"], 1)


def _is_public(name: str, value: _t.Any) -> bool:
    if name.startswith("_"):
        return False
    # Skip modules/types we don't want to shadow intentionally
    return True


# Build __all__ from the original public symbols
__all__ = sorted([name for name in dir(_orig) if _is_public(name, getattr(_orig, name))])

# Re-export everything into this module's globals
for _name in __all__:
    globals()[_name] = getattr(_orig, _name)


def _set_doc(fn_name: str, doc: str) -> None:
    """Attach/replace a function's docstring if it exists in the original module."""
    obj = globals().get(fn_name)
    if callable(obj):
        try:
            obj.__doc__ = doc
        except Exception:
            # Some callables may not allow setting __doc__ (e.g., builtins or C-accelerated);
            # ignore gracefully to avoid changing behavior.
            pass


# Enrich docstrings for frequently used functions (without altering behavior)
_set_doc(
    "set_fig_font",
    """Set matplotlib font sizes for figures.

    Side effects:
        Configures global matplotlib rcParams for font sizes (axes, ticks, legend, figure).

    Returns:
        None
    """,
)

_set_doc(
    "truncate",
    """Return a square top-left `dimension x dimension` truncation of a qutip Qobj.

    Args:
        operator: A square `qutip.Qobj` operator.
        dimension: Target dimension to keep (upper-left block).

    Returns:
        qutip.Qobj: Truncated operator.
    """,
)

_set_doc(
    "truncate_2",
    """Truncate a qutip object or ndarray to an index subset.

    Behavior:
        - If the input is square (matrix), keeps rows/cols at `indices`.
        - Otherwise (state/vector), keeps entries at `indices`.

    Args:
        qobj: qutip.Qobj or numpy array-like.
        indices: Iterable of integer indices to keep.

    Returns:
        qutip.Qobj: Truncated object.
    """,
)

_set_doc(
    "remove_global_phase",
    """Normalize a unitary by removing the global phase using element (0, 0) as reference.

    Args:
        op: qutip.Qobj unitary matrix.

    Returns:
        qutip.Qobj: Phase-normalized unitary.
    """,
)

_set_doc(
    "gate_fidelity",
    """Average gate fidelity between two same-dimension unitaries.

    Args:
        Utarg: Target unitary (qutip.Qobj).
        Ucand: Candidate unitary (qutip.Qobj).

    Returns:
        qutip.Qobj: Scalar fidelity value as Qobj (consistent with original implementation).

    Raises:
        ValueError: If dimensions do not match.
    """,
)

_set_doc(
    "drive_gauss_A",
    """Gaussian-enveloped cosine drive for qubit A.

    Args (from `args` dict):
        - drive_amp_A (float)
        - drive_freq_A (float)
        - gate_time (float)

    Returns:
        float: Drive amplitude at time t (0 <= t <= gate_time), else 0.
    """,
)

_set_doc(
    "drive_gauss_B",
    """Gaussian-enveloped cosine drive for qubit B. See `drive_gauss_A` for args.""",
)

_set_doc(
    "drag_A",
    """DRAG pulse for qubit A with Gaussian envelope and derivative quadrature.

    Args (from `args` dict):
        - drive_amp_A (float)
        - drive_freq_A (float)
        - gate_time (float)
        - alpha_A (float): DRAG coefficient.
    """,
)

_set_doc(
    "drag_B",
    """DRAG pulse for qubit B. See `drag_A` for details and argument names.""",
)

_set_doc(
    "make_power_spectrum",
    """Compute FFT magnitude spectrum and relative dB of a pulse.

    Args:
        pulse (np.ndarray): Time-domain samples.
        tlist (np.ndarray): Time points corresponding to samples.
        drive_freq (float): Expected carrier frequency (rad/s).

    Returns:
        tuple: (magnitude, freqs, peak_loc, ps_db)
    """,
)

_set_doc(
    "get_operator_two_zeropi",
    """Construct the coupled two Zero-Pi qubit model and return truncated dressed basis.

    Args:
        Ec0 (float): Charging energy to attach to ground/capacitive nodes.
        truc1 (int): Single-subsystem truncation (levels kept per qubit before coupling).
        truc_tot (int): Total dressed levels to keep after coupling diagonalization.
        charge_pick (bool): If True, select subspace via charge matrix elements.
        n_cut (int): Charge basis cutoff for subsystems.
        phi_cut (int): Phi grid size for subsystems.
        test (bool): Use smaller cutoffs for speed if True.

    Returns:
        list: [hspace_0, hspace_1, eval_tot, eket_tot]
    """,
)

_set_doc(
    "cz_fidelity_log",
    """Compute log10(1 - F) of CZ gate fidelity (ideal or noisy via args) using superoperator flow.

    Args:
        arg_all (list): [tg, drive_amp, detune, H_qbt_drive, W_target, num_cpus, c_op_list, logi_idx, option_ideal, option_noisy]

    Returns:
        float: log10(1 - fidelity)
    """,
)

_set_doc(
    "get_propagator",
    """Compute propagator for a quantum system, supporting ideal and noisy evolutions.

    Args:
        H: Static or time-dependent Hamiltonian (qutip Qobj or list format).
        tlist: Time grid.
        num_cpus: Parallel workers for SESolve/MESolve loops when applicable.
        c_op_list: Collapse operators. Empty for ideal evolution.
        pulse_args: Dict of time-dependent pulse parameters.
        option_ideal/option_noisy: qutip.Options objects for solvers.
        logi_idx: Indices of logical states to project/truncate.

    Returns:
        qutip.Qobj: Final-time propagator (ideal) or superoperator (noisy).
    """,
)

_set_doc(
    "get_fidelity_super_operator",
    """Average gate fidelity against a target gate from a (super)operator, handling phase corrections.

    Args:
        propagator: Ideal unitary or noisy superoperator at final time.
        logi_idx: Logical subspace indices.
        gate_target: qutip gate object (e.g., cz_gate(), cnot(), qt.sigmax()).
        c_op_list: Collapse operators; empty implies ideal.
        mid_state: Optional mid-state for CNOT phase correction helper.

    Returns:
        float: Average gate fidelity.
    """,
)


