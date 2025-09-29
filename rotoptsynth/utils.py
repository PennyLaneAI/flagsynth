"""This module contains utility functions to convert operators to matrices, perform
cosine-sine-decompositions of matrices, and to count (non-)Clifford angles."""

import numpy as np
import pennylane as qml
from scipy.linalg import cossin

from .validation import has_unit_determinant
from .select_su2 import SelectSU2


def ops_to_mat(ops, wire_order):
    """Convert a list of operators to their matrix, with respect to the given ``wire_order``."""
    return qml.matrix(qml.tape.QuantumScript(ops), wire_order=wire_order)


def aiii_kak(u, p, q, validate):
    """Decompose a matrix using the cosine-sine-decomposition (CSD) implemented in scipy.
    Wraps ``scipy.linalg.cossin`` with some validation and a convention adapter."""
    if validate:
        assert u.shape == (p + q, p + q)
    if p == 0 or q == 0:
        return u, np.eye(p + q), np.eye(p + q)
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(u, p=p, q=p, swap_sign=True, separate=False)

    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)
        f[:, :p] = np.roll(f[:, :p], q - p, axis=1)
        f[:p] = np.roll(f[:p], q - p, axis=0)

    if validate:
        dim = u.shape[0]
        r = min(p, q)
        s = max(p, q)
        assert np.allclose(k1[:p, p:], 0.0) and np.allclose(k1[p:, :p], 0.0)
        assert np.allclose(k1 @ k1.conj().T, np.eye(dim))
        assert np.allclose(k2[:p, p:], 0.0) and np.allclose(k2[p:, :p], 0.0)
        assert np.allclose(k2 @ k2.conj().T, np.eye(dim))

        assert np.allclose(f[r:s, r:s], np.eye(s - r))
        assert np.allclose(np.diag(np.diag(f[:r, :r])), f[:r, :r])
        assert np.allclose(f[:r, :r], f[s:, s:])
        assert np.allclose(np.diag(np.diag(f[:r, s:])), f[:r, s:])
        assert np.allclose(f[:r, s:], -f[s:, :r])
        assert np.allclose(f[r:s, :r], 0.0)
        assert np.allclose(f[r:s, s:], 0.0)
        assert np.allclose(f[:r, r:s], 0.0)
        assert np.allclose(f[s:, r:s], 0.0)
        assert np.allclose(k1 @ f @ k2, u), f"\n{k1}\n{f}\n{k2}\n{k1 @ f @ k2}\n{u}"

    return k1, f, k2


def count_clifford(theta, verbose=False, atol=1e-6):
    r"""
    Function to compute Clifford, non-Clifford and zero angles from rotation angles

    Args:
        theta (TensorLike): Rotation angles
        verbose (bool): Whether or not to output the angles and their assignments for debugging
            and sanity-checking
        atol (float): absolute tolerance when comparing values to Clifford angles and zeros

    **Example**

    >>> theta = np.array([np.pi, np.pi/2, np.pi/4, 1e-16])
    >>> cliffs, non_cliffs, zeros = count_clifford(theta, verbose=True)
    Clifford: 2.0
    Clifford: 1.0
    non-Clifford: 0.5
    Zero: 0.0
    >>> cliffs, non_cliffs, zeros
    (2, 1, 1)

    """
    params = np.mod(theta + 2 * np.pi, 2 * np.pi)
    params = params / (np.pi / 2)

    cliffs = 0
    non_cliffs = 0
    zeros = 0

    for p in params:
        if any(np.isclose(p, val, atol=atol) for val in [0.0, 4.0]):
            if verbose:
                print(f"Zero: {p}")
            zeros += 1
        elif any(np.isclose(p, val, atol=atol) for val in [1.0, 2.0, 3.0]):
            if verbose:
                print(f"Clifford: {p}")
            cliffs += 1
        else:
            if verbose:
                print(f"non-Clifford: {p}")
            non_cliffs += 1

    return (cliffs, non_cliffs, zeros)


_rotation_counts = {
    qml.RX: 1,
    qml.RY: 1,
    qml.RZ: 1,
    qml.CNOT: 0,
    qml.CZ: 0,
    qml.QubitUnitary: lambda op: 4 ** len(op.wires) - 1,  # assumes unit determinant
    qml.SelectPauliRot: lambda op: 2 ** (len(op.wires) - 1),
    SelectSU2: lambda op: 3 * 2 ** (len(op.wires) - 1),
    qml.GlobalPhase: 1,
    qml.DiagonalQubitUnitary: lambda op: 2 ** (len(op.wires)),
}


def count_rotation_angles(ops):
    """Count how many rotation angles parametrize a give sequence of operators."""
    assert all(has_unit_determinant(op.data[0]) for op in ops if isinstance(op, qml.QubitUnitary))
    return sum(
        (entry(op) if callable(entry := _rotation_counts[type(op)]) else entry) for op in ops
    )


_cnot_counts = {
    qml.CNOT: 1,
    qml.CZ: 1,
    qml.SelectPauliRot: lambda op: 2 ** (len(op.wires) - 1),
    SelectSU2: lambda op: 3 * 2 ** (len(op.wires) - 1),
    qml.DiagonalQubitUnitary: lambda op: 2 ** len(op.wires) - 2,
}


def count_cnots(ops):
    """Count how many rotation angles parametrize a give sequence of operators."""
    return sum((entry(op) if callable(entry := _cnot_counts[type(op)]) else entry) for op in ops)
