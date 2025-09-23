"""This module implements a specific two-qubit unitary decomposition from Theorem VI.3
and Fig. 3 of https://arxiv.org/pdf/quant-ph/0308033, in ``asymmetric_two_qubit_decomp``.
It is used in the up-to-diagonal decomposition ``diag_decomp``.
"""

# pylint: disable=too-many-locals
from functools import partial

import numpy as np
import pennylane as qml
from pennylane.math.decomposition import su2su2_to_tensor_products

from .utils import ops_to_mat
from .validation import (
    has_unit_determinant,
    is_orthogonal,
    is_symmetric,
    is_unitary,
    validation_enabled,
)

# Matrices and matrix functions

_cnot = qml.CNOT([0, 1]).matrix()
"""Matrix of a CNOT(0, 1) in its canonical wire ordering."""


def _rz_1(theta):
    return np.diag(np.exp([-0.5j * theta, 0.5j * theta, -0.5j * theta, 0.5j * theta]))


"""Matrix of an RZ gate acting on the second of two wires."""

_yy = qml.matrix(qml.Y(0) @ qml.Y(1), wire_order=[0, 1])
"""Matrix of the Pauli word "YY" acting on two wires."""

_magic_basis = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
"""Matrix that converts between the computational and the magic basis, as defined in the proof
of Prop. IV.3 in https://arxiv.org/pdf/quant-ph/0308033."""


@partial(qml.matrix, wire_order=[0, 1])
def _rx_rz(theta, phi):
    """Compute the combined matrix of ``RX(theta, 0) @ RZ(phi, 1)`` w.r.t. wire
    ordering ``[0, 1]``."""
    qml.RX(theta, 0)
    qml.RZ(phi, 1)


def _gamma(u):
    """Compute complex relative structure for AI decomposition in magic basis rep."""
    return u @ _yy @ u.T @ _yy


def _complex_sort(x):
    r"""Sort (and modify!) a complex-valued array by sorting Re(x) + 10⁵ Im(x)."""
    return np.sort(x.real + x.imag * 1e5)


def _v2_angles(evals):
    r"""This is a helper function for _prop_v2, implementing a step in the proof of
    Proposition V.2 from https://arxiv.org/pdf/quant-ph/0308033.
    Concretely, given the eigenvalues of a special unitary ``m`` in four dimensions,
    we know that they come in two complex conjugate pairs. This function finds one representative
    of each pair, called ``r`` and ``s`` respectively,  by looking at the absolute values of
    the angles of the eigenvalues. It then returns ``(r+s)/2`` and ``(r-s)/2``, the angles
    :math:`\Theta` and :math:`\Phi` in the proof of Prop. V.2.

    Args:
        evals (np.ndarray): Eigenvalues of special unitary in four dimensions.

    Returns:
        tuple[float]: Angles :math:`\Theta` and :math:`\Phi` as defined in the proof of Prop. V.2,
        computed from the unique absolute values of the eigenvalue angles.

    Raises:
        ValueError: If the absolute value of the angle of the first eigenvalue does not match
        any of the other absolute values. According to the proof, the eigenvalues must come in
        complex conjugate pairs, so that this scenario should never occur for valid inputs,
        up to purely numerical problems.

    """
    angles = np.angle(evals)
    abs_angles = np.abs(angles)
    if np.isclose(abs_angles[0], abs_angles[1]):
        r = abs_angles[0]
        s = abs_angles[2]
    elif np.isclose(abs_angles[0], abs_angles[2]) or np.isclose(abs_angles[0], abs_angles[3]):
        r = abs_angles[0]
        s = abs_angles[1]
    else:
        raise ValueError("Expected eigenvalues to come in complex conjugate pairs.")

    return (r + s) / 2, (r - s) / 2


def _prop_v2(u):
    r"""Implement Proposition V.2 from https://arxiv.org/pdf/quant-ph/0308033, namely
    find three angles :math:`\Psi`, :math:`\Theta` and :math:`\Phi` such that the eigenvalues of

    .. math::

        \gamma\left(u \mathrm{CNOT}(0,1) R_Z^{(1)}(\Psi) \mathrm{CNOT}(0,1)\right)

    and

    .. math::

        \gamma\left(\mathrm{CNOT}(0,1) R_X^{(0)}(\Theta) R_Z^{(1)}(\Phi) \mathrm{CNOT}(0,1)\right)

    are equal. Here, :math:`\gamma` denotes the function defined in Def. IV.1 of
    the reference above, which computes the relative complex structure of its input with
    respect to the generic AI Cartan involution in the magic basis (see ``_gamma``).

    Args:
        u (np.ndarray): Matrix to generate the angles for.

    Returns:
        tuple[float]: Angles :math:`\Psi, \Theta, \Phi` as described in Prop. V.2.

    Uses the RotOptSynth validation toggle.
    """
    # Compute Psi from the diagonal of γ(u.T).T
    t = np.diag(_gamma(u.T).T)
    psi = np.arctan2(np.sum(t).imag, (t[0] + t[3] - t[1] - t[2]).real)

    # Compute the LHS of the equation in Prop. V.2
    m = _gamma(u @ _cnot @ _rz_1(psi) @ _cnot)

    # Compute its eigenvalues and the angles Theta and Phi from the args/angles of those eigvals
    evals = np.linalg.eigvals(m)
    theta, phi = _v2_angles(evals)

    if validation_enabled():
        assert has_unit_determinant(m)
        assert is_unitary(m)
        assert np.isclose(np.trace(m).imag, 0.0)
        left = _complex_sort(np.linalg.eigvals(m))
        right = _complex_sort(np.linalg.eigvals(_gamma(_cnot @ _rx_rz(theta, phi) @ _cnot)))
        assert np.allclose(left, right), f"{left=}\n{right=}"

    return psi, theta, phi


def _prop_iv3(u, v):
    r"""Given two two-qubit matrices :math:`U, V` that are guaranteed to be equal up to
    multiplication by single-qubit unitaries on either side, find the single-qubit
    unitaries :math:`a,b,c,d` such that

    .. math::

        U = (a\otimes b) V (c\otimes d).

    This method is described in the proof of Prop. IV.3 of https://arxiv.org/pdf/quant-ph/0308033.

    Args:
        u (np.ndarray): First two-qubit unitary matrix
        v (np.ndarray): Second two-qubit unitary matrix, must be equal to ``u`` up to single-qubit
            unitaries on both sides.

    Returns
        tuple[np.ndarray]: Single qubit unitaries :math:`a,b,c,d` such that the equation above is
        satisfied.

    Uses the RotOptSynth validation toggle.
    """
    # Move u and v to new representation of unitary group. This is necessary
    # because we want to move the AI subgroup SO(4) back to the standard rep of SU(2)xSU(2)
    # later on by undoing this representation change.
    new_u = _magic_basis.conj().T @ u @ _magic_basis
    new_v = _magic_basis.conj().T @ v @ _magic_basis
    # Compute relative complex structures for AI decomposition in its standard rep.
    delta_u = new_u @ new_u.T
    delta_v = new_v @ new_v.T

    # Rescaled real and imaginary parts for real-valued diagonalization of u and v
    pi_u = delta_u.real * np.pi**2 + delta_u.imag / np.pi**2
    pi_v = delta_v.real * np.pi**2 + delta_v.imag / np.pi**2
    # Compute real-valued diagonalization
    eigvals_u, a = np.linalg.eigh(pi_u)
    eigvals_v, b = np.linalg.eigh(pi_v)
    # Fix determinants
    a[:, 0] *= np.linalg.det(a)
    b[:, 0] *= np.linalg.det(b)
    c = new_v.conj().T @ b @ a.T @ new_u

    if validation_enabled():
        assert is_symmetric(pi_u)
        assert is_symmetric(pi_v)
        assert np.allclose(eigvals_u, eigvals_v)
        assert np.allclose(a.T @ pi_u @ a, np.diag(eigvals_u))
        assert np.allclose(b.T @ pi_v @ b, np.diag(eigvals_v))
        assert is_orthogonal(a @ b.T)
        assert is_orthogonal(c)

    # Move the subgroup elements a @ b.T and c from the standard rep of SO(4)
    # back to the standard rep of SU(2)xSU(2), and decompose the result into single-qubit ops.
    left_su2_su2 = _magic_basis @ a @ b.T @ _magic_basis.conj().T
    a, b = su2su2_to_tensor_products(left_su2_su2)
    right_su2_su2 = _magic_basis @ c @ _magic_basis.conj().T
    c, d = su2su2_to_tensor_products(right_su2_su2)

    if validation_enabled():
        assert np.allclose(np.kron(a, b), left_su2_su2), f"\n{np.kron(a, b)=}\n{left_su2_su2=}"
        assert np.allclose(np.kron(c, d), right_su2_su2)
        assert np.allclose(np.kron(a, b) @ v @ np.kron(c, d), u)
    return a, b, c, d


def asymmetric_two_qubit_decomp(u, wires):
    """Compute a 3-CNOT decomposition of a two-qubit unitary of the following form:

    ```
    0: ───────────╭●──U(M0)─╭●──RX(1.36)─╭●──U(M2)─╭GlobalPhase(-0.52)─┤
    1: ──RZ(2.00)─╰X──U(M1)─╰X──RZ(0.97)─╰X──U(M3)─╰GlobalPhase(-0.52)─┤,
    ```

    where ``U`` denote arbitrary single qubit rotations.

    It is less symmetric than the standard decomposition that begins and ends with full layers
    of single-qubit rotations, in terms of the used single-parameter one-qubit blocks.
    This decomposition is reported in Theorem VI.3 and Fig. 3 of
    https://arxiv.org/pdf/quant-ph/0308033.

    Args:
        u (np.ndarray): Two-qubit unitary matrix to decompose.
        wires (Sequence): Two wires on which the operators in the unitary decomposition act

    Returns:
        list[qml.operation.Operator]: Operators in the decomposition

    Queues:
        Same operators in the decomposition that are returned.

    Uses the RotOptSynth validation toggle.
    """
    if validation_enabled():
        assert u.shape == (4, 4)
        assert is_unitary(u)
        assert len(wires) == 2
    u_mod = u @ _cnot
    gphase = np.angle(np.linalg.det(u_mod)) / 4
    u_mod = np.exp(-1j * gphase) * u_mod

    with qml.queuing.QueuingManager.stop_recording():
        psi, theta, phi = _prop_v2(u_mod)
        _u = u_mod @ _cnot @ _rz_1(psi) @ _cnot
        v = _cnot @ _rx_rz(theta, phi) @ _cnot
        a, b, c, d = _prop_iv3(_u, v)
        if validation_enabled():
            assert np.allclose(np.kron(a, b) @ v @ np.kron(c, d), _u)
            assert np.allclose(
                u_mod, np.kron(a, b) @ v @ np.kron(c, d) @ _cnot @ _rz_1(-psi) @ _cnot
            )

    ops = [
        qml.RZ(-psi, wires[1]),
        qml.CNOT(wires),
        qml.QubitUnitary(c, wires[0]),
        qml.QubitUnitary(d, wires[1]),
        qml.CNOT(wires),
        qml.RX(theta, wires[0]),
        qml.RZ(phi, wires[1]),
        qml.CNOT(wires),
        qml.QubitUnitary(a, wires[0]),
        qml.QubitUnitary(b, wires[1]),
        qml.GlobalPhase(-gphase),
    ]
    if validation_enabled():
        u_rec = ops_to_mat(ops, wires)
        assert np.allclose(u, u_rec)
    return ops
