"""This module contains the asymmetric two-qubit decomposition from
`Shende et al. <https://arxiv.org/abs/quant-ph/0308033>`__, adapted to our purposes
as described in App. A of `Kottmann et al. <https://arxiv.org/abs/unknown.id>`__ and
in Alg. 3 in particular.
"""

from functools import partial

import numpy as np
from pennylane import CNOT, RX, RZ, Y, matrix
from pennylane.math.decomposition import su2su2_to_tensor_products

# Matrices and matrix functions

_cnot = CNOT([0, 1]).matrix()
"""Matrix of a CNOT(0, 1) in its canonical wire ordering."""


def _ising_zz(theta):
    """Matrix of an RZ gate acting on the second of two wires."""
    return np.diag(np.exp([-0.5j * theta, 0.5j * theta, 0.5j * theta, -0.5j * theta]))


_yy = matrix(Y(0) @ Y(1), wire_order=[0, 1])
"""Matrix of the Pauli word "YY" acting on two wires."""

_magic_basis = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
"""Matrix that converts between the computational and the magic basis, as defined in the proof
of Prop. IV.3 in https://arxiv.org/pdf/quant-ph/0308033."""


def _gamma(u):
    """Compute complex relative structure for AI decomposition in magic basis rep."""
    # We will apply Y(0)@Y(1) by reordering columns and applying a sign diagonal.
    perm_yy = np.array([-1, 1, 1, -1])
    return (u[:, ::-1] * perm_yy @ u.T)[:, ::-1] * perm_yy


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

        \gamma\left(\mathrm{CNOT}(0,1) R_Z^{(1)}(\Psi) \mathrm{CNOT}(0,1) u \right)

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

    """
    # Compute Psi from the diagonal of γ(u.T).T
    t = np.diag(_gamma(u.conj()).T)
    psi = np.arctan2(np.sum(t).imag, (t[0] + t[3] - t[1] - t[2]).real)

    # Compute the LHS of the equation in Prop. V.2
    m = _gamma(u.conj().T @ _ising_zz(psi))

    # Compute its eigenvalues and the angles Theta and Phi from the args/angles of those eigvals
    evals = np.linalg.eigvals(m)
    theta, phi = _v2_angles(evals)
    return -psi, -theta, -phi


def _prop_iv3(u, v):
    r"""Given two two-qubit matrices :math:`U, V` that are guaranteed to be equal up to
    multiplication by single-qubit unitaries on either side, find the single-qubit
    unitaries :math:`a,b,c,d` such that

    .. math::

        U = (a\otimes b) V (c\otimes d).

    This method is described in the proof of Prop. IV.3 of
    `Shende et al. <https://arxiv.org/abs/quant-ph/0308033>`__.

    Args:
        u (np.ndarray): First two-qubit unitary matrix
        v (np.ndarray): Second two-qubit unitary matrix, must be equal to ``u`` up to single-qubit
            unitaries on both sides.

    Returns
        tuple[np.ndarray]: Single qubit unitaries :math:`a,b,c,d` such that the equation above is
        satisfied.

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
    _, a = np.linalg.eigh(pi_u)
    _, b = np.linalg.eigh(pi_v)
    # Fix determinants
    a[:, 0] *= np.linalg.det(a)
    b[:, 0] *= np.linalg.det(b)
    c = new_v.conj().T @ b @ a.T @ new_u

    # Move the subgroup elements a @ b.T and c from the standard rep of SO(4)
    # back to the standard rep of SU(2)xSU(2), and decompose the result into single-qubit ops.
    left_su2_su2 = _magic_basis @ a @ b.T @ _magic_basis.conj().T
    a, b = su2su2_to_tensor_products(left_su2_su2)
    right_su2_su2 = _magic_basis @ c @ _magic_basis.conj().T
    c, d = su2su2_to_tensor_products(right_su2_su2)

    return a, b, c, d


@partial(matrix, wire_order=[0, 1])
def _core_mat(theta, phi):
    """Compute the combined matrix of
    ``CNOT([0, 1]) @ RX(theta, 0) @ RZ(phi, 1) @ CNOT([0, 1])`` w.r.t. wire
    ordering ``[0, 1]``."""
    CNOT([0, 1])
    RX(theta, 0)
    RZ(phi, 1)
    CNOT([0, 1])


def asymmetric_decomp(v: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Compute the asymmetric decomposition of a two-qubit unitary matrix from
    `Shende et al. <https://arxiv.org/abs/quant-ph/0308033>`__, adapted to our purposes
    as described in App. A of `Kottmann et al. <https://arxiv.org/abs/unknown.id>`__ and
    in Alg. 3 in particular.

    Args:
        v (np.ndarray): Two-qubit unitary matrix to decompose

    Returns:
        tuple[list[np.ndarray], float]: Numerical data of the decomposition, as described in Alg. 3
        of Kottmann et al.

    """
    assert v.shape == (4, 4), f"{v.shape=}, expected (4, 4)"
    v_mod = _cnot @ v
    alpha = np.angle(np.linalg.det(v_mod)) / 4
    v_mod = np.exp(-1j * alpha) * v_mod

    psi, theta, phi = _prop_v2(v_mod)
    v_prime = _ising_zz(psi) @ v_mod
    w = _core_mat(theta, phi)
    single_qubit_unitaries = _prop_iv3(v_prime, w)

    return single_qubit_unitaries, alpha, -psi, theta, phi
