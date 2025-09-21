from functools import partial
import numpy as np
import pennylane as qml
from pennylane.math.decomposition import su2su2_to_tensor_products
from .validation import is_unitary, has_unit_determinant, is_symmetric, validation_enabled, is_orthogonal

# Matrices and matrix functions

_CNOT = qml.CNOT([0, 1]).matrix()
_RZ = qml.matrix(partial(qml.RZ, wires=1), wire_order=[0, 1])
_YY = qml.matrix(qml.Y(0) @ qml.Y(1), wire_order=[0, 1])
_E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)

@partial(qml.matrix, wire_order=[0, 1])
def _RX_RZ(theta, phi):
    qml.RX(theta, 0)
    qml.RZ(phi, 1)

def _gamma(u):
    """Compute complex relative structure for AI decomposition in magic basis rep."""
    return u @ _YY @ u.T @ _YY

def _complex_sort(x):
    """Sort (and modify!) a complex-valued array by sorting Re(x) + 10⁵ Im(x)."""
    return np.sort(x.real + x.imag * 1e5)

def _prop_V2(u):
    """Implement Proposition V.2 from https://arxiv.org/pdf/quant-ph/0308033, namely
    find three angles Psi, Theta and Phi such that the eigenvalues of

    γ(u CNOT(0,1) RZ(Psi, 1) CNOT(0,1))

    and

    γ(CNOT(0,1) RX(Theta, 0) RZ(Phi, 1) CNOT(0,1))

    are equal.
    """
    t = np.diag(_gamma(u.T).T)
    #print(np.sum(t).imag, (t[0]+t[3]-t[1]-t[2]).real)
    Psi = np.arctan2(np.sum(t).imag, (t[0]+t[3]-t[1]-t[2]).real)
    m = _gamma(u @ _CNOT @ _RZ(Psi) @ _CNOT)
    evals = np.linalg.eigvals(m)
    eval_angles = np.sort(np.angle(evals))
    r, s = eval_angles[2:]
    Theta = (r+s)/2
    Phi = (r-s) / 2

    if validation_enabled():
        assert has_unit_determinant(m)
        assert is_unitary(m)
        assert np.isclose(np.trace(m).imag, 0.)
        # "negative"
        negative_angles = eval_angles[:2]
        comp_neg_to_pos_angles = -(np.mod(negative_angles+np.pi+1e-10, 2 * np.pi)-np.pi-1e-10)
        # todo: Reactivate test once the precise condition is understood
        assert np.allclose(comp_neg_to_pos_angles, eval_angles[3:1:-1]), f"First two angles mod 2π, times -1: {comp_neg_to_pos_angles}\nSecond pair of angles: {eval_angles[3:1:-1]}"

        left = _complex_sort(np.linalg.eigvals(m))
        right = _complex_sort(np.linalg.eigvals(_gamma(_CNOT @ _RX_RZ(Theta, Phi) @ _CNOT)))
        assert np.allclose(left, right), f"{left=}\n{right=}"

    return Psi, Theta, Phi

def _prop_IV3(u, v):
    """Given two matrices that are guaranteed to be equal up to conjugation
    by single-qubit unitaries, find the single-qubit unitaries ``a,b,c,d``
    such that

    ``u = kron(a, b) @ v @ kron(c, d)``.
    """
    # Move u and v to new representation of unitary group. This is necessary
    # because we want to move the AI subgroup SO(4) back to the standard rep of SU(2)xSU(2)
    # later on by undoing this representation change.
    new_u = _E.conj().T @ u @ _E
    new_v = _E.conj().T @ v @ _E
    # Compute relative complex structures for AI decomposition in its standard rep.
    delta_u = new_u @ new_u.T
    delta_v = new_v @ new_v.T

    # Rescaled real and imaginary parts for real-valued diagonalization of u and v
    pi_u = delta_u.real*np.pi**2 + delta_u.imag /np.pi**2
    pi_v = delta_v.real*np.pi**2 + delta_v.imag /np.pi**2
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
    left_su2_su2 = _E @ a @ b.T @ _E.conj().T
    a, b = su2su2_to_tensor_products(left_su2_su2)
    right_su2_su2 = _E @ c @ _E.conj().T
    c, d = su2su2_to_tensor_products(right_su2_su2)

    if validation_enabled():
        assert np.allclose(np.kron(a, b), left_su2_su2), f"\n{np.kron(a, b)=}\n{left_su2_su2=}"
        assert np.allclose(np.kron(c, d), right_su2_su2)
        assert np.allclose(np.kron(a, b) @ v @ np.kron(c, d), u)
    return a, b, c, d


def asymmetric_two_qubit_decomp(u, wires):
    """Compute a 3-CNOT decomposition of a two-qubit unitary that is less symmetric
    than the standard decomposition beginning and ending with full layers of single-qubit
    rotations, in terms of the location of the used single-parameter one-qubit blocks.
    This decomposition is reported in Theorem VI.3 and Fig. 3 of
    https://arxiv.org/pdf/quant-ph/0308033."""
    if validation_enabled():
        assert u.shape == (4, 4)
        assert is_unitary(u)
        assert len(wires) == 2
    u_mod = u @ _CNOT
    gphase = np.angle(np.linalg.det(u_mod))/4
    u_mod = np.exp(-1j * gphase) * u_mod

    with qml.queuing.QueuingManager.stop_recording():
        Psi, Theta, Phi = _prop_V2(u_mod)
        U = u_mod @ _CNOT @ _RZ(Psi) @ _CNOT
        V = _CNOT @ _RX_RZ(Theta, Phi) @ _CNOT
        a, b, c, d = _prop_IV3(U, V)
        if validation_enabled():
            assert np.allclose(np.kron(a, b) @ V @ np.kron(c, d), U)
            assert np.allclose(u_mod, np.kron(a, b) @ V @ np.kron(c, d) @ _CNOT @ _RZ(-Psi) @ _CNOT)

    ops = [
        qml.RZ(-Psi, wires[1]),
        qml.CNOT(wires),
        qml.QubitUnitary(c, wires[0]),
        qml.QubitUnitary(d, wires[1]),
        qml.CNOT(wires),
        qml.RX(Theta, wires[0]),
        qml.RZ(Phi, wires[1]),
        qml.CNOT(wires),
        qml.QubitUnitary(a, wires[0]),
        qml.QubitUnitary(b, wires[1]),
        qml.GlobalPhase(-gphase),
    ]
    if validation_enabled():
        u_rec = qml.matrix(qml.tape.QuantumScript(ops), wire_order=wires)
        assert np.allclose(u, u_rec)
    return ops



