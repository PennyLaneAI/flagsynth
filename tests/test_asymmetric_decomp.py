"""Tests for flagsynth/asymmetric_decomp.py"""

from functools import partial
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from flagsynth.asymmetric_decomp import _prop_v2, _gamma, _prop_iv3, asymmetric_decomp

# pylint: disable=too-few-public-methods


def make_unit_det(mat):
    """Transform a matrix to have unit determinant."""
    gphase = np.angle(np.linalg.det(mat)) / len(mat)
    mat = np.exp(-1j * gphase) * mat
    assert np.isclose(np.linalg.det(mat), 1)
    return mat


class TestPropositions:
    """Tests for the proposition implementations used in `asymmetric_decomp`."""

    @pytest.mark.parametrize("seed", [194, 185, 824, 72])
    def test_prop_v2(self, seed):
        """Test Proposition V.2, implemented in `_prop_v2`."""
        # Make random special unitary
        u = make_unit_det(unitary_group.rvs(4, random_state=seed))

        with qml.queuing.AnnotatedQueue() as q:
            psi, theta, phi = _prop_v2(u)

        assert len(q.queue) == 0

        lhs = (
            qml.CNOT([0, 1]) @ qml.RZ(psi, 1) @ qml.CNOT([0, 1]) @ qml.QubitUnitary(u, wires=[0, 1])
        )
        rhs = qml.CNOT([0, 1]) @ qml.RX(theta, 0) @ qml.RZ(phi, 1) @ qml.CNOT([0, 1])
        lhs = _gamma(qml.matrix(lhs, wire_order=[0, 1]))
        rhs = _gamma(qml.matrix(rhs, wire_order=[0, 1]))
        lhs_eig = np.linalg.eigvals(lhs)
        rhs_eig = np.linalg.eigvals(rhs)
        sorted_lhs_eig = np.sort(lhs_eig.real + lhs_eig.imag * 1e5)
        sorted_rhs_eig = np.sort(rhs_eig.real + rhs_eig.imag * 1e5)
        assert np.allclose(sorted_lhs_eig, sorted_rhs_eig)

    @pytest.mark.parametrize("seed", [19411, 1285, 84, 7212])
    def test_prop_iv3(self, seed):
        """Test Proposition IV.3, implemented in `_prop_iv3`."""
        v = make_unit_det(unitary_group.rvs(4, random_state=seed))
        abcd = unitary_group.rvs(2, size=4, random_state=seed + 2)
        a_in, b_in, c_in, d_in = [make_unit_det(mat) for mat in abcd]
        u = np.kron(a_in, b_in) @ v @ np.kron(c_in, d_in)

        with qml.queuing.AnnotatedQueue() as q:
            a, b, c, d = _prop_iv3(u, v)
        assert len(q.queue) == 0

        u_rec = np.kron(a, b) @ v @ np.kron(c, d)
        assert np.allclose(u_rec, u)


@partial(qml.matrix, wire_order=[0, 1])
def asymmetric_circuit(one_qubit_unitaries, alpha, psi, theta, phi):
    """Construct PennyLane circuit from data computed by `asymmetric_decomp`.
    Together with the decorator qml.matrix, this computes the implemented matrix."""
    a, b, c, d = one_qubit_unitaries
    qml.QubitUnitary(c, 0)
    qml.QubitUnitary(d, 1)
    qml.CNOT([0, 1])
    qml.RX(theta, 0)
    qml.RZ(phi, 1)
    qml.CNOT([0, 1])
    qml.QubitUnitary(a, 0)
    qml.QubitUnitary(b, 1)
    qml.CNOT([0, 1])
    qml.RZ(psi, 1)
    qml.GlobalPhase(-alpha)  # Differing convention in PennyLane for global phase


class TestAsymmetricDecomp:
    """Tests for `asymmetric_decomp` itself."""

    @pytest.mark.parametrize("seed", [141, 125285, 21484, 722])
    def test_basic(self, seed):
        """Test standard usage of `asymmetric_decomp`."""
        v = unitary_group.rvs(4, random_state=seed)

        with qml.queuing.AnnotatedQueue() as q:
            data = asymmetric_decomp(v)
        assert len(q.queue) == 0

        assert np.allclose(asymmetric_circuit(*data), v)
