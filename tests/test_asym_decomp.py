import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros
from rotoptsynth.asym_decomp import _cnot, _rz_1, _yy, _rx_rz, _gamma

some_ising_xy = qml.IsingXY(0.841, [0, 1]).matrix()
some_ising_zz = qml.IsingZZ(0.7235, [0, 1]).matrix()


targets = [
    np.eye(4),
    1j * np.eye(4),
    _cnot,
    qml.X(0).matrix([1, 0]),
    qml.X(1).matrix([1, 0]),
    some_ising_zz,
    some_ising_xy,
    unitary_group.rvs(4, random_state=951210),
    unitary_group.rvs(4, random_state=835610),
    unitary_group.rvs(4, random_state=62138),
    unitary_group.rvs(4, random_state=1561),
    qml.Rot(0.12, 2.1, -0.7, 0).matrix([1, 0]),
]


class TestTiny:
    """Test that hardcoded data and small utility functions work as expected."""

    def test_matrices(self):
        """Test hardcoded matrices."""
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        assert np.allclose(_cnot, cnot)

        for theta in np.linspace(-np.pi, np.pi, num=13):
            rz = _rz_1(theta)
            diag = np.diag(rz)
            assert np.allclose(np.diag(diag), rz)
            assert np.allclose(diag[:2], diag[2:])

        yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        assert np.allclose(_yy, yy)

        _x = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
        for theta in np.linspace(-np.pi, np.pi, num=7):
            for phi in np.linspace(-np.pi, np.pi, num=7):
                mat = _rx_rz(theta, phi)
                rec = _rz_1(phi) @ (np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * _x)
                assert np.allclose(mat, rec)

    @pytest.mark.parametrize(
        "u, expected",
        [
            (_rx_rz(0.6, -2.82), np.eye(4)),
            (_cnot, -(qml.Z(0) @ qml.X(1)).matrix()),
            (some_ising_xy, some_ising_xy @ some_ising_xy),
            (some_ising_zz, some_ising_zz @ some_ising_zz),
            (_rz_1(0.1241), np.eye(4)),
        ],
    )
    def test_gamma(self, u, expected):
        assert np.allclose(_gamma(u), expected)


class TestAsymmetricTwoQubitDecomp:
    """Tests for asymmetric_two_qubit_decomp."""

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets)
    def test_builtin_validation(self, target):
        ops = ros.asymmetric_two_qubit_decomp(target, [0, 1])
        assert len(ops) == 11
        assert sum(isinstance(op, qml.CNOT) for op in ops) == 3
        assert ros.count_rotation_angles(ops) == 16

    @pytest.mark.without_validation
    @pytest.mark.parametrize("wires", [(1, 0), (0, 1), ("a", 5)])
    @pytest.mark.parametrize("target", targets)
    def test_wires(self, target, wires):
        ops = ros.asymmetric_two_qubit_decomp(target, wires)
        assert all(set(op.wires).issubset(set(wires)) for op in ops)

    @pytest.mark.without_validation
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def queuing_matches_return(self, n):
        assert not ros.validation_enabled()
        target = unitary_group.rvs(2**n, random_state=8152)
        with qml.queuing.AnnotatedQueue() as q:
            ops = ros.asymmetric_two_qubit_decomp(target, range(n))

        assert ops == q.queue

    @pytest.mark.with_validation
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def queuing_matches_return_with_validation(self, n):
        assert not ros.validation_enabled()
        target = unitary_group.rvs(2**n, random_state=8152)
        with qml.queuing.AnnotatedQueue() as q:
            ops = ros.asymmetric_two_qubit_decomp(target, range(n))

        assert ops == q.queue
