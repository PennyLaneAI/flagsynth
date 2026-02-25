"""Tests for rotoptsynth/linalg.py"""
import pytest
import numpy as np
from scipy.stats import unitary_group, special_ortho_group
import pennylane as qml
from rotoptsynth.linalg import csd, de_mux, balance_diagonal, mottonen
# pylint: disable=too-few-public-methods


class TestCsd:
    """Tests for cosine-sine decomposition in `csd`."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_unitary(self, n):
        """Test basic usage with unitary matrices."""

        N = 2**n
        V = unitary_group.rvs(N, random_state=8214)
        K00, K01, theta_Y, K10, K11 = csd(V)

        for K in [K00, K01, K10, K11]:
            assert K.shape == (N // 2, N // 2)
            assert np.allclose(K @ K.conj().T, np.eye(N // 2))

        zeros = np.zeros((N // 2, N // 2))
        K0 = np.block([[K00, zeros], [zeros, K01]])
        K1 = np.block([[K10, zeros], [zeros, K11]])
        A_op = qml.SelectPauliRot(theta_Y, control_wires=range(1, n), target_wire=0, rot_axis="Y")
        A = qml.matrix(A_op, wire_order=range(n))
        assert np.allclose(K0 @ A @ K1, V)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered"
    )  # special_ortho_group.rvs sometimes warns
    @pytest.mark.filterwarnings(
        "ignore:overflow encountered"
    )  # special_ortho_group.rvs sometimes warns
    @pytest.mark.filterwarnings("ignore:divide by zero")  # special_ortho_group.rvs sometimes warns
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_orthogonal(self, n):
        """Test basic usage with orthogonal matrices."""

        N = 2**n
        V = special_ortho_group.rvs(N, random_state=8214)
        K00, K01, theta_Y, K10, K11 = csd(V)

        for K in [K00, K01, K10, K11]:
            assert K.shape == (N // 2, N // 2)
            assert K.dtype == np.float64
            assert np.allclose(K @ K.T, np.eye(N // 2))

        zeros = np.zeros((N // 2, N // 2))
        K0 = np.block([[K00, zeros], [zeros, K01]])
        K1 = np.block([[K10, zeros], [zeros, K11]])
        A_op = qml.SelectPauliRot(theta_Y, control_wires=range(1, n), target_wire=0, rot_axis="Y")
        A = qml.matrix(A_op, wire_order=range(n)).real
        assert np.allclose(K0 @ A @ K1, V)


class TestDeMux:
    """Tests for demultiplexing via `de_mux`."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_basic(self, n):
        """Test basic usage."""
        N = 2**n
        K0, K1 = unitary_group.rvs(N, size=2, random_state=814)
        M0, theta_Z, M1 = de_mux(K0, K1)

        for M in [M0, M1]:
            assert M.shape == (N, N)
            assert np.allclose(M @ M.conj().T, np.eye(N))

        zeros = np.zeros((N, N))
        M0 = np.block([[M0, zeros], [zeros, M0]])
        M1 = np.block([[M1, zeros], [zeros, M1]])
        A_op = qml.SelectPauliRot(
            theta_Z, control_wires=range(1, n + 1), target_wire=0, rot_axis="Z"
        )
        A = qml.matrix(A_op, wire_order=range(n + 1))
        V = np.block([[K0, zeros], [zeros, K1]])
        assert np.allclose(M0 @ A @ M1, V)


class TestBalanceDiagonal:
    """Tests for `balance_diagonal`."""

    @pytest.mark.parametrize("n, target_wire", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
    def test_basic(self, n, target_wire):
        """Test basic usage of the function."""
        N = 2**n
        np.random.seed(512581)
        Delta = np.exp(1j * np.random.random(N))
        theta_Z, Delta_prime = balance_diagonal(Delta, target_wire)
        ctrls = [i for i in range(n) if i != target_wire]
        rz_op = qml.SelectPauliRot(theta_Z, ctrls, target_wire, rot_axis="Z")
        rz_mat = np.diag(qml.matrix(rz_op, wire_order=range(n)))
        if n == 1:
            diag_op_mat = Delta_prime
        else:
            diag_op = qml.DiagonalQubitUnitary(Delta_prime, wires=ctrls)
            diag_op_mat = np.diag(qml.matrix(diag_op, wire_order=range(n)))
        assert np.allclose(Delta, rz_mat * diag_op_mat)


class TestMottonen:
    """Tests for multiplexed rotations (aka `qml.SelectPauliRot`) decomposition by
    Mottonen et al."""

    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("axis", "YZ")
    def test_basic(self, seed, k, axis):
        """Test basic usage."""
        np.random.seed(seed)
        theta = np.random.random(2**k)

        with qml.queuing.AnnotatedQueue() as q:
            decomp = mottonen(theta, controls=list(range(k)), target=k, axis=axis)

        assert len(q.queue) == 0
        decomp_mat = qml.matrix(decomp, wire_order=range(k + 1))

        individual_mats = [getattr(qml, f"R{axis}")(th, 0).matrix() for th in theta]
        expected = qml.math.block_diag(individual_mats)
        assert np.allclose(expected, decomp_mat)
