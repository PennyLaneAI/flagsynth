from functools import partial
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from pennylane.ops.functions import assert_valid
from pennylane.wires import Wires

from rotoptsynth.recursive_flag_decomp import one_qubit_flag_decomp, two_qubit_flag_decomp, MultiplexedFlag, mux_ops, _decompose_mux_single_qubit_flag

class TestOneQubitFlagDecomp:
    """Tests for the one-qubit flag decomposition in `one_qubit_flag_decomp`."""

    @pytest.mark.parametrize("seed", [1412, 15285, 1484, 22])
    @pytest.mark.parametrize("wires", [[0], ["a"], [1]])
    def test_basic(self, seed, wires):
        """Test basic usage of `one_qubit_flag_decomp`."""
        v = unitary_group.rvs(2, random_state=seed)
        with qml.queuing.AnnotatedQueue() as q:
            F, delta = one_qubit_flag_decomp(v, wires=wires)
        assert len(q.queue) == 0

        assert isinstance(F, list) and len(F) == 1 # 1 flag
        rec_mat = np.diag(delta) @ qml.matrix(F, wire_order=wires)
        assert np.allclose(rec_mat, v)

    @pytest.mark.parametrize("v", [qml.X(0).matrix(), np.eye(2), qml.Y(0).matrix()])
    def test_edge_cases(self, v):
        """Test basic usage of `one_qubit_flag_decomp`."""
        wires = [0]
        F, delta = one_qubit_flag_decomp(v, wires=wires)
        assert isinstance(F, list) and len(F) == 1 # 1 flag
        rec_mat = np.diag(delta) @ qml.matrix(F, wire_order=wires)
        assert np.allclose(rec_mat, v)

class TestTwoQubitFlagDecomp:
    """Tests for the two-qubit flag decomposition in `two_qubit_flag_decomp`."""

    @pytest.mark.parametrize("seed", [1412, 15285, 1484, 22])
    @pytest.mark.parametrize("wires", [[0, 1], ["a", 5], [1, 0]])
    def test_basic(self, seed, wires):
        """Test basic usage of `two_qubit_flag_decomp`."""
        v = unitary_group.rvs(4, random_state=seed)
        with qml.queuing.AnnotatedQueue() as q:
            F, delta = two_qubit_flag_decomp(v, wires=wires)
        assert len(q.queue) == 0

        #assert isinstance(F, list) and len(F) == 8 # 2 CNOTs and 6 flags
        rec_mat = np.diag(delta) @ qml.matrix(F, wire_order=wires)
        assert np.allclose(rec_mat, v)

    @pytest.mark.parametrize("v", [qml.CNOT([0, 1]).matrix(), np.eye(4), qml.SWAP([0, 1]).matrix()])
    def test_edge_cases(self, v):
        """Test basic usage of `two_qubit_flag_decomp`."""
        wires = [0, 1]
        F, delta = two_qubit_flag_decomp(v, wires=wires)
        assert isinstance(F, list) and len(F) == 8 # 2 CNOTs and 6 flags
        rec_mat = np.diag(delta) @ qml.matrix(F, wire_order=wires)
        assert np.allclose(rec_mat, v)

class TestMultiplexedFlag:
    """Tests for `multiplexed_flag`."""

    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_standard_validity(self, seed, k):
        """Test standard operator validity."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 2**k))
        op = MultiplexedFlag(theta_z, theta_y, list(range(k+1)))
        assert_valid(op, skip_differentiation=True)

    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
    def test_decomposition(self, seed, k):
        """Test basic usage."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 2**k))
        op = MultiplexedFlag(theta_z, theta_y, list(range(k+1)))

        with qml.queuing.AnnotatedQueue() as q:
            decomp_0 = op.compute_decomposition(theta_z, theta_y, wires=list(range(k+1)))
        assert len(q.queue) == 4 * 2**k if k > 0 else 2

        with qml.queuing.AnnotatedQueue() as q:
            decomp_1 = op.decomposition()
        assert len(q.queue) == 4 * 2**k if k > 0 else 2

        decomp_mat_0 =qml.matrix(decomp_0, wire_order=range(k+1))
        decomp_mat_1 =qml.matrix(decomp_1, wire_order=range(k+1))

        individual_mats = [qml.matrix(qml.RY(thy, 0) @ qml.RZ(thz, 0)) for thz, thy in zip(theta_z, theta_y)]
        expected = qml.math.block_diag(individual_mats)
        assert np.allclose(expected, decomp_mat_0)
        assert np.allclose(expected, decomp_mat_1)


class TestMuxOps:
    """Tests for `mux_ops`."""

    @pytest.mark.parametrize("ops0,ops1", [
        ([qml.CNOT([0, 1])], [qml.CNOT([0, 1])]),
        ([qml.X(0)], [qml.X(0)]),
        ([qml.CNOT([0, 1]), qml.CY([1, 5])], [qml.CNOT([0, 1]), qml.CY([1, 5])]),
    ])
    def test_with_static_gates(self, ops0, ops1):
        """Test that static gates are just copied."""
        new_ops = mux_ops(ops0, ops1, 100)
        assert new_ops == ops0

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_with_single_mux_flag(self, k):
        """Test that a single multiplexed flag is multiplexed further."""
        theta_z0, theta_y0, theta_z1, theta_y1 = np.random.random((4, 2**k))
        ops0 = [MultiplexedFlag(theta_z0, theta_y0, list(range(k+1)))]
        ops1 = [MultiplexedFlag(theta_z1, theta_y1, list(range(k+1)))]
        with qml.queuing.AnnotatedQueue() as q:
            new_ops = mux_ops(ops0, ops1, k+1)
        assert len(q.queue) == 0

        assert len(new_ops) == 1
        new_op = new_ops[0]
        assert isinstance(new_op, MultiplexedFlag)
        assert np.allclose(new_op.data[0][:2**k], theta_z0)
        assert np.allclose(new_op.data[0][2**k:], theta_z1)
        assert np.allclose(new_op.data[1][:2**k], theta_y0)
        assert np.allclose(new_op.data[1][2**k:], theta_y1)
        assert new_op.wires == Wires([k+1]+list(range(k+1)))


    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_with_mixed_ops(self, k):
        """Test that a single multiplexed flag is multiplexed further."""
        theta_z0, theta_y0, theta_z1, theta_y1 = np.random.random((4, 2**k))
        ops0 = [MultiplexedFlag(theta_z0, theta_y0, list(range(k+1))), qml.CNOT([0, 1]), qml.H(2), qml.CY([2, 0]), MultiplexedFlag(theta_z1, theta_y1, list(range(k+1)))]
        ops1 = [MultiplexedFlag(theta_y1, theta_z1, list(range(k+1))), qml.CNOT([0, 1]), qml.H(2), qml.CY([2, 0]), MultiplexedFlag(theta_y0, theta_z0, list(range(k+1)))]
        with qml.queuing.AnnotatedQueue() as q:
            new_ops = mux_ops(ops0, ops1, k+1)

        assert len(q.queue) == 0
        assert all(isinstance(new_op, type(op0)) for new_op, op0 in zip(new_ops, ops0, strict=True))

        wire_order = list(range(k+1))
        individual_mats = [qml.matrix(ops, wire_order=wire_order) for ops in [ops0, ops1]]
        expected = qml.math.block_diag(individual_mats)
        new_mat = qml.matrix(new_ops, wire_order=[k+1]+wire_order)
        assert np.allclose(new_mat, expected)


class TestDecomposeMuxSingleQubitFlags:
    """Tests for `_decompose_mux_single_qubit_flag`."""


    @pytest.mark.parametrize("wires", [[0, 1], [1, 0], ["A", 0]])
    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    def test_base_case(self, seed, wires):
        """Test the base case decomposition."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 2))
        op = MultiplexedFlag(theta_z, theta_y, wires)
        op_mat = qml.matrix(op, wire_order=wires)

        with qml.queuing.AnnotatedQueue() as q:
            ops, D = _decompose_mux_single_qubit_flag(op)

        assert len(q.queue) == 5 # 4 rotations and a CZ, no diagonal queued
        out_mat = np.diag(D) @ qml.matrix(ops, wire_order=wires)
        out_mat2 = np.diag(D) @ qml.matrix(q.queue, wire_order=wires)
        assert np.allclose(op_mat, out_mat)
        assert np.allclose(op_mat, out_mat2)

    @pytest.mark.parametrize("wires", [[0, 1, 2], [1, 2, 0], [-3, "A", 0]])
    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    def test_two_mux_nodes(self, seed, wires):
        """Test the base case decomposition."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 4))
        op = MultiplexedFlag(theta_z, theta_y, wires)
        op_mat = qml.matrix(op, wire_order=wires)

        with qml.queuing.AnnotatedQueue() as q:
            ops, D = _decompose_mux_single_qubit_flag(op)

        assert len(q.queue) == 11 # 8 rotations and 3 CZs, no diagonal queued
        out_mat = np.diag(D) @ qml.matrix(ops, wire_order=wires)
        out_mat2 = np.diag(D) @ qml.matrix(q.queue, wire_order=wires)
        assert np.allclose(op_mat, out_mat)
        assert np.allclose(op_mat, out_mat2)

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    def test_many_mux_nodes(self, seed, k):
        """Test the base case decomposition."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 2**k))
        wires = list(range(k+1))
        op = MultiplexedFlag(theta_z, theta_y, wires)
        op_mat = qml.matrix(op, wire_order=wires)

        with qml.queuing.AnnotatedQueue() as q:
            ops, D = _decompose_mux_single_qubit_flag(op)

        assert len(q.queue) == 2**(k+1) + 2**k - 1
        out_mat = np.diag(D) @ qml.matrix(ops, wire_order=wires)
        out_mat2 = np.diag(D) @ qml.matrix(q.queue, wire_order=wires)
        assert np.allclose(op_mat, out_mat)
        assert np.allclose(op_mat, out_mat2)
