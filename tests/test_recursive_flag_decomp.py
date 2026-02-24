from functools import partial
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from pennylane.ops.functions import assert_valid
from pennylane.wires import Wires

from rotoptsynth.recursive_flag_decomp import one_qubit_flag_decomp, two_qubit_flag_decomp, MultiplexedFlag, mux_ops, _decompose_mux_single_qubit_flag, mux_multi_qubit_decomp, recursive_flag_decomp_cliff_rz

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

    @pytest.mark.parametrize("ops", [
        [qml.CNOT([0, 1])],
        [qml.X(0)],
        [qml.CNOT([0, 1]), qml.CY([1, 5])]
    ])
    @pytest.mark.parametrize("controls", ([100], [10, "aux"], [1, 2, 3, 4, 5, 6]))
    def test_with_static_gates(self, ops, controls):
        """Test that static gates are just copied."""
        k = len(controls)
        new_ops = mux_ops([ops for _ in range(2**k)], controls)
        assert new_ops == ops

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("num_controls", [1, 2, 3, 4, 5])
    def test_with_single_mux_flag(self, k, num_controls):
        """Test that a single multiplexed flag is multiplexed further."""
        thetas = np.random.random((2 * 2**num_controls, 2**k))
        ops = [[MultiplexedFlag(*thetas[2*i:2*i+2], list(range(k+1)))] for i in range(2**num_controls)]
        with qml.queuing.AnnotatedQueue() as q:
            new_ops = mux_ops(ops, list(range(k+1, k+1+num_controls)))
        assert len(q.queue) == 0

        assert len(new_ops) == 1
        new_op = new_ops[0]
        assert isinstance(new_op, MultiplexedFlag)
        exp_theta_z = np.concatenate(thetas[::2])
        exp_theta_y = np.concatenate(thetas[1::2])
        assert np.allclose(new_op.data[0], exp_theta_z)
        assert np.allclose(new_op.data[1], exp_theta_y)
        exp_wires = Wires(list(range(k+1,k+1+num_controls))+list(range(k+1)))
        assert new_op.wires == exp_wires

    @pytest.mark.parametrize("k", [2, 3, 4])
    @pytest.mark.parametrize("num_controls", [1, 2, 3])
    def test_with_mixed_ops(self, k, num_controls):
        """Test that a single multiplexed flag is multiplexed further."""
        thetas, phis = np.random.random((2, 2* 2**num_controls, 2**k))
        targets = list(range(k+1))
        controls = list(range(k+1, k+1+num_controls))
        ops = [
            [
                MultiplexedFlag(*thetas[2*i:2*i+2], targets),
                qml.CNOT([0, 1]),
                qml.H(2),
                qml.CY([2, 0]),
                MultiplexedFlag(*phis[2*i:2*i+2], targets[::-1]),
            ]
            for i in range(2**num_controls)
        ]

        with qml.queuing.AnnotatedQueue() as q:
            new_ops = mux_ops(ops, controls)

        assert len(q.queue) == 0
        assert all(isinstance(new_op, type(op0)) for new_op, op0 in zip(new_ops, ops[0], strict=True))

        individual_mats = [qml.matrix(_ops, wire_order=targets) for _ops in ops]
        expected = qml.math.block_diag(individual_mats)
        new_mat = qml.matrix(new_ops, wire_order=controls+targets)
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

class TestMuxMultiQubitDecomp:
    """Test ``mux_multi_qubit_decomp``."""

    @pytest.mark.parametrize("seed", [5112, 8622, 862])
    @pytest.mark.parametrize("num_controls", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_b", [1, 2])
    def test_two_qubit_unitaries(self, num_controls, seed, n_b):
        """Test the base case of multiplexed two-qubit unitaries."""
        mats = unitary_group.rvs(4, size=2**num_controls, random_state=seed)
        controls = list(range(num_controls))
        targets = list(range(num_controls, num_controls+2))
        ops, diag = mux_multi_qubit_decomp(mats, controls, targets, n_b)

        in_mat = qml.math.block_diag(mats)
        rec_mat = np.diag(diag) @ qml.matrix(ops, wire_order=controls+targets)
        assert np.allclose(rec_mat, in_mat)
        assert all(isinstance(op, (qml.RZ, qml.RY, qml.CZ)) for op in ops)
        num_rots = sum(isinstance(op, (qml.RZ, qml.RY)) for op in ops)
        assert num_rots == 12 * 2**num_controls # two qubit flag has 12 rotations
        if n_b == 2:
            exp_num_cnots = 6 * 2**num_controls - 4 # Eq.(D27)
        else:
            n = 2
            exp_num_cnots = (2**n - 1) * (2**(n-1+num_controls)-1)

        assert len(ops) - num_rots == exp_num_cnots

    @pytest.mark.parametrize("seed", [512, 8362])
    @pytest.mark.parametrize("num_controls, num_targets", [(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (4, 3)])
    @pytest.mark.parametrize("n_b", [1, 2])
    def test_multi_qubit_unitaries(self, num_controls, num_targets, seed, n_b):
        """Test the base case of multiplexed two-qubit unitaries."""
        mats = unitary_group.rvs(2**num_targets, size=2**num_controls, random_state=seed)
        controls = list(range(num_controls))
        targets = list(range(num_controls, num_controls+num_targets))
        ops, diag = mux_multi_qubit_decomp(mats, controls, targets, n_b=n_b)

        tape = qml.tape.QuantumScript(ops)
        in_mat = qml.math.block_diag(mats)
        rec_mat = np.diag(diag) @ qml.matrix(ops, wire_order=controls+targets)
        assert np.allclose(rec_mat, in_mat)
        assert all(isinstance(op, (qml.RZ, qml.RY, qml.CZ)) for op in ops)
        num_rots = sum(isinstance(op, (qml.RZ, qml.RY)) for op in ops)
        n = num_targets
        k = num_controls
        assert num_rots == (4**n - 2**n) * 2**k # Flag d.o.f.s
        if n_b == 2:
            exp_num_cnots = (4**n - 2**n)//2 * 2**k - 5*2**n//4 + 1
        else:
            exp_num_cnots = (2**n - 1) * (2**(n-1+k)-1)
        assert len(ops) - num_rots == exp_num_cnots

class TestRecursiveFlagDecompCliffRz:
    """Test the main recursive flag decomposition function."""

    @pytest.mark.parametrize("seed", [932, 2185, 752])
    @pytest.mark.parametrize("num_targets", [2, 3, 4, 5])
    @pytest.mark.parametrize("n_b", [1, 2])
    @pytest.mark.parametrize("selective_demux", [True, False])
    def test_main_usage(self, seed, num_targets, n_b,selective_demux):
        """Test main usage."""
        if n_b == 1 and selective_demux:
            pytest.skip(reason="We never use this scenario.")
        targets = list(range(num_targets))
        V = unitary_group.rvs(2**num_targets, random_state=seed)
        ops, diag = recursive_flag_decomp_cliff_rz(V, targets, n_b, selective_demux)
        rec_mat = np.diag(diag) @ qml.matrix(ops, wire_order=targets)
        assert np.allclose(rec_mat, V)
        assert all(isinstance(op, (qml.RZ, qml.RY, qml.CZ, qml.CNOT)) for op in ops), f"{set(type(op) for op in ops)}"
        num_rots = sum(isinstance(op, (qml.RZ, qml.RY)) for op in ops)
        n = num_targets
        assert num_rots == (4**n - 2**n)

        if n_b == 2:
            if selective_demux:
                exp_num_cnots = 4**n//2-(n+12)*2**n // 8 + 1 # Eq. (D26)
            else:
                exp_num_cnots = (4**n - 2**n)//2 - 5*2**n//4 + 1 # Eq. (D27)
        else:
            exp_num_cnots = (2**n - 1) * (2**(n-1)-1)

        assert len(ops) - num_rots == exp_num_cnots
