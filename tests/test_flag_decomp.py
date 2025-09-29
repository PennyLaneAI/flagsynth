import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros
from rotoptsynth.flag_decomp import _flag_decomp_two_qubits, _flag_decomp_one_qubit

# Test data for different gate types
test_data_rotations = [
    (qml.RX, "X"),
    (qml.RY, "Y"),
    (qml.RZ, "Z"),
]

test_data_static = [
    (qml.CNOT, [0, 1]),
    (qml.CZ, [0, 1]),
]


class TestAttachMultiplexerNode:
    """Tests for the attach_multiplexer_node function."""

    @pytest.mark.without_validation
    @pytest.mark.parametrize("rot_gate, axis", test_data_rotations)
    def test_single_rotation_gates(self, rot_gate, axis):
        """Test multiplexing of single rotation gates (RX, RY, RZ)."""
        ops0 = [rot_gate(0.6, wires=0)]
        ops1 = [rot_gate(-0.1, wires=0)]
        mpx_wire = "mpx"

        result = ros.attach_multiplexer_node(ops0, ops1, mpx_wire)

        assert len(result) == 1
        new_op = result[0]

        assert isinstance(new_op, qml.SelectPauliRot)
        assert new_op.wires == qml.wires.Wires([mpx_wire, 0])
        assert new_op.hyperparameters["rot_axis"] == axis
        np.testing.assert_allclose(new_op.parameters[0], [0.6, -0.1])

    @pytest.mark.without_validation
    @pytest.mark.parametrize("static_gate, wires", test_data_static)
    def test_static_gates(self, static_gate, wires):
        """Test multiplexing of static gates (CNOT, CZ)."""
        ops0 = [static_gate(wires=wires)]
        ops1 = [static_gate(wires=wires)]
        mpx_wire = "mpx"

        result = ros.attach_multiplexer_node(ops0, ops1, mpx_wire)

        assert len(result) == 1
        new_op = result[0]

        assert isinstance(new_op, static_gate)
        # The multiplexer wire should NOT be added for static gates
        assert new_op.wires == qml.wires.Wires(wires)

    @pytest.mark.without_validation
    def test_select_pauli_rot_gate(self):
        """Test multiplexing of existing SelectPauliRot gates."""
        ops0 = [
            qml.SelectPauliRot([0.6, 0.7], control_wires=[0], target_wire="target", rot_axis="X")
        ]
        ops1 = [
            qml.SelectPauliRot([0.2, -0.5], control_wires=[0], target_wire="target", rot_axis="X")
        ]
        mpx_wire = "new_mpx"

        result = ros.attach_multiplexer_node(ops0, ops1, mpx_wire)

        assert len(result) == 1
        new_op = result[0]

        assert isinstance(new_op, qml.SelectPauliRot)
        assert new_op.wires == qml.wires.Wires(["new_mpx", 0, "target"])
        assert new_op.hyperparameters["rot_axis"] == "X"
        np.testing.assert_allclose(new_op.parameters[0], [0.6, 0.7, 0.2, -0.5])

    @pytest.mark.without_validation
    def test_mixed_operations_list(self):
        """Test a list containing a mix of supported operations."""
        ops0 = [qml.RX(0.5, 0), qml.CNOT([0, 1]), qml.RY(1.2, 1)]
        ops1 = [qml.RX(0.1, 0), qml.CNOT([0, 1]), qml.RY(-0.4, 1)]
        mpx_wire = 2

        result = ros.attach_multiplexer_node(ops0, ops1, mpx_wire)

        assert len(result) == 3

        # Check first op (RX -> SelectPauliRot)
        assert isinstance(result[0], qml.SelectPauliRot)
        assert result[0].wires == qml.wires.Wires([2, 0])
        np.testing.assert_allclose(result[0].parameters[0], [0.5, 0.1])

        # Check second op (CNOT -> CNOT)
        assert isinstance(result[1], qml.CNOT)
        assert result[1].wires == qml.wires.Wires([0, 1])

        # Check third op (RY -> SelectPauliRot)
        assert isinstance(result[2], qml.SelectPauliRot)
        assert result[2].wires == qml.wires.Wires([2, 1])
        np.testing.assert_allclose(result[2].parameters[0], [1.2, -0.4])

    @pytest.mark.without_validation
    def test_queuing_behavior_without_validation(self):
        """Test that operations are correctly queued within a queuing context."""
        ops0 = [qml.RZ(0.8, "a"), qml.CNOT(["a", "b"])]
        ops1 = [qml.RZ(-0.8, "a"), qml.CNOT(["a", "b"])]
        mpx_wire = "c"

        with qml.queuing.AnnotatedQueue() as q:
            returned = ros.attach_multiplexer_node(ops0, ops1, mpx_wire)

        assert len(q.queue) == 2
        assert q.queue == returned

    @pytest.mark.with_validation
    def test_queuing_behavior_with_validation(self):
        """Test that operations are correctly queued within a queuing context."""
        ops0 = [qml.RZ(0.8, "a"), qml.CNOT(["a", "b"])]
        ops1 = [qml.RZ(-0.8, "a"), qml.CNOT(["a", "b"])]
        mpx_wire = "c"

        with qml.queuing.AnnotatedQueue() as q:
            returned = ros.attach_multiplexer_node(ops0, ops1, mpx_wire)

        assert len(q.queue) == 2
        assert q.queue == returned


@pytest.mark.with_validation
class TestAttachMultiplexerNodeErrors:
    """Error and validation tests for attach_multiplexer_node."""

    def test_unsupported_operation_raises_error(self):
        """Test that an unsupported op type raises NotImplementedError."""
        ops0 = [qml.Hadamard(0)]
        ops1 = [qml.Hadamard(0)]
        with pytest.raises(NotImplementedError, match="Hadamard"):
            ros.attach_multiplexer_node(ops0, ops1, "mpx")

    def test_different_lengths_raises_error(self):
        """Test that op sequences of different lengths raise ValueError."""
        ops0 = [qml.RX(0.5, 0)]
        ops1 = []  # Mismatched length
        # zip(..., strict=True) raises a ValueError
        with pytest.raises(ValueError, match="is shorter"):
            ros.attach_multiplexer_node(ops0, ops1, "mpx")

    def test_different_op_types_raises_error(self):
        """Test that mismatched op types raise AssertionError if validation is on."""
        ops0 = [qml.RX(0.5, 0)]
        ops1 = [qml.RY(0.5, 0)]  # Mismatched type
        with pytest.raises(AssertionError):
            ros.attach_multiplexer_node(ops0, ops1, "mpx")

    def test_different_wires_raises_error(self):
        """Test that mismatched wires raise AssertionError if validation is on."""
        ops0 = [qml.CNOT([0, 1])]
        ops1 = [qml.CNOT([0, 2])]  # Mismatched wires
        with pytest.raises(AssertionError):
            ros.attach_multiplexer_node(ops0, ops1, "mpx")

    def test_different_rot_axis_raises_error(self):
        """Test mismatched rot_axis in SelectPauliRot raises AssertionError."""
        ops0 = [qml.SelectPauliRot([0.1, 0.6], [0], 1, rot_axis="X")]
        ops1 = [qml.SelectPauliRot([0.2, -0.2], [0], 1, rot_axis="Y")]  # Mismatched axis
        with pytest.raises(AssertionError):
            ros.attach_multiplexer_node(ops0, ops1, "mpx")

targets_1q = [
    np.eye(2),
    qml.RX(0.623, 0).matrix(),
    qml.RZ(-2.195, 0).matrix(),
    unitary_group.rvs(2, random_state=21524),
    unitary_group.rvs(2, random_state=8364),
    unitary_group.rvs(2, random_state=9612),
    -np.eye(2),
]


class TestFlagDecompOneQubit:

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_1q)
    def test_builtin_validation(self, target):
        diagonal, other_data = _flag_decomp_one_qubit(target)
        assert isinstance(diagonal, np.ndarray)
        assert diagonal.dtype == np.complex128
        assert diagonal.shape == (2,)
        assert len(other_data) == 2
        for d in other_data:
            assert isinstance(d, float)

    @pytest.mark.without_validation
    def test_no_queuing(self):
        target = unitary_group.rvs(2, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            _ = _flag_decomp_one_qubit(target)
        assert not q.queue

    @pytest.mark.with_validation
    def test_no_queuing_with_validation(self):
        target = unitary_group.rvs(2, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            _ = _flag_decomp_one_qubit(target)
        assert not q.queue


targets_2q = [
    np.eye(4),
    qml.CNOT([0, 1]).matrix(),
    qml.RX(0.623, 0).matrix(wire_order=[0, 1]),
    qml.IsingXY(2.15, [1, 0]).matrix(),
    unitary_group.rvs(4, random_state=21524),
    unitary_group.rvs(4, random_state=8364),
    unitary_group.rvs(4, random_state=9612),
    -np.eye(4),
]


class TestFlagDecompTwoQubits:

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_2q)
    def test_builtin_validation(self, target):
        diagonal, other_data = _flag_decomp_two_qubits(target)
        assert isinstance(diagonal, np.ndarray)
        assert diagonal.dtype == np.complex128
        assert diagonal.shape == (4,)
        assert len(other_data) == 8
        for d in other_data[:-2]:
            assert isinstance(d, float)
        for d in other_data[-2:]:
            assert isinstance(d, np.ndarray)
            assert d.shape == (2, 2)
            assert d.dtype == np.complex128

    @pytest.mark.without_validation
    def test_no_queuing(self):
        target = unitary_group.rvs(4, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            _ = _flag_decomp_two_qubits(target)
        assert not q.queue

    @pytest.mark.with_validation
    def test_no_queuing_with_validation(self):
        target = unitary_group.rvs(4, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            _ = _flag_decomp_two_qubits(target)
        assert not q.queue


targets_3q = [
    np.eye(8),
    -np.eye(8),
    qml.CNOT([0, 1]).matrix(wire_order=[0, 2, 1]),
    qml.RX(0.623, 0).matrix(wire_order=[2, 0, 1]),
    qml.IsingXY(2.15, [1, 0]).matrix(wire_order=[1, 2, 0]),
    unitary_group.rvs(8, random_state=2524),
    unitary_group.rvs(8, random_state=864),
    unitary_group.rvs(8, random_state=962),
    qml.Toffoli([0, 2, 1]).matrix(),
]

targets_4q = [
    np.eye(16),
    -np.eye(16),
    qml.CNOT([0, 1]).matrix(wire_order=[0, 2, 3, 1]),
    qml.RX(0.623, 0).matrix(wire_order=[3, 2, 0, 1]),
    qml.IsingXY(2.15, [1, 0]).matrix(wire_order=[1, 2, 0, 3]),
    unitary_group.rvs(16, random_state=2524),
    unitary_group.rvs(16, random_state=864),
    unitary_group.rvs(16, random_state=962),
    qml.Toffoli([0, 2, 1]).matrix(wire_order=[3, 0, 1, 2]),
    qml.MultiControlledX([3, 0, 2, 1]).matrix(),
    qml.MultiRZ(-0.7124, [3, 0, 1, 2]).matrix(),
]

targets = targets_2q + targets_3q + targets_4q


class TestDiagDecomp:

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_2q)
    def test_builtin_validation_two_qubits(self, target):
        diag_op, other_ops = ros.flag_decomp(target, [0, 1])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)
        assert len(other_ops) == 10  # 2 CNOTs + 6 parametrized rotations + 2 SU(2) ops

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_3q)
    def test_builtin_validation_three_qubits(self, target):
        diag_op, other_ops = ros.flag_decomp(target, [0, 1, 2])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)
        assert len(other_ops) == 22  # 2 times two-qubit case + 2 multiplexers

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_4q)
    def test_builtin_validation_four_qubits(self, target):
        diag_op, other_ops = ros.flag_decomp(target, [0, 1, 2, 3])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)
        assert len(other_ops) == 46  # 2 times three-qubit case + 2 multiplexers

    @pytest.mark.without_validation
    @pytest.mark.parametrize("wires", [(1, 0, 2, -1), ("a", 5, "v", 2)])
    @pytest.mark.parametrize("target", targets)
    def test_wires(self, target, wires):
        n = len(bin(len(target))) - 3
        wires = wires[:n]
        diag_op, other_ops = ros.flag_decomp(target, wires)
        assert diag_op.wires == wires
        assert all(set(op.wires).issubset(set(wires)) for op in other_ops)

    @pytest.mark.without_validation
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_queuing_matches_return_without_validation(self, n):
        target = unitary_group.rvs(2**n, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            diag_op, other_ops = ros.flag_decomp(target, range(n))

        assert [diag_op] + other_ops == q.queue

    @pytest.mark.with_validation
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_queuing_matches_return_with_validation(self, n):
        target = unitary_group.rvs(2**n, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            diag_op, other_ops = ros.flag_decomp(target, range(n))

        assert [diag_op] + other_ops == q.queue
