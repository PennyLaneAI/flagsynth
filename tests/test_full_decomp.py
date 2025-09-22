import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros


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

targets = targets_2q +targets_3q + targets_4q

class TestRotOptSynth:

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_2q)
    def test_builtin_validation_two_qubits(self, target):
        _ = ros.rot_opt_synth(target, [0, 1])

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_3q)
    def test_builtin_validation_three_qubits(self, target):
        _ = ros.rot_opt_synth(target, [0, 1, 2])

    @pytest.mark.with_validation
    @pytest.mark.parametrize("target", targets_4q)
    def test_builtin_validation_four_qubits(self, target):
        _ = ros.rot_opt_synth(target, [0, 1, 2, 3])

    @pytest.mark.without_validation
    @pytest.mark.parametrize("wires", [(1, 0, 2, -1), ("a", 5, "v", 2)])
    @pytest.mark.parametrize("target", targets)
    def test_wires(self, target, wires):
        n = len(bin(len(target))) - 3
        wires = wires[:n]
        ops = ros.rot_opt_synth(target, wires)
        assert all(set(op.wires).issubset(set(wires)) for op in ops)

    @pytest.mark.without_validation
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def queuing_matches_return(self, n):
        target = unitary_group.rvs(2**n, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            ops = ros.rot_opt_synth(target, range(n))

        assert ops == q.queue



# Note: The test functions rely on the validation asserts inside rot_opt_synth itself
# to verify correctness. The primary test is that the function executes without error
# when validation is enabled, which means its internal reconstruction of the unitary passed.

class TestRotOptSynth:
    """Tests for the rot_opt_synth unitary synthesis function."""

    @pytest.mark.with_validation
    @pytest.mark.parametrize("seed", [42, 48, 96])
    def test_decomposition_correctness_2_qubits(self, seed):
        """Test that a 2-qubit random unitary is correctly decomposed."""
        wires = [0, 1]
        u = unitary_group.rvs(2 ** 2, random_state=seed)

        # The test passes if the function runs without raising an assertion error.
        # The function's internal validation `assert np.allclose(u, u_rec)`
        # serves as the correctness check.
        ros.rot_opt_synth(u, wires)

    @pytest.mark.with_validation
    @pytest.mark.parametrize("seed", [24, 61])
    def test_decomposition_correctness_3_qubits(self, seed):
        """Test that a 3-qubit random unitary is correctly decomposed."""
        wires = range(3)
        u = unitary_group.rvs(2 ** 3, random_state=seed)

        # This tests the main recursive logic of the function.
        # Correctness is verified by the function's internal validation.
        ros.rot_opt_synth(u, wires)

    @pytest.mark.without_validation
    @pytest.mark.parametrize("seed", [9612, 8124])
    def test_decomposition_correctness_5_qubits(self, seed):
        """Test that a 5-qubit random unitary is correctly decomposed."""
        wires = range(5)
        u = unitary_group.rvs(2 ** 5, random_state=seed)

        ops = ros.rot_opt_synth(u, wires)
        assert np.allclose(ros.ops_to_mat(ops, wires), u)

    @pytest.mark.with_validation
    @pytest.mark.parametrize("num_wires, wire_labels", [(2, ["a", "b"]), (3, [0, "c", 2])])
    def test_identity_decomposition(self, num_wires, wire_labels):
        """Test that the identity matrix is decomposed correctly."""
        identity = np.eye(2 ** num_wires)
        ros.rot_opt_synth(identity, wire_labels)

    @pytest.mark.with_validation
    def test_toffoli_decomposition(self):
        """Test that the Toffoli gate is decomposed correctly."""
        toffoli = qml.Toffoli([0, 1, 2]).matrix()
        ros.rot_opt_synth(toffoli, wires=[0, 1, 2])

    @pytest.mark.with_validation
    def test_queuing_behavior(self):
        """Test that the decomposition is correctly queued inside a tape."""
        u = unitary_group.rvs(8, random_state=10)
        wires = [0, 1, 2]

        with qml.queuing.AnnotatedQueue() as q:
            ops_list = ros.rot_opt_synth(u, wires)
        assert ops_list == q.queue


@pytest.mark.with_validation
class TestRotOptSynthErrors:
    """Error handling and validation tests for rot_opt_synth."""

    def test_incorrect_matrix_size_raises_error(self):
        """Test that a matrix of incorrect dimensions raises an AssertionError."""
        u = np.eye(4)  # 2-qubit matrix
        wires = [0, 1, 2] # 3 wires

        # The internal validation `assert len(u) == 2**num_wires` should fail.
        with pytest.raises(AssertionError):
            ros.rot_opt_synth(u, wires)

    def test_non_unitary_matrix_raises_error(self):
        """Test that a non-unitary matrix raises an AssertionError."""
        # Create a non-unitary matrix by zeroing out a row
        u = unitary_group.rvs(4, random_state=1)
        u[0, :] = 0

        # The internal validation `assert is_unitary(u)` should fail.
        with pytest.raises(AssertionError):
            ros.rot_opt_synth(u, wires=[0, 1])
