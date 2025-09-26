import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros
from rotoptsynth.rot_opt_qsd import _validate_and_arrange_zeroed_wires


class TestValidateAndArrangeZeroedWires:
    """Tests for _validate_and_arrange_zeroed_wires."""

    def _assert_unitary_equivalency(self, u_orig, wires_orig, u_new, wires_new):
        """
        Helper to assert that two unitaries are physically equivalent even if the
        matrix and wire orders are different.
        """
        # The matrix of the original op must equal the matrix of the new op
        wire_order = wires_orig
        original_matrix = qml.matrix(
            qml.QubitUnitary(u_orig, wires=wires_orig), wire_order=wire_order
        )
        reordered_matrix = qml.matrix(
            qml.QubitUnitary(u_new, wires=wires_new), wire_order=wire_order
        )
        assert np.allclose(original_matrix, reordered_matrix)

    def test_no_reordering_needed(self):
        """Test the case where zeroed_wires are already at the front."""
        wires = ["a", "b", "c"]
        zeroed_wires = ["a"]
        u = np.arange(64).reshape(8, 8)

        u_new, new_wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)

        assert np.array_equal(u, u_new)  # Matrix should be unchanged
        assert new_wires == wires  # Wires should be unchanged

    def test_reordering_single_wire_from_end(self):
        """Test reordering when a single zeroed_wire is at the end."""
        wires = [0, 1, 2]
        zeroed_wires = [2]
        u = np.arange(64).reshape(8, 8)

        u_new, new_wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)

        assert not np.array_equal(u, u_new)  # Matrix must have been permuted
        assert new_wires == [2, 0, 1]  # Wires must be reordered
        self._assert_unitary_equivalency(u, wires, u_new, new_wires)

    def test_reordering_multiple_wires_mixed(self):
        """Test reordering with multiple, out-of-order zeroed_wires."""
        wires = ["w0", "w1", "w2", "w3"]
        zeroed_wires = ["w2", "w0"]
        u = np.arange(256).reshape(16, 16)

        u_new, new_wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)

        assert not np.array_equal(u, u_new)
        # The new order should be [zeroed_wires..., other_wires...], with others in original relative order
        assert new_wires == ["w2", "w0", "w1", "w3"]
        self._assert_unitary_equivalency(u, wires, u_new, new_wires)

    def test_empty_zeroed_wires(self):
        """Test that an empty zeroed_wires list is a valid no-op."""
        wires = [0, 1]
        zeroed_wires = []
        u = np.arange(16).reshape(4, 4)

        u_new, new_wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)

        assert np.array_equal(u, u_new)
        assert new_wires == wires

    def test_all_wires_are_zeroed(self):
        """Test the case where all wires are specified as zeroed."""
        wires = [0, 1]
        zeroed_wires = [1, 0]
        u = np.arange(16).reshape(4, 4)

        u_new, new_wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)
        assert np.array_equal(u, u_new)
        assert new_wires == wires


class TestValidateAndArrangeErrors:
    """Tests for the validation and error-raising logic of the function."""

    @pytest.mark.parametrize(
        "wires, zeroed_wires",
        [
            ([0, 1], [2]),  # Zeroed wire is completely outside
            (["a", "b"], ["a", "c"]),  # One valid, one invalid zeroed wire
            ([0, 1], ["a"]),  # Mismatched wire label types
        ],
    )
    def test_error_zeroed_wire_not_in_wires(self, wires, zeroed_wires):
        """Test that a ValueError is raised if a zeroed_wire is not in wires."""
        u = np.eye(4)

        # The match argument checks that the error message contains the expected text
        with pytest.raises(ValueError, match="All provided zeroed_wires must be part of the "):
            _ = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)


targets_1q = [
    np.eye(2),
    -np.eye(2),
    qml.X(0).matrix(),
    qml.RZ(0.5612, 0).matrix(),
    qml.GlobalPhase(0.723, 0).matrix(),
    unitary_group.rvs(2, random_state=2124),
    unitary_group.rvs(2, random_state=7215),
    unitary_group.rvs(2, random_state=7613),
]

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

targets = targets_2q + targets_3q + targets_4q


def _take_zeroed_submat(arr, n, zeroed_wire):
    return np.take(np.reshape(arr, (2,) * (2 * n)), 0, axis=zeroed_wire + n)


class TestRotOptSynth:
    """Tests for rot_opt_qsd."""

    @pytest.mark.with_validation
    @pytest.mark.parametrize("num_zeroed_wires, expected_count", [(0, 4), (1, 3)])
    @pytest.mark.parametrize("target", targets_1q)
    def test_builtin_validation_one_qubit(self, target, num_zeroed_wires, expected_count):
        wires = list(range(1))
        zeroed_wires = list(range(num_zeroed_wires))
        ops = ros.rot_opt_qsd(target, wires, zeroed_wires=zeroed_wires)
        assert ros.count_rotation_angles(ops) == expected_count

    @pytest.mark.with_validation
    @pytest.mark.parametrize("num_zeroed_wires, expected_count", [(0, 16), (1, 12), (2, 11)])
    @pytest.mark.parametrize("target", targets_2q)
    def test_builtin_validation_two_qubits(self, target, num_zeroed_wires, expected_count):
        wires = list(range(2))
        zeroed_wires = list(range(num_zeroed_wires))
        ops = ros.rot_opt_qsd(target, wires, zeroed_wires=zeroed_wires)
        assert ros.count_rotation_angles(ops) == expected_count

    @pytest.mark.with_validation
    @pytest.mark.parametrize(
        "num_zeroed_wires, expected_count", [(0, 64), (1, 48), (2, 44), (3, 43)]
    )
    @pytest.mark.parametrize("target", targets_3q)
    def test_builtin_validation_three_qubits(self, target, num_zeroed_wires, expected_count):
        wires = list(range(3))
        zeroed_wires = list(range(num_zeroed_wires))
        ops = ros.rot_opt_qsd(target, wires, zeroed_wires=zeroed_wires)
        assert ros.count_rotation_angles(ops) == expected_count

    @pytest.mark.with_validation
    @pytest.mark.parametrize(
        "num_zeroed_wires, expected_count", [(0, 256), (1, 192), (2, 176), (3, 172), (4, 171)]
    )
    @pytest.mark.parametrize("target", targets_4q)
    def test_builtin_validation_four_qubits(self, target, num_zeroed_wires, expected_count):
        wires = list(range(4))
        zeroed_wires = list(range(num_zeroed_wires))
        ops = ros.rot_opt_qsd(target, wires, zeroed_wires=zeroed_wires)
        assert ros.count_rotation_angles(ops) == expected_count

    @pytest.mark.without_validation
    @pytest.mark.parametrize("wires", [(1, 0, 2, -1), ("a", 5, "v", 2)])
    @pytest.mark.parametrize("target", targets)
    def test_wires(self, target, wires):
        n = len(bin(len(target))) - 3
        wires = wires[:n]
        ops = ros.rot_opt_qsd(target, wires)
        assert all(set(op.wires).issubset(set(wires)) for op in ops)

    @pytest.mark.without_validation
    @pytest.mark.parametrize("zeroed_wires", [[], [0], [1, 0]])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_queuing_matches_return_without_validation(self, n, zeroed_wires):
        if n == 1 and len(zeroed_wires) == 2:
            pytest.skip(reason="Can't have multiple zeroed wires on a single wire")
        target = unitary_group.rvs(2**n, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            ops = ros.rot_opt_qsd(target, range(n), zeroed_wires=zeroed_wires)

        assert ops == q.queue

    @pytest.mark.with_validation
    @pytest.mark.parametrize("zeroed_wires", [[], [0], [1, 0]])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_queuing_matches_return_with_validation(self, n, zeroed_wires):
        if n == 1 and len(zeroed_wires) == 2:
            pytest.skip(reason="Can't have multiple zeroed wires on a single wire")
        target = unitary_group.rvs(2**n, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            ops = ros.rot_opt_qsd(target, range(n), zeroed_wires=zeroed_wires)

        assert ops == q.queue

    @pytest.mark.with_validation
    @pytest.mark.parametrize("num_wires, wire_labels", [(2, ["a", "b"]), (3, [0, "c", 2])])
    def test_identity_decomposition(self, num_wires, wire_labels):
        """Test that the identity matrix is decomposed correctly."""
        identity = np.eye(2**num_wires)
        ros.rot_opt_qsd(identity, wire_labels)

    @pytest.mark.with_validation
    def test_toffoli_decomposition(self):
        """Test that the Toffoli gate is decomposed correctly."""
        toffoli = qml.Toffoli([0, 1, 2]).matrix()
        ros.rot_opt_qsd(toffoli, wires=[0, 1, 2])


@pytest.mark.with_validation
class TestRotOptSynthErrors:
    """Error handling and validation tests for rot_opt_qsd."""

    def test_incorrect_matrix_size_raises_error(self):
        """Test that a matrix of incorrect dimensions raises an AssertionError."""
        u = np.eye(4)  # 2-qubit matrix
        wires = [0, 1, 2]  # 3 wires

        # The internal validation `assert len(u) == 2**num_wires` should fail.
        with pytest.raises(AssertionError):
            ros.rot_opt_qsd(u, wires)

    def test_non_unitary_matrix_raises_error(self):
        """Test that a non-unitary matrix raises an AssertionError."""
        # Create a non-unitary matrix by zeroing out a row
        u = unitary_group.rvs(4, random_state=1)
        u[0, :] = 0

        # The internal validation `assert is_unitary(u)` should fail.
        with pytest.raises(AssertionError):
            ros.rot_opt_qsd(u, wires=[0, 1])
