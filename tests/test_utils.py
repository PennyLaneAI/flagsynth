import pytest
import pennylane as qml
import numpy as np
from scipy.stats import unitary_group
import rotoptsynth as ros


class TestOpsToMat:

    def test_op_order(self):
        """Test that the operator order is used correctly."""
        ops = [qml.X(0), qml.Z(0), qml.Y(0)]
        mat = ros.utils.ops_to_mat(ops, wire_order=[0])
        assert np.allclose(mat, 1j * np.eye(2))

    def test_wire_order(self):
        """Test that wire order is respected."""
        ops = [qml.IsingXY(0.612, [0, 1]), qml.H(0)]
        mat_01 = ros.utils.ops_to_mat(ops, wire_order=[0, 1])
        mat_10 = ros.utils.ops_to_mat(ops, wire_order=[1, 0])
        assert np.allclose(qml.math.expand_matrix(mat_01, wires=[0, 1], wire_order=[1,0]), mat_10)


FIXED_COUNT_OPS = [
    (qml.RX(0.1, 0), 1),
    (qml.RY(0.2, 0), 1),
    (qml.RZ(0.3, 0), 1),
    (qml.CNOT([0, 1]), 0),
    (qml.CZ([0, 1]), 0),
    (qml.GlobalPhase(1.23), 1),
]

def _make_special_unitary(dim, seed):
    U = unitary_group.rvs(dim, random_state=seed)
    U *= np.exp(-1j * np.angle(np.linalg.det(U)) / dim)
    return U


LAMBDA_COUNT_OPS = [
    # 1-qubit QubitUnitary: 4**1 - 1 = 3
    (qml.QubitUnitary(_make_special_unitary(2, 42), 0), 3),
    # 2-qubit QubitUnitary: 4**2 - 1 = 15
    (qml.QubitUnitary(_make_special_unitary(4, 43), [0, 1]), 15),
    # 2-wire SelectPauliRot: 2**(2-1) = 2
    (qml.SelectPauliRot([0.1, 0.2], control_wires=[0], target_wire=1), 2),
    # 3-wire SelectPauliRot: 2**(3-1) = 4
    (qml.SelectPauliRot([0.1, 0.2, 0.3, 0.4], control_wires=[0, 2], target_wire=1), 4),
    # 1-wire DiagonalQubitUnitary: 2**1 = 2
    (qml.DiagonalQubitUnitary([1, 1j], wires=0), 2),
    # 2-wire DiagonalQubitUnitary: 2**2 = 4
    (qml.DiagonalQubitUnitary([1, 1j, -1, -1j], wires=[0, 1]), 4),
]

class TestCountRotationAngles:
    """Tests for the count_rotation_angles function."""

    @pytest.mark.parametrize("op, expected_count", FIXED_COUNT_OPS)
    def test_fixed_count_operators(self, op, expected_count):
        """Test operators with a fixed number of rotation angles."""
        assert ros.count_rotation_angles([op]) == expected_count

    @pytest.mark.parametrize("op, expected_count", LAMBDA_COUNT_OPS)
    def test_lambda_count_operators(self, op, expected_count):
        """Test operators where the angle count is calculated by a lambda."""
        assert ros.count_rotation_angles([op]) == expected_count

    def test_mixed_operators_list(self):
        """Test a list containing a mix of different operators."""
        ops = [
            qml.RX(0.5, 0),          # 1
            qml.CNOT([0, 1]),        # 0
            qml.QubitUnitary(_make_special_unitary(2, 1), 2),  # 3
            qml.SelectPauliRot([0.1, 0.2], control_wires=[0], target_wire=1) # 2
        ]
        # Expected total: 1 + 0 + 3 + 2 = 6
        assert ros.count_rotation_angles(ops) == 6

    def test_empty_list(self):
        """Test that an empty list of operators returns a count of 0."""
        assert ros.count_rotation_angles([]) == 0

    def test_qubit_unitary_with_non_unit_determinant_raises_assertion_error(self):
        """Test that QubitUnitary with det != 1 raises an AssertionError."""
        non_unit_det_matrix = np.array([[1, 0], [0, 2]])
        ops = [qml.QubitUnitary(non_unit_det_matrix, wires=0)]

        with pytest.raises(AssertionError):
            ros.count_rotation_angles(ops)
