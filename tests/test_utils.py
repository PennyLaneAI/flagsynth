import pennylane as qml
import numpy as np
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
