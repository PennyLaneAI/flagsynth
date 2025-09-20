import pennylane as qml
import numpy as np

def test_ops_to_mat():
    """Test ops_to_mat."""
    ops = [qml.X(0), qml.Z(0), qml.Y(0)]
    mat = ops_to_mat(ops, wires=[0])
    assert np.allclose(mat, -1j * np.eye(2))

