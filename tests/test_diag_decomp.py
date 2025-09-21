import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros
from rotoptsynth.diag_decomps import _diag_decomp_two_qubits


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


class TestDiagDecompTwoQubits:

    @pytest.mark.parametrize("target", targets_2q)
    def test_builtin_validation(self, target):
        ros.enable_validation() # todo: make easy to reuse
        diag_op, other_ops = _diag_decomp_two_qubits(target, [0, 1])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)
        assert len(other_ops) == 14 # 2 CNOTs + 12 parametrized rotations

    @pytest.mark.parametrize("wires", [(1, 0), (0, 1), ("a", 5)])
    @pytest.mark.parametrize("target", targets_2q)
    def test_wires(self, target, wires):
        ros.disable_validation() # todo: make easy to reuse
        diag_op, other_ops = _diag_decomp_two_qubits(target, wires)
        assert diag_op.wires == wires
        assert all(set(op.wires).issubset(set(wires)) for op in other_ops)

    def queuing_matches_return(self):
        ros.disable_validation() # todo: make easy to reuse
        target = unitary_group.rvs(4, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            diag_op, other_ops = _diag_decomp_two_qubits(target, [0, 1])

        assert [diag_op] + other_ops == q.queue

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

class TestDiagDecomp:

    @pytest.mark.parametrize("target", targets_2q)
    def test_builtin_validation_two_qubits(self, target):
        ros.enable_validation() # todo: make easy to reuse
        diag_op, other_ops = ros.diag_decomp(target, [0, 1])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)

    @pytest.mark.parametrize("target", targets_3q)
    def test_builtin_validation_three_qubits(self, target):
        ros.enable_validation() # todo: make easy to reuse
        diag_op, other_ops = ros.diag_decomp(target, [0, 1, 2])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)

    @pytest.mark.parametrize("target", targets_4q)
    def test_builtin_validation_four_qubits(self, target):
        ros.enable_validation() # todo: make easy to reuse
        diag_op, other_ops = ros.diag_decomp(target, [0, 1, 2, 3])
        assert isinstance(diag_op, qml.DiagonalQubitUnitary)

    @pytest.mark.parametrize("wires", [(1, 0, 2, -1), ("a", 5, "v", 2)])
    @pytest.mark.parametrize("target", targets)
    def test_wires(self, target, wires):
        ros.disable_validation() # todo: make easy to reuse
        n = len(bin(len(target))) - 3
        wires = wires[:n]
        print(wires)
        diag_op, other_ops = ros.diag_decomp(target, wires)
        assert diag_op.wires == wires
        assert all(set(op.wires).issubset(set(wires)) for op in other_ops)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def queuing_matches_return(self, n):
        ros.disable_validation() # todo: make easy to reuse
        target = unitary_group.rvs(2**n, random_state=8364)
        with qml.queuing.AnnotatedQueue() as q:
            diag_op, other_ops = ros.diag_decomp(target, range(n))

        assert [diag_op] + other_ops == q.queue


