import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros

targets = [
    np.eye(4),
    1j * np.eye(4),
    qml.CNOT([0, 1]).matrix(),
    qml.X(0).matrix([1, 0]),
    qml.X(1).matrix([1, 0]),
    qml.IsingZZ(0.7235, [0, 1]).matrix(),
    unitary_group.rvs(4, random_state=951210),
    unitary_group.rvs(4, random_state=835610),
    unitary_group.rvs(4, random_state=62138),
    unitary_group.rvs(4, random_state=1561),
    qml.Rot(0.12, 2.1, -0.7, 0).matrix([1, 0]),
]


class TestAsymmetricTwoQubitDecomp:
    """Tests for asymmetric_two_qubit_decomp."""

    @pytest.mark.parametrize("target", targets)
    def test_builtin_validation(self, target):
        ros.enable_validation() # todo: make easy to reuse
        ops = ros.asymmetric_two_qubit_decomp(target, [0, 1])
        assert len(ops) == 11
        assert sum(isinstance(op, qml.CNOT) for op in ops) == 3

    @pytest.mark.parametrize("wires", [(1, 0), (0, 1), ("a", 5)])
    @pytest.mark.parametrize("target", targets)
    def test_wires(self, target, wires):
        ros.disable_validation() # todo: make easy to reuse
        ops = ros.asymmetric_two_qubit_decomp(target, wires)
        assert all(set(op.wires).issubset(set(wires)) for op in ops)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def queuing_matches_return(self, n):
        ros.disable_validation() # todo: make easy to reuse
        target = unitary_group.rvs(2**n, random_state=8152)
        with qml.queuing.AnnotatedQueue() as q:
            ops = ros.asymmetric_two_qubit_decomp(target, range(n))

        assert ops == q.queue
