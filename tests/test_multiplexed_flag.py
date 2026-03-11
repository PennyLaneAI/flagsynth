"""Tests for flagsynth/multiplexed_flag.py"""

import pytest
import numpy as np
import pennylane as qml
from pennylane.ops.functions import assert_valid
from flagsynth.multiplexed_flag import MultiplexedFlag


class TestMultiplexedFlag:
    """Tests for `multiplexed_flag`."""

    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_standard_validity(self, seed, k):
        """Test standard operator validity."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 2**k))
        op = MultiplexedFlag(theta_z, theta_y, list(range(k + 1)))
        assert_valid(op, skip_differentiation=True)

    @pytest.mark.parametrize("seed", [825, 1285, 263, 42])
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
    def test_decomposition(self, seed, k):
        """Test basic usage."""
        np.random.seed(seed)
        theta_z, theta_y = np.random.random((2, 2**k))
        op = MultiplexedFlag(theta_z, theta_y, list(range(k + 1)))

        with qml.queuing.AnnotatedQueue() as q:
            decomp_0 = op.compute_decomposition(theta_z, theta_y, wires=list(range(k + 1)))
        assert len(q.queue) == 4 * 2**k if k > 0 else 2

        with qml.queuing.AnnotatedQueue() as q:
            decomp_1 = op.decomposition()
        assert len(q.queue) == 4 * 2**k if k > 0 else 2

        decomp_mat_0 = qml.matrix(decomp_0, wire_order=range(k + 1))
        decomp_mat_1 = qml.matrix(decomp_1, wire_order=range(k + 1))

        individual_mats = [
            qml.matrix(qml.RY(thy, 0) @ qml.RZ(thz, 0)) for thz, thy in zip(theta_z, theta_y)
        ]
        expected = qml.math.block_diag(individual_mats)
        assert np.allclose(expected, decomp_mat_0)
        assert np.allclose(expected, decomp_mat_1)
