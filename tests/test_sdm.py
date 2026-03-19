"""Tests for flagsynth/sdm.py"""

import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from flagsynth.sdm import sdm

# pylint: disable=too-few-public-methods


class TestSdm:
    """Tests for the main function sdm."""

    @pytest.mark.parametrize("seed", [932, 2185])
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_main_usage(self, seed, n):
        """Test main usage."""
        targets = list(range(n))
        V = unitary_group.rvs(2**n, random_state=seed)
        ops = sdm(V, targets)
        rec_mat = qml.matrix(ops, wire_order=targets)
        assert np.allclose(rec_mat, V)
        exp_types = (qml.RZ, qml.RY, qml.CZ, qml.CNOT, qml.GlobalPhase)
        assert all(isinstance(op, exp_types) for op in ops), f"{set(type(op) for op in ops)}"
        num_rots = sum(isinstance(op, (qml.RZ, qml.RY)) for op in ops)
        num_cnots = sum(isinstance(op, (qml.CNOT, qml.CZ)) for op in ops)
        num_gphase = sum(isinstance(op, qml.GlobalPhase) for op in ops)
        exp_num_cnots = 4**n // 2 - 3 * (n + 2) * 2**n // 8 + n - 1

        assert num_rots == 4**n - 1
        assert num_gphase == 1
        assert num_cnots == exp_num_cnots
