"""This module contains the operation class ``MultiplexedFlag`` to represent multiplexed
single-qubit flag circuits."""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from .linalg import mottonen


class HackyString:
    """A sneaky string class that appends itself one character at a time when added to a string,
    instead of being added entirely. This is used to create an operator label
    for MultiplexedFlag that differs on different wires of the operator.

    Args:
        string (str): String to be represented and added one character at a time.
    """

    def __init__(self, string):
        self._orig_str = string
        self._str = iter(string)

    def __radd__(self, other):
        return other + next(self._str)

    def replace(self, old, new, count=-1):
        """Replace character ``old`` by character ``new``. Behaves like (and uses) the
        corresponding method of the builtin ``str`` type."""
        return HackyString(self._orig_str.replace(old, new, count))


class MultiplexedFlag(Operation):
    """Multiplexed single-qubit flag circuit, represented through a sequence of ``RZ`` and ``RY``
    rotations each. A single-qubit flag circuit is given by an ``RZ`` gate followed by an ``RY``
    gate, and this operation represents the straight-forwardly multiplexed analogue.
    It can be decomposed through its ``compute_decomposition`` method (e.g. via ``qml.decompose``)
    or through ``decompose_mux_single_qubit_flags``.

    Args:
        theta_z (np.ndarray): Angles for multiplexed ``RZ`` rotation.
        theta_y (np.ndarray): Angles for multiplexed ``RY`` rotation.
        wires (list): Wires on which the circuit acts. The last wire is the target wire.

    This operation can be used to represent a "multiplexed" flag circuit with zero multiplexing
    wires, i.e. a non-multiplexed flag circuit, as well.
    """

    num_params = 2
    ndim_params = (1, 1)

    resource_keys = {"num_wires"}

    @property
    def resource_params(self):
        return {"num_wires": len(self.wires)}

    def __init__(self, theta_z, theta_y, wires):
        if isinstance(wires, int):
            wires = [wires]
        n = len(wires)
        if n == 1:
            theta_z = np.atleast_1d(theta_z)
            theta_y = np.atleast_1d(theta_y)
        assert len(theta_z) == len(theta_y) == 2 ** (n - 1)
        super().__init__(theta_z, theta_y, wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return HackyString("◑" * (len(self.wires) - 1) + "⚑")

    @staticmethod
    def compute_decomposition(theta_z, theta_y, wires):  # pylint: disable=arguments-differ
        if len(wires) == 1:
            return [qml.RZ(theta_z[0], wires), qml.RY(theta_y[0], wires)]

        ops = []
        if not np.allclose(theta_z, 0.0):
            ops.extend(mottonen(theta_z, wires[:-1], wires[-1], axis="Z"))
        if not np.allclose(theta_y, 0.0):
            ops.extend(mottonen(theta_y, wires[:-1], wires[-1], axis="Y"))
        if qml.QueuingManager.recording():
            for op in ops:
                qml.apply(op)

        return ops


def _resources_cliff_rot(num_wires):
    if num_wires == 1:
        return {resource_rep(qml.RZ): 1, resource_rep(qml.RY): 1}
    num_rot = 2 ** (num_wires - 1)
    num_cnot = 2 * 2 ** (num_wires - 1)
    return {
        resource_rep(qml.RZ): num_rot,
        resource_rep(qml.RY): num_rot,
        resource_rep(qml.CNOT): num_cnot,
    }


@register_resources(_resources_cliff_rot)
def _decomp_cliff_rot(theta_z, theta_y, wires):
    MultiplexedFlag.compute_decomposition(theta_z, theta_y, wires)


def _resources_split_into_mux_rots(num_wires):
    if num_wires == 1:
        return {resource_rep(qml.RZ): 1, resource_rep(qml.RY): 1}
    return {
        resource_rep(qml.SelectPauliRot, num_wires=num_wires, rot_axis="Z"): 1,
        resource_rep(qml.SelectPauliRot, num_wires=num_wires, rot_axis="Y"): 1,
    }


@register_resources(_resources_split_into_mux_rots)
def _decomp_split_into_mux_rots(theta_z, theta_y, wires):
    """This decomposition enables a transition into standard PennyLane operators without
    forcing full decomposition into single- and two-qubit operators."""
    if len(wires) == 1:
        qml.RZ(theta_z[0], wires)
        qml.RY(theta_y[0], wires)
    else:
        qml.SelectPauliRot(theta_z, control_wires=wires[:-1], target_wire=wires[-1], rot_axis="Z")
        qml.SelectPauliRot(theta_y, control_wires=wires[:-1], target_wire=wires[-1], rot_axis="Y")


add_decomps(MultiplexedFlag, _decomp_cliff_rot, _decomp_split_into_mux_rots)
