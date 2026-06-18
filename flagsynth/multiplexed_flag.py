"""This module contains the operation class ``MultiplexedFlag`` to represent multiplexed
single-qubit flag circuits."""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane.decomposition import (
    add_decomps, register_resources, resource_rep, change_op_basis_resource_rep, adjoint_resource_rep, 
    controlled_resource_rep
)
from pennylane.ops import Prod
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


def _resources_split_into_phase_gradient(
    num_control_wires,
    num_angle_z_wires,
    num_angle_y_wires,
    num_phase_grad_wires,
    num_work_wires,
):
    """Resource estimation for the phase gradient decomposition of MultiplexedFlag.

    Mirrors the operator tree built by ``_decomp_split_into_phase_gradient`` exactly:

        change_op_basis(
            QROM,
            Prod(op1, op2),
        )

    with
        op1 = change_op_basis(Prod(cnots),         SemiAdder_rz)
        op2 = change_op_basis(Prod(cnots, Hy_ad),  SemiAdder_ry)

    where ``cnots`` are ``|0>``-controlled X gates (one per phase-gradient wire,
    represented as ``MultiControlledX`` with a single zero control value) and
    ``Hy_ad = Hadamard @ Adjoint(S)``.
    """

    num_target_wires = num_angle_z_wires + num_angle_y_wires
    max_precision = max(num_angle_z_wires, num_angle_y_wires)

    # The SemiAdders consume ``max_precision - 1`` work wires; the QROM is given
    # the remaining work wires (``work_wires[max_precision - 1:]``).
    semiadder_work = max_precision - 1
    qrom_work = num_work_wires - semiadder_work

    # QROM loaded once + unloaded once via change_op_basis.
    qrom_rep = resource_rep(
        qml.QROM,
        num_bitstrings=2**num_control_wires,
        num_control_wires=num_control_wires,
        num_target_wires=num_target_wires,
        num_work_wires=qrom_work,
        clean=False,
    )

    # A single |0>-controlled X is a MultiControlledX with one zero control value.
    mcx_rep = resource_rep(
        qml.MultiControlledX,
        num_control_wires=1,
        num_zero_control_values=1,
        num_work_wires=0,
        work_wire_type="borrowed",
    )

    # Y-basis change on the target: Hadamard @ Adjoint(S).
    hy_ad_rep = resource_rep(
        qml.ops.Prod,
        resources={
            resource_rep(qml.Hadamard): 1,
            adjoint_resource_rep(qml.S, base_params={}): 1,
        },
    )

    semiadder_rz_rep = resource_rep(
        qml.SemiAdder,
        num_x_wires=num_angle_z_wires,
        num_y_wires=num_angle_z_wires,
        num_work_wires=semiadder_work,
    )
    semiadder_ry_rep = resource_rep(
        qml.SemiAdder,
        num_x_wires=num_angle_y_wires,
        num_y_wires=num_angle_y_wires,
        num_work_wires=semiadder_work,
    )

    # op1: basis change is one CNOT per phase-gradient wire.
    op1_rep = change_op_basis_resource_rep(
        resource_rep(qml.ops.Prod, resources={mcx_rep: num_phase_grad_wires}),
        semiadder_rz_rep,
    )
    # op2: basis change is the same CNOTs plus the Y-basis change.
    op2_rep = change_op_basis_resource_rep(
        resource_rep(
            qml.ops.Prod,
            resources={mcx_rep: num_phase_grad_wires, hy_ad_rep: 1},
        ),
        semiadder_ry_rep,
    )

    return {
        change_op_basis_resource_rep(
            qrom_rep,
            resource_rep(qml.ops.Prod, resources={op1_rep: 1, op2_rep: 1}),
        ): 1,
    }

@register_resources(_resources_split_into_phase_gradient)
def _decomp_split_into_phase_gradient(
    phis_rz: np.ndarray,
    phis_ry: np.ndarray,
    control_wires: Wires,
    target_wire: Wires,
    angle_wires_rz: Wires,
    angle_wires_ry: Wires,
    phase_grad_wires: Wires,
    work_wires: Wires,
) -> Operator:
    """Phase gradient decomposition of a multiplexed RY @ RZ rotation.

    Applies RY(phis_ry) @ RZ(phis_rz) on the target, multiplexed on the control wires,
    using a single shared QROM and phase gradient register.

    The precision is implicitly defined by the length of the angle wire registers.
    """

    precision_rz = len(angle_wires_rz)
    precision_ry = len(angle_wires_ry)
    max_precision = max(precision_rz, precision_ry)

    # Digitize both angle sets
    binary_int_rz = qml.math.binary_decimals(phis_rz, precision_rz, unit=4 * np.pi)
    binary_int_ry = qml.math.binary_decimals(phis_ry, precision_ry, unit=4 * np.pi)
    binary_int = [list(a) + list(b) for a, b in zip(binary_int_rz, binary_int_ry)]

    # Shared QROM loading both angle registers
    qrom = qml.QROM(
        binary_int,
        control_wires,
        angle_wires_rz + angle_wires_ry,
        work_wires=work_wires[max_precision-1:],
        clean=True,
    )

    # CNOTs controlled by |0> on target
    cnots = [
        qml.ctrl(qml.X(wire), control=target_wire, control_values=[0])
        for wire in phase_grad_wires
    ]

    # Op1: RZ addition — basis is given by CNOTs
    op1 = qml.change_op_basis(
        qml.prod(*cnots),
        qml.SemiAdder(angle_wires_rz, phase_grad_wires[:precision_rz], work_wires=work_wires[:max_precision-1]),
    )

    # Op2: RY addition — basis is given by Y-basis change + CNOTs
    Hy_ad = qml.Hadamard(target_wire) @ qml.adjoint(qml.S(target_wire))
    compute = [Hy_ad] + cnots
    op2 = qml.change_op_basis(
        qml.prod(*compute[::-1]),
        qml.SemiAdder(angle_wires_ry, phase_grad_wires[:precision_ry], work_wires=work_wires[:max_precision-1]),
    )

    # Basis given by QROM
    op_phsg = qml.change_op_basis(
        qrom,
        qml.prod(op2, op1),
    )

    return op_phsg

def _resources_split_into_phase_gradient_partial(
    phis_rz: np.ndarray,
    phis_ry: np.ndarray,
    control_range,
    control_wires: Wires,
    target_wire: Wires,
    angle_wires_rz: Wires,
    angle_wires_ry: Wires,
    phase_grad_wires: Wires,
    work_wires: Wires,
) -> dict:
    """Resources for the phase-gradient decomposition of a RY @ RZ rotation
    controlled by a contiguous range of control states [k_start, k)."""
    precision_rz = len(angle_wires_rz)
    precision_ry = len(angle_wires_ry)
    max_precision = max(precision_rz, precision_ry)

    k_start, _ = control_range
    num_control = len(control_wires)
    num_pg = len(phase_grad_wires)

    # Work-wire splits mirror the qfunc exactly.
    semi_work = len(work_wires[: max_precision - 1])
    qrom_work = len(work_wires[max_precision - 1 :])
    num_bitstrings = len(phis_rz)

    # ctrl(X, control=target_wire, control_values=[0]) -> 1 control, 1 zero-ctrl.
    ctrl_x_rep = controlled_resource_rep(
        qml.X, base_params={}, num_control_wires=1, num_zero_control_values=1
    )

    # Hy_ad = Hadamard @ adjoint(S) on the target wire.
    hy_prod_rep = resource_rep(
        Prod,
        resources={resource_rep(qml.Hadamard): 1, adjoint_resource_rep(qml.S): 1},
    )

    # op1: change_op_basis(prod(*cnots), SemiAdder(rz, phase_grad[:precision_rz]))
    op1_compute = resource_rep(Prod, resources={ctrl_x_rep: num_pg})
    op1_target = resource_rep(
        qml.SemiAdder,
        num_x_wires=precision_rz,
        num_y_wires=len(phase_grad_wires[:precision_rz]),
        num_work_wires=semi_work,
    )
    op1_rep = change_op_basis_resource_rep(op1_compute, op1_target)

    # op2: change_op_basis(prod(*[Hy_ad] + cnots), SemiAdder(ry, phase_grad[:precision_ry]))
    op2_compute = resource_rep(Prod, resources={ctrl_x_rep: num_pg, hy_prod_rep: 1})
    op2_target = resource_rep(
        qml.SemiAdder,
        num_x_wires=precision_ry,
        num_y_wires=len(phase_grad_wires[:precision_ry]),
        num_work_wires=semi_work,
    )
    op2_rep = change_op_basis_resource_rep(op2_compute, op2_target)

    # target_op of the outer change_op_basis = prod(op2, op1)
    target_prod_rep = resource_rep(Prod, resources={op2_rep: 1, op1_rep: 1})

    # QROM (clean=False), loaded with the combined rz+ry binary angles.
    qrom_rep = resource_rep(
        qml.QROM,
        clean=False,
        num_bitstrings=num_bitstrings,
        num_control_wires=num_control,
        num_target_wires=precision_rz + precision_ry,
        num_work_wires=qrom_work,
    )

    # compute_op of the outer change_op_basis: bare QROM, or QROM shifted by a BasisState.
    if k_start == 0:
        compute_rep = qrom_rep
    else:
        basis_rep = resource_rep(qml.BasisState, num_wires=num_control)
        compute_rep = change_op_basis_resource_rep(basis_rep, qrom_rep)

    top_rep = change_op_basis_resource_rep(compute_rep, target_prod_rep)
    return {top_rep: 1}

@register_resources(_resources_split_into_phase_gradient_partial)
def _decomp_split_into_phase_gradient_partial(
    phis_rz: np.ndarray,
    phis_ry: np.ndarray,
    control_start,
    control_wires: Wires,
    target_wire: Wires,
    angle_wires_rz: Wires,
    angle_wires_ry: Wires,
    phase_grad_wires: Wires,
    work_wires: Wires,
) -> Operator:
    """Phase-gradient decomposition of a RY @ RZ rotation controlled by a
    contiguous range of control states ``[k_start, k)``.

    ``phis_rz`` / ``phis_ry`` are the non-trivial angles, in the order of the 
    active control states. ``control_start`` defines the first active control 
    state. Precision is set implicitly by the angle-wire lengths.
    """
    precision_rz = len(angle_wires_rz)
    precision_ry = len(angle_wires_ry)
    max_precision = max(precision_rz, precision_ry)

    # Digitize both angle sets and combine them for parallel loading
    binary_int_rz = qml.math.binary_decimals(phis_rz, precision_rz, unit=4 * np.pi)
    binary_int_ry = qml.math.binary_decimals(phis_ry, precision_ry, unit=4 * np.pi)
    binary_int = [list(a) + list(b) for a, b in zip(binary_int_rz, binary_int_ry)]

    qrom = qml.QROM(
        binary_int,
        control_wires,
        angle_wires_rz + angle_wires_ry,
        work_wires=work_wires[max_precision - 1:],
        clean=True,
    )

    if control_start == 0:
        qrom_shifted = qrom
    else:
        sub = qml.BasisState(control_start, control_wires)
        qrom_shifted = qml.change_op_basis(sub, qrom)

    # CNOTs controlled by |0> on target
    cnots = [
        qml.ctrl(qml.X(wire), control=target_wire, control_values=[0])
        for wire in phase_grad_wires
    ]

    # Op1: RZ addition
    op1 = qml.change_op_basis(
        qml.prod(*cnots),
        qml.SemiAdder(angle_wires_rz, phase_grad_wires[:precision_rz], work_wires=work_wires[:max_precision - 1]),
    )

    # Op2: RY addition (Y-basis change + CNOTs)
    Hy_ad = qml.Hadamard(target_wire) @ qml.adjoint(qml.S(target_wire))
    compute = [Hy_ad] + cnots
    op2 = qml.change_op_basis(
        qml.prod(*compute[::-1]),
        qml.SemiAdder(angle_wires_ry, phase_grad_wires[:precision_ry], work_wires=work_wires[:max_precision - 1]),
    )

    ops = qml.change_op_basis(qrom_shifted, qml.prod(op2, op1))
    
    return ops


add_decomps(
    MultiplexedFlag, 
    _decomp_cliff_rot, 
    _decomp_split_into_mux_rots, 
    _decomp_split_into_phase_gradient,
    _decomp_split_into_phase_gradient_partial)
