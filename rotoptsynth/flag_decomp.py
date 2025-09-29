"""This module implements a specific unitary decomposition that pulls out a diagonal to
one side. It is built recursively upon the two-qubit case, which in turn is presented
in https://arxiv.org/pdf/quant-ph/0406176.
"""

# pylint: disable=too-many-locals
from collections.abc import Hashable, Sequence

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.op_math.decompositions import one_qubit_decomposition
from pennylane.math.decomposition import zyz_rotation_angles
from pennylane.wires import WiresLike

from .asym_decomp import asymmetric_two_qubit_decomp
from .select_su2 import SelectSU2
from .utils import aiii_kak, ops_to_mat
from .validation import is_block_diagonal, is_unitary, validation_enabled

# static matrix
_cnot = qml.CNOT([0, 1]).matrix()


def _diag_from_angles(angles):
    """Create diagonal from four angles of R_ZZ, R_Z^0, R_Z^1 and GlobalPhase."""
    rzz, rz_0, rz_1, gphase = angles
    return np.exp(
        -0.5j
        * np.array(
            [
                rzz + rz_0 + rz_1,
                -rzz + rz_0 - rz_1,
                -rzz - rz_0 + rz_1,
                rzz - rz_0 - rz_1,
            ]
        )
        - 1j * gphase
    )


def _flag_decomp_one_qubit(u: np.ndarray) -> tuple[Operator, list[Operator]]:
    """Compute the decomposition of a single-qubit unitary into a diagonal and a remaining
    decomposition:

    ```
    0: ─Diag──RY──RZ─┤
    ```

    This decomposition is trivially derived from a ZYZ decomposition, but it is adjusted to the
    syntax of ``flag_decomp``.

    Args:
        u (np.ndarray): Unitary matrix to be decomposed.

    Returns:
        tuple[qml.operation.Operator, list[qml.operation.Operator]]: Diagonal operator
        (of type ``DiagonalQubitUnitary``) and list of other operators. The decomposition
        is provided in the circuit decomposition order, the inverse of the matrix multiplication
        order.

    Queues:
        The same operators as are returned.

    Uses the RotOptSynth validation toggle.
    """
    rz0_angle, *other_data, gphase = zyz_rotation_angles(u, return_global_phase=True)
    diagonal = np.exp([-0.5j * rz0_angle + 1j * gphase, 0.5j * rz0_angle + 1j * gphase])
    if validation_enabled():
        diag_op = qml.DiagonalQubitUnitary(diagonal, [0])
        other_ops = [qml.RY(other_data[0], 0), qml.RZ(other_data[1], 0)]
        u_rec = ops_to_mat([diag_op] + other_ops, [0])
        assert np.allclose(u, u_rec, atol=1e-7)

    return diagonal, other_data


def _flag_decomp_two_qubits(u: np.ndarray) -> tuple[Operator, list[Operator]]:
    """Compute the decomposition of a two-qubit unitary into a diagonal and a remaining
    decomposition:

    ```
    0: ─╭Diag──RY──RZ─╭●──RX─╭●──RZ──RY──RZ─┤
    1: ─╰Diag──RY──RZ─╰X──RZ─╰X──RZ──RY──RZ─┤
    ```

    This decomposition is reported in Theorem 14 of the Quantum Shannon decomposition paper
    (Shende et al. https://arxiv.org/pdf/quant-ph/0406176). It is built on top of
    ``asymmetric_two_qubit_decomp``, an implementation of Theorem VI.3 and Fig. 3 of
    https://arxiv.org/pdf/quant-ph/0308033.

    Args:
        u (np.ndarray): Unitary matrix to be decomposed.

    Returns:
        tuple[qml.operation.Operator, list[qml.operation.Operator]]: Diagonal operator
        (of type ``DiagonalQubitUnitary``) and list of other operators. The decomposition
        is provided in the circuit decomposition order, the inverse of the matrix multiplication
        order.

    Queues:
        The same operators as are returned.

    Uses the RotOptSynth validation toggle.
    """
    if validation_enabled():
        assert u.shape == (4, 4)
        assert is_unitary(u)

    u_mod = u @ _cnot  # Optimize cnot multiplication away
    psi_rz, c, d, theta_rx, phi_rz, a, b, gphase = asymmetric_two_qubit_decomp(u_mod)

    # Decompose A, B, C and D via Euler decomposition
    c_rz0_angle, c_ry_angle, c_rz1_angle = zyz_rotation_angles(c)
    d_rz0_angle, d_ry_angle, d_rz1_angle = zyz_rotation_angles(d)

    diagonal = _diag_from_angles([psi_rz, c_rz0_angle, d_rz0_angle, gphase])
    other_data = [c_ry_angle, c_rz1_angle, d_ry_angle, d_rz1_angle, theta_rx, phi_rz, a, b]

    if validation_enabled():
        diag_op = qml.DiagonalQubitUnitary(diagonal, [0, 1])
        other_ops = [
            qml.RY(c_ry_angle, 0), qml.RZ(c_rz1_angle, 0),
            qml.RY(d_ry_angle, 1), qml.RZ(d_rz1_angle, 1),
            qml.CNOT([0, 1]), 
            qml.RX(theta_rx, 0),
            qml.RZ(phi_rz, 1),
            qml.CNOT([0, 1]),
            qml.QubitUnitary(a, 0),
            qml.QubitUnitary(b, 1),
        ]
        u_rec = ops_to_mat([diag_op] + other_ops, [0, 1])
        assert np.allclose(u, u_rec, atol=1e-7)

    return diagonal, other_data


def attach_multiplexer_node(
    ops0: Sequence[Operator], ops1: Sequence[Operator], multiplexer_wire: Hashable
) -> list[Operator]:
    """Create multiplexed operations from two sequences of operators of same type sequences.

    Args:
        ops0 (Sequence[qml.operation.Operator]): First sequence of operators.
        ops1 (Sequence[qml.operation.Operator]): Second sequence of operators. Needs to have
            the same length as ``ops0`` and ``ops1[i]`` needs to be of the same type
            as ``ops0[i]``, and act on the same wires, for all applicable ``i``.
        multiplexer_wire (Hashable): Label for the multiplexing (i.e. controlling) wire.

    Returns:
        list[qml.operation.Operator]: Multiplexed operators.

    Queues:
        Same multiplexed operators.

    .. note::

        Currently, only the following operator types are supported:
        ``qml.RX``,
        ``qml.RY``,
        ``qml.RZ``,
        ``qml.QubitUnitary`` with a single wire,
        ``qml.CNOT``,
        ``qml.CZ``,
        ``qml.SelectPauliRot``.

    **Examples**

    A pair of rotation gates of type, say, ``qml.RY`` will be multiplexed into a
    ``qml.SelectPauliRot``:

    >>> import rotoptsynth as ros
    >>> ros.attach_multiplexer_node([qml.RY(0.6, 0)], [qml.RY(-0.1, 0)], "mpx")
    [SelectPauliRot(array([ 0.6, -0.1]), wires=['mpx', 0])]

    A pair of ``qml.SelectPauliRot`` gates will simply obtain a new multiplexing wire:

    >>> ops0 = [qml.SelectPauliRot([0.6, 0.7], [0], target_wire="target", rot_axis="X")]
    >>> ops1 = [qml.SelectPauliRot([0.2, -0.5], [0], target_wire="target", rot_axis="X")]
    >>> ros.attach_multiplexer_node(ops0, ops1, "new mpx")
    [SelectPauliRot(array([ 0.6,  0.7,  0.2, -0.5]), wires=['new mpx', 0, 'target'])]

    A pair of static gates, like ``qml.CNOT``, simply leads to the same gate:

    >>> ros.attach_multiplexer_node([qml.CNOT([0, 1])], [qml.CNOT([0, 1])], "mpx")
    [CNOT(wires=[0, 1])]

    Uses the RotOptSynth validation toggle.
    """
    if validation_enabled():
        for op0, op1 in zip(ops0, ops1, strict=True):
            assert isinstance(op0, type(op1))
            assert op0.wires == op1.wires
            if isinstance(op0, qml.SelectPauliRot):
                assert op0.hyperparameters["rot_axis"] == op1.hyperparameters["rot_axis"]
            if isinstance(op0, (qml.QubitUnitary, SelectSU2)):
                assert np.allclose(np.linalg.det(op0.data[0]), 1.)
                assert np.allclose(np.linalg.det(op1.data[0]), 1.)

    new_ops = []
    for op0, op1 in zip(ops0, ops1):
        if isinstance(op0, (qml.RX, qml.RY, qml.RZ)):
            new_ops.append(
                qml.SelectPauliRot(
                    angles=[op0.data[0], op1.data[0]],
                    control_wires=[multiplexer_wire],
                    target_wire=op0.wires[0],
                    rot_axis=op0.name[-1],
                )
            )
        elif isinstance(op0, qml.QubitUnitary):
            new_ops.append(
                SelectSU2(
                    matrices=np.stack([op0.data[0], op1.data[0]]),
                    control_wires=[multiplexer_wire],
                    target_wire=op0.wires[0],
                )
            )
        elif isinstance(op0, (qml.CNOT, qml.CZ)):
            new_ops.append(op0)
            if qml.queuing.QueuingManager.recording():
                qml.apply(op0)
        elif isinstance(op0, qml.SelectPauliRot):
            new_ops.append(
                qml.SelectPauliRot(
                    angles=np.concatenate([op0.data[0], op1.data[0]]),
                    control_wires=[multiplexer_wire] + list(op0.hyperparameters["control_wires"]),
                    target_wire=op0.hyperparameters["target_wire"],
                    rot_axis=op0.hyperparameters["rot_axis"],
                )
            )
        elif isinstance(op0, SelectSU2):
            new_ops.append(
                SelectSU2(
                    matrices=np.concatenate([op0.data[0], op1.data[0]]),
                    control_wires=[multiplexer_wire] + list(op0.hyperparameters["control_wires"]),
                    target_wire=op0.hyperparameters["target_wire"],
                )
            )
        else:
            raise NotImplementedError(
                f"attaching multiplexer node to op of type {type(op0)} ({op0}) is not supported."
            )

    return new_ops


def balance_diagonal(diag0: np.ndarray, diag1: np.ndarray) -> tuple[np.ndarray]:
    """Balance diagonals into a diagonal on one qubit less and the angles for an RZ multiplexer.
    Adapted from ``qml.DiagonalQubitUnitary.compute_decomposition`` to split off the first qubit
    instead of the last, and to adjust for typical use case of input diagonal being given in two
    halves.

    Args:
        diag0 (np.ndarray): First half of the diagonal to be split/balanced
        diag1 (np.ndarray): Second half of the diagonal to be split/balanced

    Returns:
        tuple(np.ndarray): Diagonal on one qubit less and angles to be passed
        into ``qml.SelectPauliRot`` so that a ``DiagonalQubitUnitary`` of the first return value
        multiplied with ``qml.SelectPauliRot`` of the second return value with ``rot_axis="Z"``
        yields the (concatenated) inputs.

    """
    angles0 = np.angle(diag0)
    angles1 = np.angle(diag1)
    diff = angles1 - angles0
    mean = (angles0 + angles1) / 2
    return np.exp(1j * mean), diff


def flag_decomp(u: np.ndarray, wires: WiresLike, base_case_dim: int=4) -> tuple[Operator, list[Operator]]:
    """Compute the decomposition of ``u`` into a diagonal, which is returned separately,
    and other operators. Uses recursion, with ``_flag_decomp_two_qubits`` as the base case by
    default. By setting ``base_case_dim=2``, the single-qubit flag decomposition can be used.

    Args:
        u (np.ndarray): Unitary to be decomposed.
        wires (qml.wires.WiresLike): Wires that the operators in the decomposition should act on.
        base_case_dim (int): Dimension to consider as the base case for the recursive flag decomposition.
            Currently, ``2`` for the single qubit base case and ``4`` for the two-qubit
            base case are supported.

    Returns:
        tuple[qml.operation.Operator, list[qml.operation.Operator]]: Diagonal operator
        (of type ``DiagonalQubitUnitary``) and list of other operators. The decomposition
        is provided in the circuit decomposition order, the inverse of the matrix multiplication
        order.

    Queues:
        The same operators as are returned.

    Uses the RotOptSynth validation toggle.
    """
    dim = len(u)
    if dim == 2:
        diagonal, other_data = _flag_decomp_one_qubit(u)
        diag_op = qml.DiagonalQubitUnitary(diagonal, wires)
        other_ops = [qml.RY(other_data[0], wires[0]), qml.RZ(other_data[1], wires[0])]
        return diag_op, other_ops

    if dim == 4 and base_case_dim == 4:
        diagonal, other_data = _flag_decomp_two_qubits(u)
        diag_op = qml.DiagonalQubitUnitary(diagonal, wires)
        other_ops = [
            qml.RY(other_data[0], wires[0]), qml.RZ(other_data[1], wires[0]),
            qml.RY(other_data[2], wires[1]), qml.RZ(other_data[3], wires[1]),
            qml.CNOT(wires), 
            qml.RX(other_data[4], wires[0]),
            qml.RZ(other_data[5], wires[1]),
            qml.CNOT(wires),
            qml.QubitUnitary(other_data[6], wires[0]),
            qml.QubitUnitary(other_data[7], wires[1]),
        ]
        return diag_op, other_ops

    p = q = dim // 2
    k0, a, k1 = aiii_kak(u, p, q, validate=validation_enabled())
    if validation_enabled():
        assert is_unitary(k0) and is_block_diagonal(k0, p)
        assert is_unitary(k1) and is_block_diagonal(k1, p)

    with qml.queuing.QueuingManager.stop_recording():
        k0_0_diag_op, k0_0_ops = flag_decomp(k0[:p, :p], wires=wires[1:], base_case_dim=base_case_dim)
        k0_1_diag_op, k0_1_ops = flag_decomp(k0[p:, p:], wires=wires[1:], base_case_dim=base_case_dim)
        smaller_diag, multiplexer_angles_k0 = balance_diagonal(
            k0_0_diag_op.data[0], k0_1_diag_op.data[0]
        )
        k1_0_diag_op, k1_0_ops = flag_decomp(np.diag(smaller_diag) @ k1[:p, :p], wires=wires[1:], base_case_dim=base_case_dim)
        k1_1_diag_op, k1_1_ops = flag_decomp(np.diag(smaller_diag) @ k1[p:, p:], wires=wires[1:], base_case_dim=base_case_dim)
        multiplexer_angles_a = -2 * np.arctan2(np.diag(a, k=p), np.diag(a)[:p])

        diagonal = np.concatenate([k1_0_diag_op.data[0], k1_1_diag_op.data[0]])
        diag_op = qml.DiagonalQubitUnitary(diagonal, wires=wires)
        other_ops = [
            *attach_multiplexer_node(k1_0_ops, k1_1_ops, multiplexer_wire=wires[0]),
            qml.SelectPauliRot(multiplexer_angles_a, wires[1:], target_wire=wires[0], rot_axis="Y"),
            qml.SelectPauliRot(
                multiplexer_angles_k0, wires[1:], target_wire=wires[0], rot_axis="Z"
            ),
            *attach_multiplexer_node(k0_0_ops, k0_1_ops, multiplexer_wire=wires[0]),
        ]
        if validation_enabled():
            u_rec = ops_to_mat([diag_op] + other_ops, wires)
            assert np.allclose(
                u, u_rec, atol=1e-7
            ), f"Maximal difference (abs): {np.max(np.abs(u-u_rec))}"

    if qml.QueuingManager.recording():
        qml.apply(diag_op)
        for op in other_ops:
            qml.apply(op)

    return diag_op, other_ops
