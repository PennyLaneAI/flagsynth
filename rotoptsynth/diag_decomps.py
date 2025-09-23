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
from pennylane.wires import WiresLike

from .asym_decomp import asymmetric_two_qubit_decomp
from .utils import aiii_kak, ops_to_mat
from .validation import is_block_diagonal, is_unitary, validation_enabled

# static matrix
_cnot = qml.CNOT([0, 1]).matrix()


def _diag_decomp_two_qubits(u: np.ndarray, wires: WiresLike) -> tuple[Operator, list[Operator]]:
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
        wires (qml.wires.WiresLike): Wires on which the operators in the decomposition should act.

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
    with qml.QueuingManager.stop_recording():
        rz, _, c_op, d_op, *rest_ops, a_op, b_op, gphase = asymmetric_two_qubit_decomp(u_mod, wires)

        # Decompose A, B, C and D via Euler decomposition
        a_dec = one_qubit_decomposition(a_op.data[0], wire=wires[0])
        b_dec = one_qubit_decomposition(b_op.data[0], wire=wires[1])
        c_rz0, c_ry, c_rz1 = one_qubit_decomposition(c_op.data[0], wire=wires[0])
        d_rz0, d_ry, d_rz1 = one_qubit_decomposition(d_op.data[0], wire=wires[1])

        diag_ops = [qml.IsingZZ(rz.data[0], wires=wires), c_rz0, d_rz0, gphase]
        diagonal = np.diag(ops_to_mat(diag_ops, wires))
        diag_op = qml.DiagonalQubitUnitary(diagonal, wires=wires)
        other_ops = [c_ry, c_rz1, d_ry, d_rz1, *rest_ops, *a_dec, *b_dec]

    if validation_enabled():
        u_rec = ops_to_mat([diag_op] + other_ops, wires)
        assert np.allclose(u, u_rec, atol=1e-7)
        assert isinstance(_, qml.CNOT)

    if qml.QueuingManager.recording():
        qml.apply(diag_op)
        for op in other_ops:
            qml.apply(op)

    return diag_op, other_ops


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
        assert all(isinstance(op0, type(op1)) for op0, op1 in zip(ops0, ops1, strict=True))
        assert all(op0.wires == op1.wires for op0, op1 in zip(ops0, ops1, strict=True))
        assert all(
            op0.hyperparameters["rot_axis"] == op1.hyperparameters["rot_axis"]
            for op0, op1 in zip(ops0, ops1, strict=True)
            if isinstance(op0, qml.SelectPauliRot)
        )

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
        else:
            raise NotImplementedError(
                f"attaching multiplexer node to op of type {type(op0)} ({op0}) is not supported."
            )

    return new_ops


def split_diagonal(diag: np.ndarray) -> tuple[np.ndarray]:
    """Split a diagonal into a diagonal on one qubit less and the angles for an RZ multiplexer.
    Adapted from ``qml.DiagonalQubitUnitary.compute_decomposition`` to split off the first qubit
    instead of the last.

    Args:
        diag (np.ndarray): Diagonal to be split

    Returns:
        tuple(np.ndarray): Diagonal on one qubit less and angles to be passed
        into ``qml.SelectPauliRot`` so that a ``DiagonalQubitUnitary`` of the first return value
        multiplied with ``qml.SelectPauliRot`` of the second return value with ``rot_axis="Z"``
        yields the input diagonal.

    """
    angles = np.angle(diag)  # pylint: disable=no-member
    split = len(diag) // 2
    diff = angles[..., split:] - angles[..., :split]
    mean = (angles[..., :split] + angles[..., split:]) / 2
    return np.exp(1j * mean), diff


def diag_decomp(u: np.ndarray, wires: WiresLike) -> tuple[Operator, list[Operator]]:
    """Compute the decomposition of ``u`` into a diagonal, which is returned separately,
    and other operators. Uses recursion, with ``_diag_decomp_two_qubits`` as the base case.

    Args:
        u (np.ndarray): Unitary to be decomposed.
        wires (qml.wires.WiresLike): Wires that the operators in the decomposition should act on.

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
    if dim == 4:
        return _diag_decomp_two_qubits(u, wires)
    if dim == 8:
        sub_decomp = _diag_decomp_two_qubits
    else:
        sub_decomp = diag_decomp

    p = q = dim // 2
    k1, a, k2 = aiii_kak(u, p, q, validate=validation_enabled())
    if validation_enabled():
        assert is_unitary(k1) and is_block_diagonal(k1, p)
        assert is_unitary(k2) and is_block_diagonal(k2, p)

    with qml.queuing.QueuingManager.stop_recording():
        k1_0_diag_op, k1_0_ops = sub_decomp(k1[:p, :p], wires=wires[1:])
        k1_1_diag_op, k1_1_ops = sub_decomp(k1[p:, p:], wires=wires[1:])
        smaller_diag, multiplexer_angles_k1 = split_diagonal(
            np.concatenate([k1_0_diag_op.data[0], k1_1_diag_op.data[0]])
        )
        k2_0_diag_op, k2_0_ops = sub_decomp(np.diag(smaller_diag) @ k2[:p, :p], wires=wires[1:])
        k2_1_diag_op, k2_1_ops = sub_decomp(np.diag(smaller_diag) @ k2[p:, p:], wires=wires[1:])
        multiplexer_angles_a = -2 * np.arctan2(np.diag(a, k=p), np.diag(a)[:p])

        diagonal = np.concatenate([k2_0_diag_op.data[0], k2_1_diag_op.data[0]])
        diag_op = qml.DiagonalQubitUnitary(diagonal, wires=wires)
        other_ops = [
            *attach_multiplexer_node(k2_0_ops, k2_1_ops, multiplexer_wire=wires[0]),
            qml.SelectPauliRot(multiplexer_angles_a, wires[1:], target_wire=wires[0], rot_axis="Y"),
            qml.SelectPauliRot(
                multiplexer_angles_k1, wires[1:], target_wire=wires[0], rot_axis="Z"
            ),
            *attach_multiplexer_node(k1_0_ops, k1_1_ops, multiplexer_wire=wires[0]),
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
