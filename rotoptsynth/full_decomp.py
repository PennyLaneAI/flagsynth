from collections.abc import Sequence

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.op_math.decompositions.unitary_decompositions import (
    _compute_udv, _cossin_decomposition, _decompose_3_cnots)
from pennylane.wires import WiresLike

from .diag_decomps import attach_multiplexer_node, diag_decomp, split_diagonal
from .utils import ops_to_mat
from .validation import is_unitary, validation_enabled


def rot_opt_synth(u: np.ndarray, wires: WiresLike) -> Sequence[Operator]:
    r"""Unitary synthesis with optimal number of rotation angles.

    Args:
        u (np.ndarray): Unitary matrix to be decomposed.
        wires (qml.wires.WireLike): Wires on which the operators in the decomposition should act.

    Returns:
        Sequence[qml.operation.Operator]: Operators in the rotation-angle-optimal decomposition.

    Queues:
        The same operators as are returned.

    Uses the RotOptSynth validation toggle.
    """
    num_wires = len(wires)
    assert len(u) == 2**num_wires
    if validation_enabled():
        assert is_unitary(u)

    if num_wires == 2:
        with qml.queuing.AnnotatedQueue() as q:
            u, global_phase = qml.math.convert_to_su4(u, return_global_phase=True)
            global_phase += _decompose_3_cnots(u, wires, global_phase)
            qml.GlobalPhase(-global_phase)
        ops = q.queue
        if qml.queuing.QueuingManager.recording():
            for op in ops:
                qml.apply(op)
        return ops

    p = len(u) // 2
    (K00, K01), mplx_angles_ry, (K10, K11) = _cossin_decomposition(u, p)

    with qml.QueuingManager.stop_recording():
        # Diag-decompose K00 and K01
        diag_k00, other_ops_00 = diag_decomp(K00, wires[1:])
        diag_k01, other_ops_01 = diag_decomp(K01, wires[1:])

        sub_diag, mplx_angles_rz = split_diagonal(
            np.concatenate([diag_k00.data[0], diag_k01.data[0]])
        )
        # todo: make multiplication more efficient
        K10 = np.diag(sub_diag) @ K10
        K11 = np.diag(sub_diag) @ K11

        u_sub, d_sub, v_sub = _compute_udv(K10, K11)
        diag_u_sub, other_ops_u_sub = diag_decomp(u_sub, wires[1:])
        v_sub = np.diag(diag_u_sub.data[0]) @ v_sub

        new_ops = [
            *rot_opt_synth(v_sub, wires[1:]),
            qml.SelectPauliRot(
                -2 * qml.math.angle(d_sub), wires[1:], target_wire=wires[0], rot_axis="Z"
            ),
            *other_ops_u_sub,
            qml.SelectPauliRot(2 * mplx_angles_ry, wires[1:], target_wire=wires[0], rot_axis="Y"),
            qml.SelectPauliRot(mplx_angles_rz, wires[1:], target_wire=wires[0], rot_axis="Z"),
            *attach_multiplexer_node(other_ops_00, other_ops_01, wires[0]),
        ]

    if validation_enabled():
        assert np.allclose(sub_diag * np.exp(-0.5j * mplx_angles_rz), diag_k00.data[0])
        assert np.allclose(sub_diag * np.exp(0.5j * mplx_angles_rz), diag_k01.data[0])
        assert np.allclose(ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(d_sub) @ v_sub, K10)
        assert np.allclose(
            ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(np.conj(d_sub)) @ v_sub, K11
        )

        u_rec = ops_to_mat(new_ops, wires)
        assert np.allclose(u, u_rec, atol=1e-7)

    if qml.QueuingManager.recording():
        for op in new_ops:
            qml.apply(op)

    return new_ops
