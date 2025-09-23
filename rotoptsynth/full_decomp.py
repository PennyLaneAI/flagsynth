"""This module contains the main function to perform rotation-angle-optimal unitary synthesis,
called ``rot_opt_synth`` (at the moment)."""

# pylint: disable=too-many-locals
from collections.abc import Sequence, Hashable
from typing import Optional

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.op_math.decompositions.unitary_decompositions import (
    _compute_udv,
    _cossin_decomposition,
    _decompose_3_cnots,
)
from pennylane.wires import WiresLike

from .diag_decomps import attach_multiplexer_node, diag_decomp, split_diagonal
from .utils import ops_to_mat
from .validation import is_unitary, validation_enabled

def _decompose_first_mplx(a, b, wires, zeroed_wires):
    """Decompose the first multiplexer (in circuit ordering, not matrix multiplication ordering)
    of an AIII Cartan decomposition, using information about the first wire being zeroed or not."""

    if zeroed_wires:
        assert zeroed_wires[0] == wires[0] # Consistency check
        return rot_opt_synth(a, wires[1:], zeroed_wires[1:])

    u_sub, d_sub, v_sub = _compute_udv(a, b)
    diag_u_sub, other_ops_u_sub = diag_decomp(u_sub, wires[1:])
    v_sub = np.diag(diag_u_sub.data[0]) @ v_sub

    if validation_enabled():
        assert np.allclose(ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(d_sub) @ v_sub, a)
        assert np.allclose(
            ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(np.conj(d_sub)) @ v_sub, b
        )
    return [
        *rot_opt_synth(v_sub, wires[1:]),
        qml.SelectPauliRot(
            -2 * qml.math.angle(d_sub), wires[1:], target_wire=wires[0], rot_axis="Z"
        ), # pylint: disable=no-member
        *other_ops_u_sub,
    ]

def _rot_opt_synth_two_qubits(u, wires):
    with qml.queuing.AnnotatedQueue() as q:
        u, global_phase = qml.math.convert_to_su4(u, return_global_phase=True)
        global_phase += _decompose_3_cnots(u, wires, global_phase)
        qml.GlobalPhase(-global_phase)
    ops = q.queue
    if qml.queuing.QueuingManager.recording():
        for op in ops:
            qml.apply(op)
    return ops

def _rot_opt_synth_two_qubits_first_zeroed(u, wires):
    raise NotImplementedError
    # todo
    u = qml.math.expand_matrix(u, wires=wires, wire_order=wires[::-1])
    wires = wires[::-1]

    with qml.queuing.AnnotatedQueue() as q:
        u, global_phase = qml.math.convert_to_su4(u, return_global_phase=True)
        global_phase += _decompose_3_cnots(u, wires, global_phase)
        qml.GlobalPhase(-global_phase)
    ops = q.queue
    if qml.queuing.QueuingManager.recording():
        for op in ops:
            qml.apply(op)
    return ops

return _rot_opt_synth_two_qubits_all_zeroed(u, wires):
    raise NotImplementedError
    # todo

def rot_opt_synth(u: np.ndarray, wires: WiresLike, zeroed_wires: Optional[WiresLike]=None) -> Sequence[Operator]:
    r"""Unitary synthesis with optimal number of rotation angles.

    Args:
        u (np.ndarray): Unitary matrix to be decomposed.
        wires (qml.wires.WireLike): Wires on which the operators in the decomposition should act.
        zeroed_wires (qml.wires.WiresLike): Wires that are guaranteed to be in the state :math:`|0\rangle`.
            By default, no wires come with this assumption/guarantee. Must be contained
            in ``wires``.

    Returns:
        Sequence[qml.operation.Operator]: Operators in the rotation-angle-optimal decomposition.

    Queues:
        The same operators as are returned.

    Uses the RotOptSynth validation toggle.
    """
    if zeroed_wires is None:
        zeroed_wires = []
    elif not isinstance(zeroed_wires, list):
        zeroed_wires = list(zeroed_wires)

    if zeroed_wires:
        if not all(z in wires for z in zeroed_wires):
            raise ValueError(
                "All provided zeroed_wires must be part of the provided wires. "
                f"Got {zeroed_wires=} and {wires=}"
            )
        wires_list = list(wires)
        zeroed_ids = [wires_list.index(z) for z in zeroed_wires]
        if sorted(zeroed_ids) != list(range(len(zeroed_wires))):
            other_wires = [w for i, w in enumerate(wires) if i not in zeroed_ids]
            new_wires = zeroed_wires + other_wires
            u = qml.math.expand_matrix(u, wires=wires, wire_order=new_wires)
            return rot_opt_synth(u, wires=new_wires, zeroed_wires=zeroed_wires)
    else:

    num_wires = len(wires)
    assert len(u) == 2**num_wires
    if validation_enabled():
        assert is_unitary(u)

    if num_wires == 2:
        if len(zeroed_wires) == 1:
            return _rot_opt_synth_two_qubits_first_zeroed(u, wires)
        elif len(zeroed_wires) == 2:
            return _rot_opt_synth_two_qubits_all_zeroed(u, wires)
        return _rot_opt_synth_two_qubits(u, wires)

    p = len(u) // 2
    (k00, k01), mplx_angles_ry, (k10, k11) = _cossin_decomposition(u, p)

    with qml.QueuingManager.stop_recording():
        # Diag-decompose k00 and k01
        diag_k00, other_ops_00 = diag_decomp(k00, wires[1:])
        diag_k01, other_ops_01 = diag_decomp(k01, wires[1:])

        sub_diag, mplx_angles_rz = split_diagonal(
            np.concatenate([diag_k00.data[0], diag_k01.data[0]])
        )
        k10 = sub_diag[:, None] * k10
        k11 = sub_diag[:, None] * k11

        new_ops = [
            *_decompose_first_mplx(k10, k11, wires, zeroed_wires),
            qml.SelectPauliRot(2 * mplx_angles_ry, wires[1:], target_wire=wires[0], rot_axis="Y"),
            qml.SelectPauliRot(mplx_angles_rz, wires[1:], target_wire=wires[0], rot_axis="Z"),
            *attach_multiplexer_node(other_ops_00, other_ops_01, wires[0]),
        ]

    if validation_enabled():
        assert np.allclose(sub_diag * np.exp(-0.5j * mplx_angles_rz), diag_k00.data[0])
        assert np.allclose(sub_diag * np.exp(0.5j * mplx_angles_rz), diag_k01.data[0])

        u_rec = ops_to_mat(new_ops, wires)
        for _ in zeroed_wires:
            u_rec = np.take(u_rec.reshape((2,) * (2*num_wires)), 0, num_wires)
            u= np.take(u.reshape((2,) * (2*num_wires)), 0, num_wires)
        assert np.allclose(u, u_rec, atol=1e-7)

    if qml.QueuingManager.recording():
        for op in new_ops:
            qml.apply(op)

    return new_ops
