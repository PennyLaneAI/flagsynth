"""This module contains the main function to perform rotation-angle-optimal unitary synthesis,
called ``rot_opt_qsd`` (at the moment)."""

# pylint: disable=too-many-locals
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.op_math.decompositions.unitary_decompositions import (
    _compute_udv,
    _cossin_decomposition,
    _decompose_3_cnots,
    one_qubit_decomposition,
)
from pennylane.wires import WiresLike

from .flag_decomp import attach_multiplexer_node, flag_decomp, balance_diagonal
from .utils import ops_to_mat
from .validation import is_unitary, validation_enabled


def _decompose_first_mplx(a, b, wires, zeroed_wires):
    """Decompose the first multiplexer (in circuit ordering, not matrix multiplication ordering)
    of an AIII Cartan decomposition, using information about the first wire being zeroed or not."""

    if zeroed_wires:
        assert wires[0] in zeroed_wires  # Consistency check
        new_zeroed_wires = [z for z in zeroed_wires if z != wires[0]]
        return rot_opt_qsd(a, wires[1:], new_zeroed_wires)

    u_sub, d_sub, v_sub = _compute_udv(a, b)
    diag_u_sub, other_ops_u_sub = flag_decomp(u_sub, wires[1:])
    v_sub = np.diag(diag_u_sub.data[0]) @ v_sub

    if validation_enabled():
        assert np.allclose(ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(d_sub) @ v_sub, a)
        assert np.allclose(
            ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(np.conj(d_sub)) @ v_sub, b
        )
    return [
        *rot_opt_qsd(v_sub, wires[1:]),
        qml.SelectPauliRot(-2 * np.angle(d_sub), wires[1:], target_wire=wires[0], rot_axis="Z"),
        *other_ops_u_sub,
    ]


def _rot_opt_qsd_one_qubit(u, wire, zeroed):
    with qml.QueuingManager.stop_recording():
        new_ops = one_qubit_decomposition(u, wire, return_global_phase=True)
        if zeroed:
            rz, *other_ops, gphase = new_ops
            new_ops = other_ops + [qml.GlobalPhase(gphase.data[0] + 0.5 * rz.data[0])]
    if qml.QueuingManager.recording():
        for op in new_ops:
            qml.apply(op)
    return new_ops


def _rot_opt_qsd_two_qubits(u, wires):
    with qml.queuing.AnnotatedQueue() as q:
        u, global_phase = qml.math.convert_to_su4(u, return_global_phase=True)
        global_phase += _decompose_3_cnots(u, wires, global_phase)
        qml.GlobalPhase(-global_phase)
    ops = q.queue
    if qml.queuing.QueuingManager.recording():
        for op in ops:
            qml.apply(op)
    return ops


def _validate_and_arrange_zeroed_wires(u: np.ndarray, wires: WiresLike, zeroed_wires: WiresLike):
    if not all(z in wires for z in zeroed_wires):
        raise ValueError(
            "All provided zeroed_wires must be part of the provided wires. "
            f"Got {zeroed_wires=} and {wires=}"
        )
    wires_list = list(wires)
    zeroed_ids = [wires_list.index(z) for z in zeroed_wires]
    if sorted(zeroed_ids) != list(range(len(zeroed_wires))):
        # zeroed_wires are not in the first positions. Permute matrix and wires to make change that
        other_wires = [w for i, w in enumerate(wires) if i not in zeroed_ids]
        new_wires = zeroed_wires + other_wires
        u = qml.math.expand_matrix(u, wires=wires, wire_order=new_wires)
    else:
        new_wires = wires

    return u, new_wires


def rot_opt_qsd(
    u: np.ndarray, wires: WiresLike, zeroed_wires: Optional[WiresLike] = None
) -> Sequence[Operator]:
    r"""Unitary synthesis with optimal number of rotation angles.

    Args:
        u (np.ndarray): Unitary matrix to be decomposed.
        wires (qml.wires.WireLike): Wires on which the operators in the decomposition should act.
        zeroed_wires (qml.wires.WiresLike): Wires that are guaranteed to be in the
            state :math:`|0\rangle`. By default, no wires come with this assumption/guarantee.
            Must be contained in ``wires``.

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

    u, wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)

    num_wires = len(wires)
    assert len(u) == 2**num_wires
    if validation_enabled():
        assert is_unitary(u)

    if num_wires == 1:
        return _rot_opt_qsd_one_qubit(u, wires[0], zeroed=bool(zeroed_wires))

    if num_wires == 2 and len(zeroed_wires) == 0:
        return _rot_opt_qsd_two_qubits(u, wires)

    p = len(u) // 2
    (k00, k01), mplx_angles_ry, (k10, k11) = _cossin_decomposition(u, p)

    with qml.QueuingManager.stop_recording():
        # Diag-decompose k00 and k01
        diag_k00, other_ops_00 = flag_decomp(k00, wires[1:])
        diag_k01, other_ops_01 = flag_decomp(k01, wires[1:])

        sub_diag, mplx_angles_rz = balance_diagonal(diag_k00.data[0], diag_k01.data[0])
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

        u_rec = ops_to_mat(new_ops, wires).reshape((2,) * (2 * num_wires))
        u = u.reshape((2,) * (2 * num_wires))
        for _ in zeroed_wires:
            u_rec = np.take(u_rec, 0, num_wires)
            u = np.take(u, 0, num_wires)
        assert np.allclose(u, u_rec, atol=1e-7), f"Mismatch:\n{u=}\n{u_rec=}"

    if qml.QueuingManager.recording():
        for op in new_ops:
            qml.apply(op)

    return new_ops
