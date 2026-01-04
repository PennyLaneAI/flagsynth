"""This module contains the main function to perform rotation-angle-optimal unitary synthesis,
called ``rot_opt_qsd`` (at the moment)."""

# pylint: disable=too-many-locals
from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy.linalg import cossin, det
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
        rec_a = ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(d_sub) @ v_sub
        assert np.allclose(rec_a, a, atol=1e-7), f"{np.max(np.abs(rec_a-a))=}"
        assert np.allclose(
            ops_to_mat(other_ops_u_sub, wires[1:]) @ np.diag(np.conj(d_sub)) @ v_sub, b, atol=1e-7
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

def bdi_kak(mat):
    """BDI(p, q) Cartan decomposition of special orthogonal o

    Args:
        o (np.ndarray): The special orthogonal matrix to decompose. It must be square-shaped with
            size p+q.
        p (int): First subspace size for SO(p) x SO(q), the vertical subspace
        q (int): Second subspace size for SO(p) x SO(q), the vertical subspace

    Returns:
        np.ndarray: The first K from the KAK decomposition
        np.ndarray: The exponentiated Cartan subalgebra element A from the KAK decomposition
        np.ndarray: The second K from the KAK decomposition


    The input ``o`` and all three output matrices are group elements, not algebra elements.
    """
    p = len(mat) // 2
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    (k00, k01), theta, (k10, k11) = cossin(mat, p=p, q=p, separate=True)

    # banish negative determinants
    d00, d01 = det(k00), det(k01)
    d10, d11 = det(k10), det(k11)
    assert np.isclose(d00 * d01 * d10 * d11, 1.0), f"{d00 * d01 * d10 * d11}"

    k00[:, 0] *= d00
    k01[:, 0] *= d01
    k10[0] *= d10
    k11[0] *= d11

    theta[0] *= d00 * d01
    if d00 * d10 < 0:
        theta[0] += np.pi

    if validation_enabled():
        for k in [k00, k01, k10, k11]:
            assert np.allclose(k @ k.T, np.eye(p)) and np.isclose(np.linalg.det(k), 1.0)
            assert np.allclose(k.imag, 0.)
        K0 = np.block([[k00, 0*k00], [0*k01, k01]])
        K1 = np.block([[k10, 0*k10], [0*k11, k11]])
        C, S = np.diag(np.cos(theta)), np.diag(np.sin(theta))
        A = np.block([[C, -S], [S, C]])
        assert np.allclose(K0 @ A @ K1, mat), f"BDI Cartan decomposition failed: KAK={K0@A@K1}\nMAT={mat}"

    return (k00, k01), theta, (k10, k11)


def iterative_cossin_decomposition(u, level=0, num_wires=None):
    if level == 0:
        num_wires = int(np.log2(len(u[0][0][0])))
    elif level == num_wires-1:
        new_u = []
        # entries are of the form (data, mux_wires) where data is a list of matrices or a single 1-d
        # array of angles.
        for data, mux_wires in u:
            if isinstance(data, np.ndarray) and np.ndim(data) == 1:
                new_u.append((data, mux_wires))
                continue
            assert len(data) == 2**(num_wires - 1)
            assert all(np.shape(mat)==(2, 2) for mat in data)

            angles = 2 * np.concatenate([bdi_kak(mat)[1] for mat in data])
            new_u.append((angles, mux_wires))
        return new_u

    new_u = []
    # entries are of the form (data, mux_wires) where data is a list of matrices or a single 1-d
    # array of angles.
    for data, mux_wires in u:
        if isinstance(data, np.ndarray) and np.ndim(data) == 1:
            new_u.append((data, mux_wires))
            continue
        before, angles, after = [], [], []
        for mat in data:
            post_blocks, angles_ry, pre_blocks = bdi_kak(mat)
            if validation_enabled():
                post0, post1 = post_blocks
                pre0, pre1 = pre_blocks
                A = np.block([[np.diag(np.cos(angles_ry)), np.diag(-np.sin(angles_ry))],[np.diag(np.sin(angles_ry)), np.diag(np.cos(angles_ry))]])
                assert np.allclose(
                    np.block([[post0, 0*post0],[0*post0, post1]]) @
                    A @
                    np.block([[pre0, 0*pre0],[0*pre0, pre1]]),
                    mat
                )
                with qml.QueuingManager.stop_recording():
                    op_mat = qml.matrix(qml.tape.QuantumScript([
                        qml.SelectPauliRot(2 * angles_ry, range(level+1, num_wires), level, rot_axis="Y")
                    ]), wire_order=range(level, num_wires))
                    assert np.allclose(A, op_mat)
            before.extend(pre_blocks)
            angles.append(2 * angles_ry)
            after.extend(post_blocks)
        new_u.append((before, mux_wires + [level]))
        new_u.append((np.concatenate(angles), mux_wires + list(range(level+1, num_wires))))
        new_u.append((after, mux_wires + [level]))
    return iterative_cossin_decomposition(new_u, level=level+1, num_wires=num_wires)


def filter_by_zeroed_qubit(angles, mux_wires, target, zeroed_qubits):
    for w in zeroed_qubits:
        if w == target:
            assert w == len(zeroed_qubits)-1, f"{w=}, {len(zeroed_qubits)-1=}, {zeroed_qubits=}"
            zeroed_qubits = zeroed_qubits[:-1]
            break
        angles = angles[:len(angles)//2]
        mux_wires = mux_wires[1:]
    return angles, mux_wires, zeroed_qubits



def real_rot_opt_qsd(u: np.ndarray, wires: WiresLike, zeroed_wires: Optional[WiresLike]=None):

    u[:, -1] *= np.linalg.det(u)
    assert np.linalg.det(u) > 0 # # todo: Only SO is supported at the moment

    if zeroed_wires is None:
        zeroed_wires = []
    elif not isinstance(zeroed_wires, list):
        zeroed_wires = list(zeroed_wires)

    u, wires = _validate_and_arrange_zeroed_wires(u, wires, zeroed_wires)

    num_wires = len(wires)
    assert len(u) == 2**num_wires
    if validation_enabled():
        assert u.dtype == np.float64
        assert is_unitary(u)

    circuit_data = iterative_cossin_decomposition([([u],[])])
    ops = []
    zeroed_qubits = range(len(zeroed_wires))
    for angles, mux_wires in circuit_data:
        target = next(iter(w for w in range(len(wires)) if w not in mux_wires))
        angles, mux_wires, zeroed_qubits = filter_by_zeroed_qubit(angles, mux_wires, target, zeroed_qubits)
        target = wires[target]
        control = [wires[w] for w in mux_wires]
        if len(angles) == 1:
            assert not mux_wires
            ops.append(qml.RY(angles[0], wires=target))
        else:
            ops.append(qml.SelectPauliRot(angles, control, target, rot_axis="Y"))

    if validation_enabled():
        mat = qml.matrix(qml.tape.QuantumScript(ops), wire_order=wires)
        assert np.allclose(mat.imag, 0.)
        mat = mat.real
        mat_sliced = mat.copy().reshape((2,) * (2*len(wires)))
        u_sliced = u.copy().reshape((2,) * (2*len(wires)))
        zeroed_ids = sorted([wires.index(w) for w in zeroed_wires], reverse=True)
        for idx in zeroed_ids:
            mat_sliced = np.take(mat_sliced, 0, len(wires) + idx)
            u_sliced = np.take(u_sliced, 0, len(wires) + idx)

        assert np.allclose(mat_sliced, u_sliced), f"{mat_sliced}\n{u_sliced}"
    return ops


