import numpy as np
import pennylane as qml
from pennylane.ops.op_math.decompositions import one_qubit_decomposition
from .validation import validation_enabled, is_unitary, is_block_diagonal
from .asym_decomp import asymmetric_two_qubit_decomp
from .utils import ops_to_mat, aiii_kak

# static matrix
_CNOT = qml.CNOT([0, 1]).matrix()

def _diag_decomp_two_qubits(u, wires):
    """Compute the decomposition of a two-qubit unitary into a diagonal and a remaining decomposition as
    reported in Theorem 14 of the QSD paper (Shende et al. https://arxiv.org/pdf/quant-ph/0406176)."""
    if validation_enabled():
        assert u.shape == (4, 4)
        assert is_unitary(u)

    u_mod = u @ _CNOT # Optimize cnot multiplication away
    rz, _cnot, c_op, d_op, *rest_ops, a_op, b_op, gphase = asymmetric_two_qubit_decomp(u_mod, wires)

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
        #print(f"{u=}")
        #print(f"{u_rec=}")
        #print(u-u_rec)
        assert np.allclose(u, u_rec, atol=1e-7)
        assert isinstance(_cnot, qml.CNOT)
    return diag_op, other_ops


def _split_diag(D):
    angles = qml.math.angle(D)
    n = len(D) // 2
    diff = angles[..., n:] - angles[..., :n]
    mean = (angles[..., :n] + angles[..., n:]) / 2
    return np.exp(1j * mean), diff


def attach_multiplexer_node(ops0, ops1, multiplexer_wire):
    if validation_enabled():
        assert all(isinstance(op0, type(op1)) for op0, op1 in zip(ops0, ops1, strict=True))
        assert all(op0.wires==op1.wires for op0, op1 in zip(ops0, ops1))

    new_ops = []
    for op0, op1 in zip(ops0, ops1):
        if isinstance(op0, (qml.RX, qml.RY, qml.RZ)):
            new_ops.append(qml.SelectPauliRot(
                angles=[op0.data[0], op1.data[0]],
                control_wires=[multiplexer_wire],
                target_wire=op0.wires[0],
                rot_axis=op0.name[-1],
            ))
        elif isinstance(op0, (qml.CNOT, qml.CZ)):
            new_ops.append(op0)
            if qml.queuing.QueuingManager.recording():
                qml.apply(op0)
        elif isinstance(op0, qml.SelectPauliRot):
            new_ops.append(qml.SelectPauliRot(
                angles=np.concatenate([op0.data[0], op1.data[0]]),
                control_wires=[multiplexer_wire]+list(op0.hyperparameters["control_wires"]),
                target_wire=op0.hyperparameters["target_wire"],
                rot_axis=op0.hyperparameters["rot_axis"],
            ))
        else:
            raise NotImplementedError(f"attaching multiplexer node to op of type {type(op0)} ({op0}).")
    return new_ops

def diag_decomp(u, wires):
    """Compute the decomposition of ``u`` up to a diagonal, which is returned separately.
    Uses recursion with ``_diag_decomp_two_qubits`` as base case.
    """
    N = len(u)
    if N == 4:
        return _diag_decomp_two_qubits(u, wires)
    elif N == 8:
        sub_decomp = _diag_decomp_two_qubits
    else:
        sub_decomp = diag_decomp

    p = q = N//2
    K1, A, K2 = aiii_kak(u, p, q, validate=validation_enabled())
    if validation_enabled():
        assert is_unitary(K1) and is_block_diagonal(K1, p)
        assert is_unitary(K2) and is_block_diagonal(K2, p)

    with qml.queuing.QueuingManager.stop_recording():
        K1_0_diag_op, K1_0_ops = sub_decomp(K1[:p, :p], wires=wires[1:])
        K1_1_diag_op, K1_1_ops = sub_decomp(K1[p:, p:], wires=wires[1:])
        smaller_diag, multiplexer_angles_K1 = _split_diag(
            np.concatenate([K1_0_diag_op.data[0], K1_1_diag_op.data[0]])
        )
        K2_0_diag_op, K2_0_ops = sub_decomp(np.diag(smaller_diag) @ K2[:p, :p], wires=wires[1:])
        K2_1_diag_op, K2_1_ops = sub_decomp(np.diag(smaller_diag) @ K2[p:, p:], wires=wires[1:])
        multiplexer_angles_A = -2 * np.arctan2(np.diag(A, k=p), np.diag(A)[:p])

    diagonal = np.concatenate([K2_0_diag_op.data[0], K2_1_diag_op.data[0]])
    diag_op = qml.DiagonalQubitUnitary(diagonal, wires=wires)
    other_ops = [
        *attach_multiplexer_node(K2_0_ops, K2_1_ops, multiplexer_wire=wires[0]),
        qml.SelectPauliRot(multiplexer_angles_A, wires[1:], target_wire=wires[0], rot_axis="Y"),
        qml.SelectPauliRot(multiplexer_angles_K1, wires[1:], target_wire=wires[0], rot_axis="Z"),
        *attach_multiplexer_node(K1_0_ops, K1_1_ops, multiplexer_wire=wires[0]),
    ]
    if validation_enabled():
        u_rec = ops_to_mat([diag_op] + other_ops, wires)
        assert np.allclose(u, u_rec, atol=1e-7), f"Maximal difference (abs): {np.max(np.abs(u-u_rec))}"
    return diag_op, other_ops

# def parameter_optimal_qsd(u, wires, validate=True):

def split_diagonal(diag):
    """Split a diagonal into a diagonal on one qubit less and the angles for an RZ multiplexer.
    Adapted from ``qml.DiagonalQubitUnitary.compute_decomposition`` to split off the first qubit
    instead.
    """
    angles = qml.math.angle(diag)
    split = len(diag) // 2
    diff = angles[..., split:] - angles[..., :split]
    mean = (angles[..., :split] + angles[..., split:]) / 2
    return [np.exp(1j * mean), diff]
