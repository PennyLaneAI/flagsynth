"""This module implements the main function for the parameter-optimal Quantum
Shannon Decomposition."""

import numpy as np
import pennylane as qml
from pennylane.decomposition import register_resources, add_decomps, resource_rep

from .linalg import csd, de_mux, mottonen, re_and_de_mux
from .recursive_flag_decomp import recursive_flag_decomp_cliff_rz


@register_resources(lambda num_wires: {qml.RZ: 2, qml.RY:1} if num_wires == 1 else {qml.RZ: 7, qml.RY: 8, qml.CNOT: 3})
def _fixed_qubit_unitary_decomp(matrix, wires):
    if len(matrix) == 4:
        return qml.ops.two_qubit_decomposition(matrix, wires)
    assert len(matrix) == 2
    return qml.ops.op_math.decompositions.unitary_decompositions.zyz_decomp_rule(U, wires)


def _two_qubit_unitary(matrix, wires):
    """Decompose a two-qubit unitary matrix into a circuit of RZ, RY, CNOT and
    GlobalPhase gates."""
    if qml.decomposition.enabled_graph():
        kwargs = {"fixed_decomps": {qml.QubitUnitary: _fixed_qubit_unitary_decomp}}
    else:
        kwargs = {}
    tape = qml.tape.QuantumScript([qml.QubitUnitary(matrix, wires)])
    (tape,), _ = qml.decompose(tape, gate_set={"RY", "RZ", "CNOT", "GlobalPhase"}, **kwargs)
    (tape,), _ = qml.transforms.combine_global_phases(tape)
    return tape.operations


def po_qsd(matrix: np.ndarray, wires: list) -> list:
    r"""Implements the Parameter-Optimal Quantum Shannon Decomposition (PO-QSD).

    Args:
        matrix (np.ndarray): Matrix to implement
        wires (list): Wires on which the matrix should act.

    Returns:
        list: Circuit description in terms of ``qml.RZ``, ``qml.RY``, ``qml.CZ`` and ``qml.CNOT``
        gates.

    Note that this function queues the decomposition it computes to the queuing system in
    PennyLane.

    The PO-QSD decomposes an :math:`n`-qubit matrix of size :math:`2^n\times 2^n` into
    :math:`4^n-1` rotation gates and :math:`\frac12 4^n - \frac38 (n + 2) 2^n + n - 1`
    entanglers (``qml.CZ`` and ``qml.CNOT``).

    """
    n = len(wires)
    assert matrix.shape == (2**n, 2**n)

    # Base Case
    if n == 2:
        return _two_qubit_unitary(matrix, wires)

    with qml.QueuingManager.stop_recording():
        controls = wires[1:]
        target = wires[0]

        # 1. Cosine-Sine Decomposition
        K00, K01, theta_y, K10, K11 = csd(matrix)
        M10, theta_Z_1, M11 = de_mux(K10, K11)
        M00, theta_Z_0, M01 = de_mux(K00, K01)

        # Create re-multiplexed matrix into which we absorb CY gates. Note that we change
        # the basis on the target qubit from Y to Z implicitly. We don't need to compute this
        # basis change, we just use `CZ` instead of `CY` and pass `theta_y` to the _Z_ angles of
        # MultiplexedFlag. The change back is then by passing the newly computed angles
        # theta_y_new to `mottonen` with `axis="Y"`.
        M01_new, theta_y_new, M10_new = re_and_de_mux(M01, M10, theta_y, wires, side="both")

        rec_kw = {"n_b": 2, "selective_demux": True}

        ops_11, Delta_11 = recursive_flag_decomp_cliff_rz(M11, controls, **rec_kw)
        ops_10, Delta_10 = recursive_flag_decomp_cliff_rz(M10_new * Delta_11, controls, **rec_kw)
        ops_01, Delta_01 = recursive_flag_decomp_cliff_rz(M01_new * Delta_10, controls, **rec_kw)

        # Recurse PO-QSD on the final modified M00 block
        ops_00 = po_qsd(M00 * Delta_01, controls)

        mux_z_rot_1 = mottonen(theta_Z_1, controls, target, axis="Z", symmetrized="right")
        mux_y_rot = mottonen(theta_y_new, controls, target, axis="Y")
        mux_z_rot_0 = mottonen(theta_Z_0, controls, target, axis="Z", symmetrized="left")
    circuit = ops_11 + mux_z_rot_1 + ops_10 + mux_y_rot + ops_01 + mux_z_rot_0 + ops_00

    if qml.QueuingManager.recording():
        for op in circuit:
            qml.apply(op)
    return circuit

def _po_qsd_resources(num_wires):
    n = num_wires
    exp_num_cnots = 4**n // 2 - 3 * (n + 2) * 2**n // 8 + n - 1
    return {
        resource_rep(qml.RZ): 4**n//2 - 1,
        resource_rep(qml.RY): 4**n//2,
        resource_rep(qml.CNOT): exp_num_cnots - 1,
        resource_rep(qml.CZ): 1,
    }

po_qsd_rule = register_resources(_po_qsd_resources, po_qsd, exact=False)
add_decomps(qml.QubitUnitary, po_qsd_rule)
