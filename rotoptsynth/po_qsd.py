"""This module implements the main function for the parameter-optimal Quantum
Shannon Decomposition."""

import numpy as np
import pennylane as qml

from .linalg import csd, de_mux, mottonen, re_and_de_mux
from .recursive_flag_decomp import recursive_flag_decomp_cliff_rz


def _two_qubit_unitary(matrix, wires):
    """Decompose a two-qubit unitary matrix into a circuit of RZ, RY, CNOT and
    GlobalPhase gates."""
    tape = qml.tape.QuantumScript([qml.QubitUnitary(matrix, wires)])
    (tape,), _ = qml.decompose(tape, gate_set={"RY", "RZ", "CNOT", "GlobalPhase"})
    (tape,), _ = qml.transforms.combine_global_phases(tape)
    return tape.operations


def po_qsd(matrix: np.ndarray, wires: list) -> list:
    """Implements the Parameter-Optimal Quantum Shannon Decomposition (PO-QSD)."""
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

        F_11, Delta_11 = recursive_flag_decomp_cliff_rz(M11, controls, **rec_kw)
        F_10, Delta_10 = recursive_flag_decomp_cliff_rz(M10_new * Delta_11, controls, **rec_kw)
        F_01, Delta_01 = recursive_flag_decomp_cliff_rz(M01_new * Delta_10, controls, **rec_kw)

        # Recurse PO-QSD on the final modified M00 block
        C00 = po_qsd(M00 * Delta_01, controls)

        mux_z_rotation_1 = mottonen(theta_Z_1, controls, target, axis="Z", symmetrized="right")
        mux_y_rotation = mottonen(theta_y_new, controls, target, axis="Y")
        mux_z_rotation_0 = mottonen(theta_Z_0, controls, target, axis="Z", symmetrized="left")
        circuit = F_11 + mux_z_rotation_1 + F_10 + mux_y_rotation + F_01 + mux_z_rotation_0 + C00

    if qml.QueuingManager.recording():
        for op in circuit:
            qml.apply(op)
    # Combine all operations [cite: 230]
    return circuit
