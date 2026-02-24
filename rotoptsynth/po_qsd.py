import numpy as np
import pennylane as qml

from .recursive_flag_decomp import recursive_flag_decomp
from .linalg import csd, de_mux, re_and_de_mux, mottonen

def two_qubit_unitary(V, wires):
    tape = qml.tape.QuantumScript([qml.QubitUnitary(V, wires)])
    (tape,), _ = qml.decompose(tape, gate_set={"RY", "RZ", "CNOT", "GlobalPhase"})
    (tape,), _ = qml.transforms.combine_global_phases(tape)
    return tape.operations

def po_qsd(V: np.ndarray, wires: list) -> list:
    """
    Implements the Parameter-Optimal Quantum Shannon Decomposition (PO-QSD).
    Assumes wires[0] is the target/multiplexed qubit (q_0) and wires[1:] are the controls (q_[1:]).
    """
    n = len(wires)
    assert V.shape == (2**n, 2**n)

    # Base Case
    if n == 2:
        return two_qubit_unitary(V, wires)

    with qml.QueuingManager.stop_recording():
        controls = wires[1:]
        target = wires[0]

        # 1. Cosine-Sine Decomposition
        K00, K01, theta_y, K10, K11 = csd(V)
        M10, theta_Z_1, M11 = de_mux(K10, K11)
        M00, theta_Z_0, M01 = de_mux(K00, K01)

        # Create re-multiplexed matrix into which we absorb CY gates. Note that we change
        # the basis on the target qubit from Y to Z implicitly. We don't need to compute this
        # basis change, we just use `CZ` instead of `CY` and pass `theta_y` to the _Z_ angles of
        # MultiplexedFlag. The change back is then by passing the newly computed angles
        # theta_y_new to `mottonen` with `axis="Y"`.
        M01_new, theta_y_new, M10_new = re_and_de_mux(M01, M10, theta_y, wires, side="both")

        F_11, Delta_11 = recursive_flag_decomp(M11, controls, n_b=2, selective_demux=True)
        F_10, Delta_10 = recursive_flag_decomp(M10_new * Delta_11, controls, n_b=2, selective_demux=True)
        F_01, Delta_01 = recursive_flag_decomp(M01_new * Delta_10, controls, n_b=2, selective_demux=True)

        # Recurse PO-QSD on the final modified M00 block
        C00 = po_qsd(M00 * Delta_01, controls)

        # First Mottonen decomposition (Z-rotations)
        C_Z_1 = mottonen(theta_Z_1, controls, target, axis='Z', symmetrized="right")
        # 6. Second Mottonen decomposition (Y-rotations)
        C_Y = mottonen(theta_y_new, controls, target, axis='Y')
        C_Z_0 = mottonen(theta_Z_0, controls, target, axis='Z', symmetrized="left")
        circuit = F_11 + C_Z_1 + F_10 + C_Y + F_01 + C_Z_0 + C00

    if qml.QueuingManager.recording():
        for op in circuit:
            qml.apply(op)
    # Combine all operations [cite: 230]
    return circuit
