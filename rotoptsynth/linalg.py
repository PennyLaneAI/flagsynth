import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import _uniform_rotation_dagger_ops
from scipy.linalg import cossin


def csd(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the Cosine-Sine Decomposition (CSD) of a unitary matrix V.

    Args:
        V (np.ndarray): A 2^n x 2^n complex unitary matrix.

    Returns:
        tuple: (K00, K01, theta_Y, K10, K11) where:
            - K00, K01 are the 2^{n-1} x 2^{n-1} blocks of the left unitary multiplier.
            - theta_Y is the array of 2^{n-1} angles for the multiplexed RY rotation.
            - K10, K11 are the 2^{n-1} x 2^{n-1} blocks of the right unitary multiplier.
    """
    half_dim = V.shape[0] // 2

    # scipy.linalg.cossin computes V = U * CS * Vh. Since we are controlling on a single qubit,
    # we split the matrix exactly in half with p=half_dim. We use the same subgroup on both
    # sides of the KAK decomposition, so p=q in scipy convention.
    (K00, K01), theta, (K10, K11) = cossin(V, p=half_dim, q=half_dim, separate=True)

    # RY(alpha) is defined with half-angles: cos(alpha/2), sin(alpha/2).
    # SciPy returns theta for cos(theta), sin(theta), so we multiply by 2.
    theta_Y = 2.0 * theta

    return K00, K01, theta_Y, K10, K11


def de_mux(K0: np.ndarray, K1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    De-multiplexes two unitary blocks K0 and K1 into a sequence of M0, a multiplexed
    Z-rotation, and M1.

    Args:
        K0 (np.ndarray): The unitary block corresponding to the 0-control state.
        K1 (np.ndarray): The unitary block corresponding to the 1-control state.

    Returns:
        tuple: (M0, theta_Z, M1) where:
            - M0 is a unitary matrix.
            - theta_Z is an array of Z-rotation angles.
            - M1 is a unitary matrix.
    """
    # 1. Form the product K0 * K1^\dagger
    # We perform an eigenvalue decomposition of K0 K1^\dagger as M0 D^2 M0^\dagger
    X = K0 @ K1.conj().T

    # 2. Eigendecomposition of X
    evals, M0 = np.linalg.eig(X)

    # 3. Extract D, which is the square root of the diagonal eigenvalue matrix D^2
    # Since X is unitary, its eigenvalues are phases on the unit circle.
    # We use np.angle to safely extract the phases and halve them.
    phases = np.angle(evals)
    D_diag = np.exp(0.5j * phases)

    # 4. Calculate M1
    # We set M1 = D M0^\dagger K1
    # We can broadcast the diagonal multiplication for efficiency.
    M1 = (D_diag[:, np.newaxis] * M0.conj().T) @ K1

    # 5. Extract the Z-rotation angles
    # The Z-rotation angles are given by the phases of the diagonal, theta_Z = -2 arg(D)
    theta_Z = -2.0 * np.angle(D_diag)

    return M0, theta_Z, M1


def re_and_de_mux(A, B, angles, wires, side: str = None):
    target, *controls = wires
    _cz = qml.CZ([controls[0], target]).matrix(wire_order=wires)
    re_muxed = (
        np.kron(np.eye(2), A)
        @ qml.matrix(mottonen(angles, controls, target, axis="Z"), wire_order=wires)
        @ np.kron(np.eye(2), B)
    )
    if side in ("left", "both"):
        re_muxed = re_muxed @ _cz
    if side in ("right", "both"):
        re_muxed = _cz @ re_muxed
    _N = 2 ** len(controls)
    return de_mux(re_muxed[:_N, :_N], re_muxed[_N:, _N:])


def balance_diagonal(Delta: np.ndarray, control_wire: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a multiplexed diagonal operator into a Z-rotation angle array and a new
    reduced diagonal operator.

    Args:
        Delta (np.ndarray): 1D array of length 2^k representing the diagonal phases.
        control_wire (int): The wire being used to split the branches.

    Returns:
        tuple: (theta_Z, Delta_prime) where:
            - theta_Z is a 1D array of angles for the multiplexed RZ rotation.
            - Delta_prime is a 1D array representing the newly balanced diagonal.
    """
    N = len(Delta)
    num_wires = int(np.log2(N))
    assert N == 2**num_wires
    if control_wire < 0:
        control_wire = num_wires + control_wire
    assert control_wire < N
    # Split the diagonal in half.
    Delta = Delta.reshape((2,) * num_wires)
    D0 = np.take(Delta, 0, axis=control_wire).reshape((-1,))
    D1 = np.take(Delta, 1, axis=control_wire).reshape((-1,))

    # Extract the phases
    angle0 = np.angle(D0)
    angle1 = np.angle(D1)

    # Calculate theta_Z = arg(Delta1) - arg(Delta0)
    theta_Z = angle1 - angle0

    # Calculate the balanced diagonal Delta' = exp(i/2 * (arg(Delta0) + arg(Delta1)))
    Delta_prime = np.exp(0.5j * (angle0 + angle1))

    return theta_Z, Delta_prime


def mottonen(
    theta: np.ndarray, controls: list, target: int, axis: str = "Y", symmetrized: str | None = None
) -> list:
    """
    Implements the standard Möttönen decomposition for a multiplexed rotation.
    Decomposes a k-controlled rotation into exactly 2^k single-qubit rotations
    interleaved with 2^k CNOT gates using a Gray code sequence.

    Args:
        theta (np.ndarray): Array of 2^k angles corresponding to each control state.
        controls (list): The list of control wires.
        target (int): The target wire.
        axis (str): The rotation axis, either 'Y' or 'Z'.

    Returns:
        list: PennyLane operations defining the decomposed multiplexed rotation.
    """
    assert axis in "YZ"
    gate = qml.RZ if axis == "Z" else qml.RY
    # This function uses the other bit ordering
    with qml.QueuingManager.stop_recording():
        ops = _uniform_rotation_dagger_ops(gate, theta, controls[::-1], target)
        if symmetrized is not None:
            assert symmetrized in ("right", "left")
            mid_idx = len(ops) // 2 - 1
            assert isinstance(ops[mid_idx], qml.CNOT)
            assert isinstance(ops[-1], qml.CNOT)
            assert ops[mid_idx].wires == ops[-1].wires
            ops = ops[:-1]
            if symmetrized == "left":
                ops = ops[::-1]
            sign = 1 if axis == "Z" else -1

            first_rot = ops[0]
            ops[0] = gate(first_rot.data[0] - sign * np.pi / 2, first_rot.wires)
            last_rot = ops[-1]
            ops[-1] = gate(last_rot.data[0] + sign * np.pi / 2, last_rot.wires)

    return ops


def merge_diagonals(diag_0, wires_0, diag_1, wires_1):
    all_wires = sorted(set(wires_0 + wires_1))
    if wires_0:
        diag_0 = np.diag(qml.math.expand_matrix(np.diag(diag_0), wires_0, wire_order=all_wires))
    if wires_1:
        diag_1 = np.diag(qml.math.expand_matrix(np.diag(diag_1), wires_1, wire_order=all_wires))
    diag = diag_0 * diag_1
    return diag, all_wires
