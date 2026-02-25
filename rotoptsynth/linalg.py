import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import _uniform_rotation_dagger_ops
from scipy.linalg import cossin


def csd(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Perform the Cosine-Sine Decomposition (CSD) of a unitary matrix :math:`V`.

    Decomposes a :math:`2^n \times 2^n` unitary into four :math:`2^{n-1} \times 2^{n-1}`
    unitary blocks and a multiplexed :math:`R_Y` rotation as used in the
    Quantum Shannon Decomposition. Uses the standard implementation in SciPy.

    Args:
        V (np.ndarray): A :math:`2^n \times 2^n` complex unitary matrix.

    Returns:
        tuple[np.ndarray]: (K00, K01, theta_Y, K10, K11) where:
            - K00, K01 (np.ndarray): :math:`2^{n-1} \times 2^{n-1}` blocks of the left factor.
            - theta_Y (np.ndarray): Array of :math:`2^{n-1}` angles for a multiplexed :math:`R_Y`
            - K10, K11 (np.ndarray): :math:`2^{n-1} \times 2^{n-1}` blocks of the right factor.
    """
    half_dim = V.shape[0] // 2

    # scipy.linalg.cossin computes V = U * CS * Vh. Since we are controlling on a single qubit,
    # we split the matrix exactly in half with p=half_dim. We use the same subgroup on both
    # sides of the KAK decomposition, so p=q in scipy convention.
    (K00, K01), theta, (K10, K11) = cossin(V, p=half_dim, q=half_dim, separate=True)

    # RY(alpha) is defined with half-angles: cos(alpha/2), sin(alpha/2).
    # SciPy returns theta for cos(theta), sin(theta), so we multiply by 2.
    theta *= 2.0

    return K00, K01, theta, K10, K11


def de_mux(K0: np.ndarray, K1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    De-multiplexes a singly-multiplexed unitary, given in terms of the two unitary blocks
    :math:`K_0` and :math:`K_1`, into a sequence of a unitary :math:`1_2\otimes M_0`, a
    multiplexed :math:`R_Z` rotation, and another unitary :math:`1_2\otimes M_1`.

    Args:
        K0 (np.ndarray): Unitary block for the :math:`|0\rangle` control state.
        K1 (np.ndarray): Unitary block for the :math:`|1\rangle` control state. Must have the
            same shape as ``K0``.

    Returns:
        tuple[np.ndarray]: (M0, theta_Z, M1) where:
            - M0 is a unitary matrix of the same shape as ``K0`` and ``K1``
            - theta_Z is an array of Z-rotation angles.
            - M1 is a unitary matrix of the same shape as ``K0`` and ``K1``

    This de-multiplexing technique is described, among many places, in Sec. 5.2 of
    Shende et al. (2004) `arXiv:quant-ph/0406176 <https://arxiv.org/abs/quant-ph/0406176>`__.
    """
    # 1. Form the product K0 * K1^\dagger
    # We perform an eigenvalue decomposition of K0 K1^\dagger as M0 D^2 M0^\dagger
    X = K0 @ K1.conj().T

    # 2. Eigendecomposition of X
    evals, M0 = np.linalg.eig(X)

    # 3. Extract D, which is the square root of the diagonal eigenvalue matrix D^2
    # Since X is unitary, its eigenvalues are phases on the unit circle.
    D = np.exp(0.5j * np.angle(evals))

    # 4. Calculate M1
    # We set M1 = D M0^\dagger K1
    M1 = (D[:, np.newaxis] * M0.conj().T) @ K1

    # Extract the Z-rotation angles
    # The Z-rotation angles are given by the phases of the diagonal, theta_Z = -2 arg(D)
    theta_Z = -2.0 * np.angle(D)

    return M0, theta_Z, M1


def re_and_de_mux(A: np.ndarray, B: np.ndarray, angles: np.ndarray, wires: list, side: str):
    """
    Re-multiplexes, then de-multiplexes unitaries absorbing one or two ``qml.CZ`` gates.

    This function is used to absorb entangler gates that originate from Möttönen decompositions
    of multiplexed single-qubit rotations, effectively symmetrizing the Möttönen decomposition.
    See Sec. IV A of Kottmann et al. (`arxiv:unknown.id <arxiv.org/abs/unknown.id>`__) for details.

    Args:
        A (np.ndarray): First unitary matrix for re-multiplexing.
        B (np.ndarray): Second unitary matrix for re-multiplexing.
        angles (np.ndarray): Rotation angles for the central multiplexer. It is always assumed
            to be an ``RZ`` multiplexer. For ``RY`` multiplexers, the basis change must be
            compensated for in the function using ``re_and_de_mux``.
        wires (list): wires that all operations act on. The first is the target of the
            multiplexed rotation, the others are the controls, which are the qubits that ``A``
            and ``B`` act on as well.
        side (str): Direction to apply CZ gate(s) from Must be one of
            ``("left", "right", "both")``.

    Returns:
        tuple: Results of the de-multiplexing step, see ``de_mux``.
    """
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
    r"""
    Split a multiplexed diagonal operator into angles for a multiplexed :math:`R_Z` rotation
    and a reduced diagonal. This performs the diagonal balancing required to maintain
    parameter-optimal counts in flag circuit synthesis during decompositions of multiplexed flags.

    Args:
        Delta (np.ndarray): 1D array of :math:`2^k` diagonal phases.
        control_wire (int): Wire index used to split the diagonal. Must be smaller or equal to
            :math:`k` for an input ``Delta`` of length :math:`2^k`.

    Returns:
        tuple: (theta_Z, Delta_prime) containing the :math:`R_Z` angles and the new diagonal.
    """
    N = len(Delta)
    num_wires = int(np.log2(N))
    assert N == 2**num_wires
    if control_wire < 0:
        control_wire = num_wires + control_wire
    assert control_wire < num_wires
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
    Implements the standard (or symmetrized) Möttönen decomposition for a multiplexed rotation.
    Decomposes a k-controlled rotation into exactly :math:`2^k` single-qubit rotations
    interleaved with :math:`2^k` CNOT gates using a Gray code sequence.
    The symmetrization is discussed in Sec. IV A and App. D2 of
    Kottmann et al. (`arxiv:unknown.id <arxiv.org/abs/unknown.id>`__).

    Args:
        theta (np.ndarray): Array of :math:`2^k` rotation angles.
        controls (list): List of control qubit indices.
        target (int): The target qubit index.
        axis (str): Rotation axis, either ``"Y"`` or ``"Z"``.
        symmetrized (str, optional): Whether the symmetrized decomposition should be used. Note
            that if ``symmetrized`` is not ``None``, the decomposition will not be exactly that of
            a multiplexed rotation, but it will be missing an entangling gate that is assumed to
            be compensated for elsewhere.

    Returns:
        list: PennyLane operations defining the decomposed multiplexed rotation.
    """
    assert axis in "YZ"
    gate = qml.RZ if axis == "Z" else qml.RY
    with qml.QueuingManager.stop_recording():
        # This function uses the reversed bit ordering
        ops = _uniform_rotation_dagger_ops(gate, theta, controls[::-1], target)

        if symmetrized is not None:
            assert symmetrized in ("right", "left")
            mid_idx = len(ops) // 2 - 1
            # Perform some safety validation of the obtained Mottonen decomp
            assert isinstance(ops[mid_idx], qml.CNOT)
            assert isinstance(ops[-1], qml.CNOT)
            assert ops[mid_idx].wires == ops[-1].wires

            ops = ops[:-1]
            if symmetrized == "left":
                # If we want to pull the entangler to the left, we need to reverse the
                # decomposition. Even though the gates do not commute, reversing maintains the
                # correctness of the raw decomposition
                ops = ops[::-1]

            # Arrange basis change to make the pulled off gate a CZ, by absorbing a pi/2 angle into
            # the outer-most rotations. See App. D2 for details.
            sign = 1 if axis == "Z" else -1

            # Implement
            first_rot = ops[0]
            ops[0] = gate(first_rot.data[0] - sign * np.pi / 2, first_rot.wires)
            last_rot = ops[-1]
            ops[-1] = gate(last_rot.data[0] + sign * np.pi / 2, last_rot.wires)

    return ops


def expand_diagonal_matrix(diag: np.ndarray, wires: list, wire_order: list) -> np.ndarray:
    """Expand and/or permute a one-dimensional array as if it was a diagonal matrix acting
    on ``wires`` that needs to be adapted to a new ``wire_order``. This is the analogue
    of ``qml.math.expand_matrix`` for diagonal matrices.

    Args:
        diag (np.ndarray): One-dimensional array capturing the diagonal.
        wires (list): Wires that the array is acting on non-trivially.
        wire_order (list): Wires that the array should be expanded to/permuted for.

    Returns:
        np.ndarray: Expanded and/or permuted array.
    """
    return np.diag(qml.math.expand_matrix(np.diag(diag), wires, wire_order=wire_order))
