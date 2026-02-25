"""This module contains recursive flag decompositions, both into {Clifford+Rot} gates and into
multiplexed single-qubit flags, represented as ``MultiplexedFlag`` objects."""

from functools import partial, singledispatchmethod
from itertools import chain

import numpy as np
import pennylane as qml
from pennylane.math.decomposition import zyz_rotation_angles
from pennylane.operation import Operation

from .asymmetric_decomp import asymmetric_decomp
from .linalg import balance_diagonal, csd, de_mux, mottonen, re_and_de_mux, expand_diagonal_matrix
from .multiplexed_flag import MultiplexedFlag


@qml.QueuingManager.stop_recording()
def one_qubit_flag_decomp(matrix: np.ndarray, wires: list) -> tuple[list, np.ndarray]:
    """
    Implements the one-qubit flag decomposition returning the two-gate flag circuit
    and the trailing two-element diagonal. This is based on a standard Euler decomposition.

    Args:
        matrix (np.ndarray): Matrix of shape ``(2, 2)`` to be decomposed.
        wires (list): Wires on which the operations should act. Should have length 1.

    Returns:
        tuple[list, np.ndarray]: List of a single operation (a ``MultiplexedFlag``) and a
        one-dimensional array of length ``2`` representing the diagonal.
    """
    phi, theta, omega, alpha = zyz_rotation_angles(matrix, return_global_phase=True)
    F = [MultiplexedFlag(phi, theta, wires)]
    Delta = np.exp(1j * np.array([-omega / 2 + alpha, omega / 2 + alpha]))
    return F, Delta


_cnot = qml.CNOT([0, 1]).matrix()
_y = qml.RY(-np.pi / 2, 0).matrix()


@qml.QueuingManager.stop_recording()
def two_qubit_flag_decomp(matrix: np.ndarray, wires: list) -> tuple[list, np.ndarray]:
    """Two-qubit flag decomposition as described in Alg. 4 in App. A of
    `Kottmann et al. <arxiv.org/abs/unknown.id>`__.

    Args:
        matrix (np.ndarray): Matrix of shape ``(4, 4)`` to be decomposed.
        wires (list): Wires on which the operations should act. Should have length 2.

    Returns:
        tuple[list, np.ndarray]: List of operations (``MultiplexedFlag`` and ``qml.CZ``) and a
        one-dimensional array of length ``4`` representing the diagonal.
    """

    a, b, c, d, alpha, Psi, Theta, Phi = asymmetric_decomp(_cnot @ matrix)

    phi_a, theta_a, omega_a = zyz_rotation_angles(a)
    phi_c, theta_c, omega_c = zyz_rotation_angles(c)
    phi_d, theta_d, omega_d = zyz_rotation_angles(_y @ d)
    _ops = [qml.RZ(omega_d, wires[1]), qml.RX(-Phi, wires[1])]
    phi_e, theta_e, omega_e = zyz_rotation_angles(qml.matrix(_ops, wires[1:]))
    _ops = [qml.RX(omega_e, wires[1])]
    phi_f, theta_f, omega_f = zyz_rotation_angles(b @ qml.matrix(_ops, wires[1:]) @ _y.conj().T)

    F = [
        MultiplexedFlag(phi_c, theta_c, wires[0]),
        MultiplexedFlag(phi_d, theta_d, wires[1]),
        qml.CZ(wires),
        MultiplexedFlag(omega_c + np.pi / 2, Theta, wires[0]),
        MultiplexedFlag(phi_e, theta_e, wires[1]),
        qml.CZ(wires),
        MultiplexedFlag(phi_a - np.pi / 2, theta_a, wires[0]),
        MultiplexedFlag(phi_f, theta_f, wires[1]),
    ]
    Delta = np.diag(
        qml.matrix(
            [
                qml.RZ(omega_a, wires[0]),
                qml.RZ(omega_f, wires[1]),
                qml.IsingZZ(Psi, wires),
                qml.GlobalPhase(-alpha),
            ],
            wire_order=wires,
        )
    )

    return F, Delta


@partial(qml.matrix, wire_order=[0])
def _flag(theta_z, theta_y):
    """Single qubit flag matrix."""
    qml.RZ(theta_z, 0)
    qml.RY(theta_y, 0)


def _decompose_mux_single_qubit_flag_base_case(
    theta_z: np.ndarray, theta_y: np.ndarray, wires: list
):
    """Decomposition of a singly-multiplexed single-qubit flag circuit into
    one entangler, four single-qubit rotations, and a diagonal, as described in
    `Bergholm et al. <https://arxiv.org/abs/quant-ph/0410066>`__.

    Args:
        theta_z (np.ndarray): Rotation angles for multiplexed ``RZ`` rotation.
        theta_y (np.ndarray): Rotation angles for multiplexed ``RY`` rotation.
        wires (list): Wires on which the multiplexed flag acts.

    Returns:
        tuple[list, np.ndarray]: Decomposed circuit and remaining diagonal.

    """
    K0, K1 = list(map(_flag, theta_z, theta_y))
    X = K0 @ K1.conj().T
    phi = np.angle(np.linalg.det(X))
    arg_a = np.angle(X[0, 0] * np.exp(-1j * phi / 2))
    rho0 = (np.pi - phi) / 4 - arg_a / 2
    rho1 = (3 * np.pi - phi) / 4 + arg_a / 2
    r = np.exp(1j * np.array([rho0, rho1]))
    Y = np.diag(r) @ X * r
    _I, L0 = np.linalg.eig(Y)
    assert np.allclose(_I, np.array([1j, -1j])), "Fix ordering"
    d = np.exp(1j * np.pi / 4 * np.array([1, -1]))
    L1 = np.diag(d) @ L0.conj().T @ np.diag(r.conj()) @ K1
    D = np.concatenate([d, d.conj()])
    R = np.concatenate([r.conj(), r])

    L1 = np.diag(np.array([1, -1j])) @ L1
    D = np.exp(-1j * np.pi / 4) * np.array([1, 1j, 1j, -1]) * D
    R = np.exp(1j * np.pi / 4) * R * np.array([1, 1, -1j, -1j])

    # Gate decomposition
    phi1, theta1, omega1, alpha1 = zyz_rotation_angles(L1, return_global_phase=True)
    phi2, theta2, omega2, alpha2 = zyz_rotation_angles(L0, return_global_phase=True)
    ops = [
        qml.RZ(phi1, wires[1]),
        qml.RY(theta1, wires[1]),
        qml.CZ(wires),
        qml.RZ(omega1 + phi2, wires[1]),
        qml.RY(theta2, wires[1]),
    ]
    Delta = (
        np.diag(qml.matrix(qml.RZ, wire_order=wires)(omega2, wires[1]))
        * R
        * np.exp(1j * (alpha1 + alpha2))
    )
    return ops, Delta


@qml.QueuingManager.stop_recording()
def _decompose_mux_single_qubit_flag(op):
    """Decomposition of a multiplexed single-qubit flag circuit into
    one two-qubit entanglers, single-qubit rotations, and a diagonal, as described in
    `Bergholm et al. <https://arxiv.org/abs/quant-ph/0410066>`__.

    Args:
        op (MultiplexedFlag): Multiplexed flag operation.

    Returns:
        tuple[list, np.ndarray]: Decomposed circuit and remaining diagonal.

    This function recursively applies itself, using ``_decompose_mux_single_qubit_flag_base_case``
    as base case.
    """
    assert isinstance(op, MultiplexedFlag)
    theta_z, theta_y = op.data
    wires = op.wires
    n = len(wires)
    if n == 2:
        return _decompose_mux_single_qubit_flag_base_case(theta_z, theta_y, wires)

    N = 2 ** (n - 2)
    all_ops, all_diags = zip(
        *[
            _decompose_mux_single_qubit_flag_base_case(
                theta_z[i::N],
                theta_y[i::N],
                [wires[0], wires[-1]],
            )
            for i in range(N)
        ]
    )

    diag = np.array(all_diags).reshape((N, 2, 2))
    diag = np.moveaxis(diag, 1, 0).reshape(-1)
    l1_z = np.array([_ops[0].data[0] for _ops in all_ops])
    l1_y = np.array([_ops[1].data[0] for _ops in all_ops])
    l0_z = np.array([_ops[3].data[0] for _ops in all_ops])
    l0_y = np.array([_ops[4].data[0] for _ops in all_ops])
    L1_op = MultiplexedFlag(l1_z, l1_y, wires[1:])
    final_ops, Delta_1 = _decompose_mux_single_qubit_flag(L1_op)
    l0_z_mod, Delta_1_prime = balance_diagonal(Delta_1, -1)
    l0_z += l0_z_mod
    diag *= np.kron(np.ones(2), np.kron(Delta_1_prime, np.ones(2)))

    final_ops.append(qml.CZ([wires[0], wires[-1]]))

    L0_op = MultiplexedFlag(l0_z, l0_y, wires[1:])
    l0_ops, Delta_0 = _decompose_mux_single_qubit_flag(L0_op)
    final_ops.extend(l0_ops)
    diag *= np.kron(np.ones(2), Delta_0)

    return final_ops, diag


@qml.QueuingManager.stop_recording()
def _merge_diag_into_mux(op, main_diag, main_diag_wires):
    target = op.wires[-1]
    balance_idx = main_diag_wires.index(target)
    main_diag_wires.pop(balance_idx)
    theta, main_diag = balance_diagonal(main_diag, balance_idx)
    missing_ctrls = [w for w in op.wires[:-1] if w not in main_diag_wires]
    theta_broad = expand_diagonal_matrix(
        np.kron(np.ones(2 ** len(missing_ctrls)), theta),
        wires=missing_ctrls + main_diag_wires,
        wire_order=op.wires[:-1],
    )
    op = MultiplexedFlag(op.data[0] + theta_broad, op.data[1], op.wires)
    return op, main_diag, main_diag_wires


class CollectionOfDiagonals:
    """A collection of diagonals acting on individual sets of wires. An instance is supposed to
    be created without any data, and individual diagonals and their wires should be added
    via ``CollectionOfDiagonals.append``. The key functionality of this class is in
    ``move_through_op``, which computes the effect of moving the diagonals through a circuit
    operation on both the diagonals and the operation itself.

    This class is used in ``decompose_mux_single_qubit_flags``.
    """

    def __init__(self):
        self.diagonals = []
        self.diag_wires = []

    def append(self, diagonal, wires):
        """Add a diagonal and the wires it acts on to the collection."""
        self.diagonals.append(diagonal)
        self.diag_wires.append(wires)

    @singledispatchmethod
    def move_through_op(self, op: Operation):
        """Move all diagonals through a quantum circuit operation, modifying the diagonals and
        the wires they act on, as well as the operation itself.

        This function is dispatched based on the type of the circuit operation."""
        raise NotImplementedError(f"Moving diagonals through {op=} is not supported.")

    @move_through_op.register
    def _(self, op: MultiplexedFlag):
        """Move diagonals through a ``MultiplexedFlag``. This is described in
        App. C2 of Kottmann et al."""
        for j, (d, d_wires) in enumerate(zip(self.diagonals, self.diag_wires, strict=True)):
            if op.wires[-1] in d_wires:
                assert all(
                    w in op.wires for w in d_wires
                ), "can't merge a diagonal into a smaller MultiplexedFlag."
                op, self.diagonals[j], self.diag_wires[j] = _merge_diag_into_mux(op, d, d_wires)

        if len(op.wires) == 1:
            return [qml.RZ(op.data[0][0], op.wires), qml.RY(op.data[1][0], op.wires)]

        ops, _d = _decompose_mux_single_qubit_flag(op)
        self.append(_d, list(op.wires))
        return ops

    @move_through_op.register
    def _(self, op: qml.CZ | qml.RZ):
        """Just commute diagonals through a ``CZ`` or an ``RZ`` gate, as they commute."""
        return [op]

    @move_through_op.register
    def _(self, op: qml.CNOT):
        """Just commute diagonals through a ``CNOT`` gate if (and only if) the diagonals
        only overlap with the control qubit of the ``CNOT``, not with the target."""
        assert all(op.wires[1] not in d_wires for d_wires in self.diag_wires)
        return [op]

    @move_through_op.register
    def _(self, op: qml.RY):
        """We can commute diagonals through a ``RY`` gate if (and only if) the diagonals
        do not overlap with the qubit of the ``RY``."""
        assert all(op.wires[0] not in d_wires for d_wires in self.diag_wires)
        return [op]

    def merge_all(self, wires):
        """Merge all diagonals in the ``CollectionOfDiagonals``, respecting and combining the
        individual wires they act on.

        Args:
            wire_order (list): Wire ordering to which the merged diagonal should be adapted.
        """
        merged = np.ones(2 ** len(wires), dtype=complex)
        for d, d_wires in zip(self.diagonals, self.diag_wires, strict=True):
            merged *= expand_diagonal_matrix(d, d_wires, wires)
        return merged


@qml.QueuingManager.stop_recording()
def decompose_mux_single_qubit_flags(ops):
    """Sweep through a circuit, decomposing multiplexed single-qubit flags
    and moving remainder diagonals along.

    Args:
        ops (list[Operation]): Circuit description. Only instances of ``MultiplexedFlag`` will be
            decomposed. Other gates must be compatible with moving diagonals through them.

    Returns:
        tuple[list[Operation], np.ndarray]: New circuit description with ``MultiplexedFlag``
        instances decompose, as well as a diagonal.

    Also see ``_decompose_mux_single_qubit_flag`` for details.
    """

    diagonals = CollectionOfDiagonals()
    new_ops = list(chain.from_iterable([diagonals.move_through_op(op) for op in ops]))
    tape = qml.tape.QuantumScript(ops)
    diagonal = diagonals.merge_all(sorted(tape.wires))
    return new_ops, diagonal


@qml.QueuingManager.stop_recording()
def mux_ops(ops: list, controls: list) -> list:
    """
    Attaches :math:`k` control wires to :math:`2^k` lists of operations, exploiting the Multiplexer
    Extension Property. If a gate is static (no parameters), it is applied unconditionally
    to save control nodes.

    Args:
        ops (list[list]): Operations to apply for different control values
        controls (list): The control wires.

    Returns:
        list: A new list of multiplexed PennyLane operations.
    """
    n = len(controls)
    assert len(ops) == 2**n
    muxed_ops = []

    if n == 1:
        # Note that len(ops)==2 is guaranteed here
        for op0, op1 in zip(*ops):
            assert isinstance(op0, type(op1)) and op0.wires == op1.wires

            match op0:
                case qml.RZ():
                    theta_z = (np.array([op0.data[0], op1.data[0]]),)
                    theta_y = np.zeros(2)
                case qml.RZ():
                    theta_z = np.zeros(2)
                    theta_y = (np.array([op0.data[0], op1.data[0]]),)
                case MultiplexedFlag():
                    theta_z = np.concatenate([op0.data[0], op1.data[0]])
                    theta_y = np.concatenate([op0.data[1], op1.data[1]])
                case _ if op0.num_params == 0:
                    muxed_ops.append(op0)
                    continue
                case _:
                    raise NotImplementedError(f"Can't multiplex {op0} and {op1}.")
            muxed_ops.append(MultiplexedFlag(theta_z, theta_y, controls + op0.wires))
        return muxed_ops

    # Recursively attach multiplexing nodes
    ops0 = mux_ops(ops[: 2 ** (n - 1)], controls=controls[1:])
    ops1 = mux_ops(ops[2 ** (n - 1) :], controls=controls[1:])
    muxed_ops = mux_ops([ops0, ops1], controls=controls[:1])

    return muxed_ops


@qml.QueuingManager.stop_recording()
def mux_multi_qubit_decomp(
    mats: list[np.ndarray], mux_wires: list, target_wires: list, n_b: int, break_down: bool
):
    """Decompose a multiplexed multi-qubit unitary via flag circuits.

    Args:
        mats (list[np.ndarray]): Matrices representing the unitary for each value of
            the ``mux_wires``.
        mux_wires (list): Multiplexing wires.
        target_wires (list): Wires on which the ``mats`` act.
        n_b (int): Base case decomposition to use in recursion. Must be one of ``1`` or ``2``.
        break_down (bool): Whether to decompose multiplexed single-qubit flags within the
            decomposition. The PO-QSD uses ``break_down=True`` because it targets the
            {Clifford+Rot} gate set. The ``recursive_flag_decomp`` uses ``break_down=False``
            because it aims to implement multiplexed single-qubit flags via ``QROM`` and phase
            gradients.

    Returns:

    """
    if n_b == len(target_wires):
        # Simply apply the base case decomposition to each matrix individually and attach
        # multiplexing nodes afterwards (using Multiplexer Extension Property)
        base_case_fn = one_qubit_flag_decomp if n_b == 1 else two_qubit_flag_decomp
        decs, diags = zip(*(base_case_fn(mat, target_wires) for mat in mats))
        muxed_ops = mux_ops(decs, controls=mux_wires)
        diag = np.concatenate(diags)

        if not break_down:
            return muxed_ops, diag

        # If requested, decompose the multiplexed single-qubit flags resulting from `mux_ops`
        new_ops, new_diag = decompose_mux_single_qubit_flags(muxed_ops)
        return new_ops, diag * new_diag

    K00, K01, theta_y, K10, K11 = zip(*[csd(mat) for mat in mats])
    K0 = list(chain.from_iterable(zip(K00, K01)))
    K1 = list(chain.from_iterable(zip(K10, K11)))
    new_mux = target_wires[:1]
    new_targets = target_wires[1:]
    # Recurse on the first circuit part, multiplexed K10 and K11
    ops1, diag1 = mux_multi_qubit_decomp(K1, mux_wires + new_mux, new_targets, n_b, break_down)
    # Balance the diagonal into an RZ multiplexer that can be combined with theta_y multiplexer
    # and a diagonal that can be merged into the multiplexed K00 and K01
    theta_z, diag_mid = balance_diagonal(diag1, len(mux_wires))
    theta_y = np.concatenate(theta_y)
    ops_a = [MultiplexedFlag(theta_z, theta_y, mux_wires + new_targets + new_mux)]

    diag_mid = expand_diagonal_matrix(diag_mid, mux_wires + new_targets, mux_wires + target_wires)
    if break_down:
        # Decompose the multiplexer if break_down is requested.
        ops_a, diag_a = decompose_mux_single_qubit_flags(ops_a)
        diag_mid = diag_mid * diag_a

    # Merge diag_mid into K0, in bits.
    sub_size = len(K0[0])
    K0 = [k * diag_mid[sub_size * i : sub_size * (i + 1)] for i, k in enumerate(K0)]

    # Recurse on the first circuit part, multiplexed K10 and K11
    ops0, diag0 = mux_multi_qubit_decomp(K0, mux_wires + new_mux, new_targets, n_b, break_down)

    return ops1 + ops_a + ops0, diag0


def _selective_de_multiplexed_branch(matrix, wires, n_b):
    """Selective de-multiplexing branch of recursive_flag_decomp_cliff_rz."""
    K00, K01, theta_Y, K10, K11 = csd(matrix)
    controls = wires[1:]
    target = wires[0]

    # Selective de-multiplexing branch
    M10, theta_Z_prime, M11 = de_mux(K10, K11)

    F11, Delta = recursive_flag_decomp_cliff_rz(M11, controls, n_b=n_b, selective_demux=True)
    F11, Delta_11_mod = decompose_mux_single_qubit_flags(F11)
    Delta = Delta * Delta_11_mod

    F_Z = mottonen(theta_Z_prime, controls, target, axis="Z", symmetrized="right")

    # The following de-multiplexing only is done to enable symmetrized Mottonen
    M00, theta_Z_0, M01 = de_mux(K00, K01)
    M01_new, theta_Y_new, M10_new = re_and_de_mux(M01, M10, theta_Y, wires, side="left")

    F10, Delta_prime = recursive_flag_decomp_cliff_rz(
        M10_new * Delta, controls, n_b=n_b, selective_demux=True
    )
    F10, Delta_10_mod = decompose_mux_single_qubit_flags(F10)
    Delta_prime = Delta_prime * Delta_10_mod

    F_Y = mottonen(theta_Y_new, controls, target, axis="Y", symmetrized="right")

    F_top = F11 + F_Z + F10 + F_Y

    _cz = qml.CZ([controls[0], target]).matrix(wire_order=wires)
    re_mux_rhs = (
        np.kron(np.eye(2), M00)
        @ qml.matrix(mottonen(theta_Z_0, controls, target, axis="Z"), wire_order=wires)
        @ np.kron(np.eye(2), M01_new)
        @ _cz
    )
    n = len(wires)
    K00 = re_mux_rhs[: 2 ** (n - 1), : 2 ** (n - 1)] * Delta_prime
    K01 = re_mux_rhs[2 ** (n - 1) :, 2 ** (n - 1) :] * Delta_prime
    return F_top, K00, K01


def _non_de_multiplexed_branch(matrix, wires, n_b):
    """Non-de-multiplexing branch of recursive_flag_decomp_cliff_rz."""
    # 1. Cosine-Sine Decomposition
    K00, K01, theta_Y, K10, K11 = csd(matrix)
    controls = wires[1:]
    target = wires[:1]

    # Standard n_b = 1 recursive branch
    F_top, Delta_top = mux_multi_qubit_decomp(
        [K10, K11], mux_wires=target, target_wires=controls, n_b=n_b, break_down=True
    )

    # Combine the diagonals and split them into a Z-rotation and a residual diagonal
    theta_Z, Delta_prime = balance_diagonal(Delta_top, 0)

    F_A = [MultiplexedFlag(theta_Z, theta_Y, controls + target)]
    F_A, Delta_prime_mod = decompose_mux_single_qubit_flags(F_A)
    N = len(Delta_prime_mod)
    Delta_prime0 = Delta_prime * Delta_prime_mod[: N // 2]
    Delta_prime1 = Delta_prime * Delta_prime_mod[N // 2 :]
    K00, K01 = [K00 * Delta_prime0, K01 * Delta_prime1]

    # attach a multiplexer control to F10 and F11 based on the target qubit
    F_top = F_top + F_A
    return F_top, K00, K01


@qml.QueuingManager.stop_recording()
def recursive_flag_decomp_cliff_rz(
    matrix: np.ndarray, wires: list, n_b: int = 2, selective_demux: bool = False
) -> tuple[list, np.ndarray]:
    """Recursive flag decomposition as used within the parameter-optimal Quantum
    Shannon Decomposition.

    Args:
        matrix (np.ndarray): Unitary matrix to decompose.
        wires (list): Wires on which ``matrix`` acts.
        n_b (int): Base case decomposition to use
        selective_demux (bool): Whether to use selective de-multiplexing.

    Returns:
        tuple[list, np.ndarray]: List of operations and a one-dimensional array
        representing a diagonal.

    """
    n = len(wires)

    # Base cases
    if n_b == 1 and n == 1:
        return one_qubit_flag_decomp(matrix, wires)
    if n_b == 2 and n == 2:
        ops, diag = two_qubit_flag_decomp(matrix, wires)
        ops, _ = decompose_mux_single_qubit_flags(ops)
        return ops, diag

    controls = wires[1:]
    target = wires[:1]

    if selective_demux:
        F_top, K00, K01 = _selective_de_multiplexed_branch(matrix, wires, n_b)
    else:
        F_top, K00, K01 = _non_de_multiplexed_branch(matrix, wires, n_b)

    # Common continuation for K00 and K01 blocks
    F_bottom, Delta_out = mux_multi_qubit_decomp(
        [K00, K01],
        mux_wires=target,
        target_wires=controls,
        n_b=n_b,
        break_down=True,
    )
    return F_top + F_bottom, Delta_out


def recursive_flag_decomp(matrix: np.ndarray, wires):
    """Recursive flag decomposition to the phase gradient gate set.

    Args:
        matrix (np.ndarray): Unitary matrix to decompose.
        wires (list): Wires on which ``matrix`` acts.

    Returns:
        list: List of multiplexers and a diagonal Operation (``qml.DiagonalQubitUnitary``)
        implementing ``matrix``.

    Note that this function queues the decomposition it computes to the queuing system in
    PennyLane.
    """
    ops, diag = mux_multi_qubit_decomp([matrix], [], wires, n_b=1, break_down=False)
    if qml.QueuingManager.recording():
        for op in ops:
            qml.apply(op)
    ops.append(qml.DiagonalQubitUnitary(diag, wires))
    return ops
