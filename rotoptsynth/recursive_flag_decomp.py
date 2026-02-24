import copy
from functools import partial
from itertools import chain

import numpy as np
import pennylane as qml
from pennylane.math.decomposition import zyz_rotation_angles
from pennylane.operation import Operation

from .asymmetric_decomp import asymmetric_decomp
from .linalg import balance_diagonal, csd, de_mux, merge_diagonals, mottonen, re_and_de_mux


@qml.QueuingManager.stop_recording()
def one_qubit_flag_decomp(V: np.ndarray, wires: list) -> tuple[list, np.ndarray]:
    """
    Implements the 1-qubit flag decomposition (Algorithm 3), returning the flag operations
    and the trailing diagonal.
    """
    phi, theta, omega, alpha = zyz_rotation_angles(V, return_global_phase=True)
    F = [MultiplexedFlag(phi, theta, wires)]
    Delta = np.exp(1j * np.array([-omega / 2 + alpha, omega / 2 + alpha]))
    return F, Delta


_cnot = qml.CNOT([0, 1]).matrix()
_y = qml.RY(-np.pi / 2, 0).matrix()


@qml.QueuingManager.stop_recording()
def two_qubit_flag_decomp(v, wires):
    a, b, c, d, alpha, Psi, Theta, Phi = asymmetric_decomp(_cnot @ v)

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


class HackyString:
    """A sneaky string class that appends itself one character at a time when added to a string,
    instead of being added entirely. This is used to create an operator label
    for MultiplexedFlag that differs on different wires of the operator."""

    def __init__(self, string):
        self._orig_str = string
        self._str = iter(string)

    def __radd__(self, other):
        return other + next(self._str)

    def replace(self, old, new, count=-1):
        return HackyString(self._orig_str.replace(old, new, count))


class MultiplexedFlag(Operation):
    num_params = 2
    ndim_params = (1, 1)

    def __init__(self, theta_z, theta_y, wires):
        if isinstance(wires, int):
            wires = [wires]
        n = len(wires)
        if n == 1:
            theta_z = np.atleast_1d(theta_z)
            theta_y = np.atleast_1d(theta_y)
        assert len(theta_z) == len(theta_y) == 2 ** (n - 1)
        super().__init__(theta_z, theta_y, wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return HackyString("◑" * (len(self.wires) - 1) + "⚑")

    @staticmethod
    def compute_decomposition(theta_z, theta_y, wires):  # pylint: disable=arguments-differ
        if len(wires) == 1:
            return [qml.RZ(theta_z[0], wires), qml.RY(theta_y[0], wires)]

        ops = []
        if not np.allclose(theta_z, 0.0):
            ops.extend(mottonen(theta_z, wires[:-1], wires[-1], axis="Z"))
        if not np.allclose(theta_y, 0.0):
            ops.extend(mottonen(theta_y, wires[:-1], wires[-1], axis="Y"))
        if qml.QueuingManager.recording():
            for op in ops:
                qml.apply(op)

        return ops


@partial(qml.matrix, wire_order=[0])
def _flag(theta_z, theta_y):
    qml.RZ(theta_z, 0)
    qml.RY(theta_y, 0)


def _decompose_mux_single_qubit_flag_base_case(theta_z, theta_y, wires):
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

    if qml.QueuingManager.recording():
        for op in final_ops:
            qml.apply(op)

    return final_ops, diag


@qml.QueuingManager.stop_recording()
def _merge_diag_into_mux(op, main_diag, main_diag_wires):
    target = op.wires[-1]
    balance_idx = main_diag_wires.index(target)
    main_diag_wires.pop(balance_idx)
    theta, main_diag = balance_diagonal(main_diag, balance_idx)
    missing_ctrls = [w for w in op.wires[:-1] if w not in main_diag_wires]
    theta_broad = np.diag(
        qml.math.expand_matrix(
            np.diag(np.kron(np.ones(2 ** len(missing_ctrls)), theta)),
            wires=missing_ctrls + main_diag_wires,
            wire_order=op.wires[:-1],
        )
    )
    op = MultiplexedFlag(op.data[0] + theta_broad, op.data[1], op.wires)
    return op, main_diag, main_diag_wires


@qml.QueuingManager.stop_recording()
def decompose_mux_single_qubit_flags(ops):
    """Transform that sweeps through a circuit, decomposing multiplexed single-qubit flags
    and moves remainder diagonals along."""

    tape = qml.tape.QuantumScript(ops)

    main_diags = []
    main_diag_wires = []

    new_ops = []
    for op in ops:  # tape.operations:
        if isinstance(op, MultiplexedFlag):
            for j, (d, d_wires) in enumerate(zip(main_diags, main_diag_wires, strict=True)):
                if op.wires[-1] in d_wires:
                    assert all(
                        w in op.wires for w in d_wires
                    ), "can't merge a diagonal into a smaller MultiplexedFlag."
                    op, d, d_wires = _merge_diag_into_mux(op, d, d_wires)
                    main_diags[j] = d
                    main_diag_wires[j] = d_wires
            if len(op.wires) == 1:
                new_ops.extend([qml.RZ(op.data[0][0], op.wires), qml.RY(op.data[1][0], op.wires)])
                continue
            _ops, _d = _decompose_mux_single_qubit_flag(op)
            new_ops.extend(_ops)
            main_diags.append(_d)
            main_diag_wires.append(list(op.wires))
            continue
        if isinstance(op, qml.CZ):
            pass
        elif isinstance(op, qml.CNOT):
            assert all(op.wires[1] not in d_wires for d_wires in main_diag_wires)
        elif isinstance(op, (qml.RZ, qml.RY)):
            assert all(op.wires[0] not in d_wires for d_wires in main_diag_wires)
        elif isinstance(op, qml.GlobalPhase):
            pass
            # main_diag *= np.exp(-1j*op.data[0])
            # continue
        else:
            raise NotImplementedError(f"{op=}")
        new_ops.append(op)

    wires = sorted(tape.wires)
    main_diag = np.ones(2 ** len(wires))
    for d, d_wires in zip(main_diags, main_diag_wires, strict=True):
        main_diag = main_diag * np.diag(
            qml.math.expand_matrix(np.diag(d), wires=d_wires, wire_order=wires)
        )
    return new_ops, main_diag


@qml.QueuingManager.stop_recording()
def mux_ops(ops: list, controls: list) -> list:
    """
    Attaches k control wires to 2**k lists of operations, exploiting the Multiplexer
    Extension Property. If a gate is static (no parameters), it is applied unconditionally
    to save control nodes.

    Args:
        ops (list[list]): Operations to apply for different control values
        control (int): The control wire.

    Returns:
        list: A new list of optimized controlled PennyLane operations.
    """
    n = len(controls)
    assert len(ops) == 2**n
    muxed_ops = []

    if n == 1:
        for op0, op1 in zip(*ops):
            assert isinstance(op0, type(op1)) and op0.wires == op1.wires
            # Check if the operation is static (0 parameters, e.g., CNOT, X, Y, Z)
            if op0.num_params == 0:
                # The gate is identical in both branches, so it acts unconditionally.
                muxed_ops.append(op0)
            elif isinstance(op0, MultiplexedFlag):
                new_op = MultiplexedFlag(
                    np.concatenate([op0.data[0], op1.data[0]]),
                    np.concatenate([op0.data[1], op1.data[1]]),
                    controls + op0.wires,
                )
                muxed_ops.append(new_op)
            elif isinstance(op0, qml.RZ):
                new_op = MultiplexedFlag(
                    np.array([op0.data[0], op1.data[0]]),
                    np.zeros(
                        2,
                    ),
                    controls + op0.wires,
                )
                muxed_ops.append(new_op)
            elif isinstance(op0, qml.RY):
                new_op = MultiplexedFlag(
                    np.zeros(
                        2,
                    ),
                    np.array([op0.data[0], op1.data[0]]),
                    controls + op0.wires,
                )
                muxed_ops.append(new_op)
            else:
                raise NotImplementedError(f"Can't multiplex {op0} and {op1}.")
        return muxed_ops

    ops0 = mux_ops(ops[: 2 ** (n - 1)], controls=controls[1:])
    ops1 = mux_ops(ops[2 ** (n - 1) :], controls=controls[1:])
    muxed_ops = mux_ops([ops0, ops1], controls=controls[:1])

    return muxed_ops


@qml.QueuingManager.stop_recording()
def mux_multi_qubit_decomp(mats, mux_wires, target_wires, n_b, break_down):
    """Decompose a multiplexed multi-qubit unitary via flag circuits."""
    if n_b == len(target_wires):
        base_case_fn = one_qubit_flag_decomp if n_b == 1 else two_qubit_flag_decomp
        decs, diags = zip(*(base_case_fn(mat, target_wires) for mat in mats))
        muxed_ops = mux_ops(decs, controls=mux_wires)
        diag = np.concatenate(diags)
        if not break_down:
            return muxed_ops, diag

        new_ops, new_diag = decompose_mux_single_qubit_flags(muxed_ops)
        return new_ops, diag * new_diag

    K00, K01, theta_Y, K10, K11 = zip(*[csd(mat) for mat in mats])
    K0 = list(chain.from_iterable(zip(K00, K01)))
    K1 = list(chain.from_iterable(zip(K10, K11)))
    ops1, diag1 = mux_multi_qubit_decomp(
        K1, mux_wires + target_wires[:1], target_wires[1:], n_b, break_down
    )
    theta_Y = np.concatenate(theta_Y)
    theta_Z, diag_mid = balance_diagonal(diag1, len(mux_wires))
    F_A = [MultiplexedFlag(theta_Z, theta_Y, mux_wires + target_wires[1:] + target_wires[:1])]
    diag_mid = np.diag(
        qml.math.expand_matrix(
            np.diag(diag_mid),
            wires=mux_wires + target_wires[1:],
            wire_order=mux_wires + target_wires,
        )
    )
    if break_down:
        F_A, diag_a = decompose_mux_single_qubit_flags(F_A)
        diag_mid = diag_mid * diag_a
    sub_size = len(K0[0])
    K0 = [k * diag_mid[sub_size * i : sub_size * (i + 1)] for i, k in enumerate(K0)]
    ops0, diag0 = mux_multi_qubit_decomp(
        K0, mux_wires + target_wires[:1], target_wires[1:], n_b, break_down
    )

    return ops1 + F_A + ops0, diag0


@qml.QueuingManager.stop_recording()
def recursive_flag_decomp_cliff_rz(
    V: np.ndarray, wires: list, n_b: int = 2, selective_demux: bool = False
) -> tuple[list, np.ndarray]:
    """ """
    n = len(wires)

    # Base cases
    if n_b == 1 and n == 1:
        return one_qubit_flag_decomp(V, wires)
    if n_b == 2 and n == 2:
        ops, diag = two_qubit_flag_decomp(V, wires)
        ops, _ = decompose_mux_single_qubit_flags(ops)
        return ops, diag

    # 1. Cosine-Sine Decomposition
    K00, K01, theta_Y, K10, K11 = csd(V)

    controls = wires[1:]
    target = wires[0]

    if selective_demux:
        # Selective de-multiplexing branch
        M10, theta_Z_prime, M11 = de_mux(K10, K11)

        F11, Delta = recursive_flag_decomp_cliff_rz(
            M11, controls, n_b=n_b, selective_demux=selective_demux
        )
        F11, Delta_11_mod = decompose_mux_single_qubit_flags(F11)
        Delta = Delta * Delta_11_mod

        F_Z = mottonen(theta_Z_prime, controls, target, axis="Z", symmetrized="right")

        # The following de-multiplexing only is done to enable symmetrized Mottonen
        M00, theta_Z_0, M01 = de_mux(K00, K01)
        M01_new, theta_Y_new, M10_new = re_and_de_mux(M01, M10, theta_Y, wires, side="left")

        F10, Delta_prime = recursive_flag_decomp_cliff_rz(
            M10_new * Delta, controls, n_b=n_b, selective_demux=selective_demux
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
        K00 = re_mux_rhs[: 2 ** (n - 1), : 2 ** (n - 1)]
        K01 = re_mux_rhs[2 ** (n - 1) :, 2 ** (n - 1) :]
        Delta_prime0 = Delta_prime1 = Delta_prime

    else:
        # Standard n_b = 1 recursive branch
        F_top, Delta_top = mux_multi_qubit_decomp(
            [K10, K11], mux_wires=[target], target_wires=controls, n_b=n_b, break_down=True
        )

        # Combine the diagonals and split them into a Z-rotation and a residual diagonal
        theta_Z, Delta_prime = balance_diagonal(Delta_top, 0)

        F_A = [MultiplexedFlag(theta_Z, theta_Y, controls + [target])]
        F_A, Delta_prime_mod = decompose_mux_single_qubit_flags(F_A)
        N = len(Delta_prime_mod)
        Delta_prime0 = Delta_prime * Delta_prime_mod[: N // 2]
        Delta_prime1 = Delta_prime * Delta_prime_mod[N // 2 :]

        # attach a multiplexer control to F10 and F11 based on the target qubit
        F_top = F_top + F_A

    # Common continuation for K00 and K01 blocks
    F_bottom, Delta_out = mux_multi_qubit_decomp(
        [K00 * Delta_prime0, K01 * Delta_prime1],
        mux_wires=[target],
        target_wires=controls,
        n_b=n_b,
        break_down=True,
    )
    return F_top + F_bottom, Delta_out


def recursive_flag_decomp(V: np.ndarray, wires):
    """ """
    ops, diag = mux_multi_qubit_decomp([V], [], wires, n_b=1, break_down=False)
    if qml.QueuingManager.recording():
        for op in ops:
            qml.apply(op)
    ops.append(qml.DiagonalQubitUnitary(diag, wires))
    return ops
