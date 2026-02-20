from functools import partial
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.math.decomposition import zyz_rotation_angles
from .asymmetric_decomp import asymmetric_decomp
from .linalg import balance_diagonal, mottonen, merge_diagonals

def one_qubit_flag_decomp(V: np.ndarray, wires: list) -> tuple[list, np.ndarray]:
    """
    Implements the 1-qubit flag decomposition (Algorithm 3), returning the flag operations
    and the trailing diagonal.
    """
    phi, theta, omega, alpha = zyz_rotation_angles(V, return_global_phase=True)
    with qml.QueuingManager.stop_recording():
        F = [MultiplexedFlag(phi, theta, wires)]
    Delta = np.exp(1j*np.array([-omega/2+alpha, omega/2 + alpha]))
    return F, Delta

_cnot = qml.CNOT([0, 1]).matrix()
_y = qml.RY(-np.pi/2, 0).matrix()

def two_qubit_flag_decomp(v, wires):
    a, b, c, d, alpha, Psi, Theta, Phi = asymmetric_decomp(_cnot @ v)

    with qml.QueuingManager.stop_recording():
        phi_a, theta_a, omega_a = zyz_rotation_angles(a)
        phi_c, theta_c, omega_c = zyz_rotation_angles(c)
        phi_d, theta_d, omega_d = zyz_rotation_angles(_y @ d)
        _ops = [qml.RZ(omega_d, wires[1]),qml.RX(-Phi, wires[1])]
        phi_e, theta_e, omega_e = zyz_rotation_angles(qml.matrix(_ops, wires[1:]))
        _ops = [qml.RX(omega_e, wires[1])]
        phi_f, theta_f, omega_f = zyz_rotation_angles(b @ qml.matrix(_ops, wires[1:])@_y.conj().T)

        F = [
            MultiplexedFlag(phi_c, theta_c, wires[0]),
            MultiplexedFlag(phi_d, theta_d, wires[1]),
            qml.CZ(wires),
            MultiplexedFlag(omega_c+np.pi/2, Theta, wires[0]),
            MultiplexedFlag(phi_e, theta_e, wires[1]),
            qml.CZ(wires),
            MultiplexedFlag(phi_a-np.pi/2, theta_a, wires[0]),
            MultiplexedFlag(phi_f, theta_f, wires[1]),
        ]
        Delta = np.diag(qml.matrix([
            qml.RZ(omega_a, wires[0]),
            qml.RZ(omega_f, wires[1]),
            qml.IsingZZ(Psi, wires),
            qml.GlobalPhase(-alpha),
        ], wire_order=wires))

    return F, Delta


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
        assert len(theta_z) == len(theta_y) == 2**(n-1)
        super().__init__(theta_z, theta_y, wires)

    @staticmethod
    def compute_decomposition(theta_z, theta_y, wires):
        if len(wires) == 1:
            return [qml.RZ(theta_z[0], wires), qml.RY(theta_y[0], wires)]

        ops = []
        if not np.allclose(theta_z, 0.):
            ops.extend(mottonen(theta_z, wires[:-1], wires[-1], axis='Z'))
        if not np.allclose(theta_y, 0.):
            ops.extend(mottonen(theta_y, wires[:-1], wires[-1], axis='Y'))
        if qml.QueuingManager.recording():
            for op in ops:
                qml.apply(op)

        return ops

@partial(qml.matrix, wire_order=[0])
def _flag(theta_z, theta_y):
    qml.RZ(theta_z, 0)
    qml.RY(theta_y, 0)

def _decompose_mux_single_qubit_flag_base_case(theta_z, theta_y, wires):
    K0, K1 = [_flag(thz, thy) for thz, thy in zip(theta_z, theta_y)]
    X = K0 @ K1.conj().T
    phi = np.angle(np.linalg.det(X))
    arg_a = np.angle(X[0, 0] * np.exp(-1j*phi/2))
    rho0 = (np.pi-phi)/4 - arg_a/2
    rho1 = (3*np.pi-phi)/4 + arg_a/2
    r = np.exp(1j*np.array([rho0, rho1]))
    Y = np.diag(r) @ X * r
    _I, L0 = np.linalg.eig(Y)
    assert np.allclose(_I, np.array([1j, -1j])), "Fix ordering"
    d = np.exp(1j*np.pi/4 * np.array([1, -1]))
    L1 = np.diag(d) @ L0.conj().T @ np.diag(r.conj())@K1
    D = np.concatenate([d, d.conj()])
    R = np.concatenate([r.conj(), r])

    L1 = np.diag(np.array([1, -1j])) @ L1
    D = np.exp(-1j * np.pi/4) * np.array([1, 1j, 1j, -1]) * D
    R = np.exp(1j * np.pi/4) * R * np.array([1, 1, -1j, -1j])

    # Gate decomposition
    phi1, theta1, omega1, alpha1 = zyz_rotation_angles(L1, return_global_phase=True)
    phi2, theta2, omega2, alpha2 = zyz_rotation_angles(L0, return_global_phase=True)
    ops = [
        qml.RZ(phi1, wires[1]),
        qml.RY(theta1, wires[1]),
        qml.CZ(wires),
        qml.RZ(omega1+phi2, wires[1]),
        qml.RY(theta2, wires[1]),
    ]
    Delta = (
        np.diag(qml.matrix(qml.RZ,wire_order=wires)(omega2, wires[1]))
        * R
        * np.exp(1j*(alpha1+alpha2))
    )
    return ops, Delta

def _decompose_mux_single_qubit_flag(op):
    assert isinstance(op, MultiplexedFlag)
    theta_z, theta_y = op.data
    wires = op.wires
    n = len(wires)
    if n == 2:
        return _decompose_mux_single_qubit_flag_base_case(theta_z, theta_y, wires)

    with qml.QueuingManager.stop_recording():
        N = 2**(n-2)
        all_ops, all_diags = zip(*[
            _decompose_mux_single_qubit_flag_base_case(
                theta_z[i::N],
                theta_y[i::N],
                [wires[0], wires[-1]],
            ) for i in range(N)
        ])

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



def _merge_diag_into_mux(op, main_diag, main_diag_wires):
    #print(f"{op.wires=}")
    #print(f"{main_diag_wires=}")
    target = op.wires[-1]
    #print(f'{target=}')
    balance_idx = main_diag_wires.index(target)
    #print(f"{balance_idx=}")
    main_diag_wires.pop(balance_idx)
    theta, main_diag = balance_diagonal(main_diag, balance_idx)
    #print(f"{theta=}")
    #print(f"{main_diag=}")
    missing_ctrls = [w for w in op.wires[:-1] if w not in main_diag_wires]
    #print(f"{missing_ctrls=}")
    #print(f"wires: {missing_ctrls+main_diag_wires=}")
    #print(f"wire_order: {op.wires[:-1]=}")
    theta_broad = np.diag(qml.math.expand_matrix(np.diag(np.kron(np.ones(2**len(missing_ctrls)), theta)), wires=missing_ctrls+main_diag_wires, wire_order=op.wires[:-1]))
    op = MultiplexedFlag(op.data[0] + theta_broad, op.data[1], op.wires)
    return op, main_diag, main_diag_wires

def _apply_cnot_to_diag(diag, diag_wires, cnot_wires):
    assert cnot_wires[1] in diag_wires
    if cnot_wires[0] not in diag_wires:
        diag = np.kron(np.eye(2), diag)
        diag_wires = [cnot_wires[0]] + diag_wires

    _cnot = qml.CNOT(cnot_wires).matrix(wire_order=diag_wires)
    diag = np.diag(_cnot @ np.diag(diag) @ _cnot)
    return diag, diag_wires

def decompose_mux_single_qubit_flags(ops):
    """Transform that sweeps through a circuit, decomposing multiplexed single-qubit flags
    and moves remainder diagonals along."""

    #def proc(results):
        #return results[0]

    tape = qml.tape.QuantumScript(ops)
    print(qml.drawer.tape_text(tape, show_matrices=False, wire_order=sorted(tape.wires)))
    print("\n"*3)

    main_diags = []
    main_diag_wires = []

    new_ops = []
    for op in ops:#tape.operations:
        #print(f"Handling {op=}. {main_diag_wires=}")
        if isinstance(op, MultiplexedFlag):
            for j, (d, d_wires) in enumerate(zip(main_diags, main_diag_wires, strict=True)):
                if op.wires[-1] in d_wires:
                    assert all(w in op.wires for w in d_wires), "can't merge a diagonal into a smaller MultiplexedFlag."
                    op, d, d_wires = _merge_diag_into_mux(op, d, d_wires)
                    main_diags[j] = d
                    main_diag_wires[j] = d_wires
            if len(op.wires) == 1:
                new_ops.extend([qml.RZ(op.data[0][0], op.wires), qml.RY(op.data[1][0], op.wires)])
                continue
            _ops, _d = _decompose_mux_single_qubit_flag(op)
            #main_diag, main_diag_wires = merge_diagonals(main_diag, main_diag_wires, _d, op.wires)
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
            #main_diag *= np.exp(-1j*op.data[0])
            #continue
        else:
            raise NotImplementedError(f"{op=}")
        new_ops.append(op)

    wires = sorted(tape.wires)
    main_diag = np.ones(2**len(wires))
    for d, d_wires in zip(main_diags, main_diag_wires, strict=True):
        main_diag = main_diag * np.diag(qml.math.expand_matrix(np.diag(d), wires=d_wires, wire_order=wires))
    return new_ops, main_diag
    return (tape.copy(operations=new_ops),), proc


def mux_ops(ops0: list, ops1: list, control: int) -> list:
    """
    Attaches a control wire to two lists of operations, exploiting the Multiplexer
    Extension Property. If a gate is static (no parameters), it is applied unconditionally
    to save control nodes.

    Args:
        ops0 (list): Operations to apply if control is 0.
        ops1 (list): Operations to apply if control is 1.
        control (int): The control wire.

    Returns:
        list: A new list of optimized controlled PennyLane operations.
    """
    muxed_ops = []

    with qml.QueuingManager.stop_recording():
        for op0, op1 in zip(ops0, ops1):
            assert isinstance(op0, type(op1)) and op0.wires == op1.wires
            # Check if the operation is static (0 parameters, e.g., CNOT, X, Y, Z)
            if op0.num_params == 0:
                # The gate is identical in both branches, so it acts unconditionally.
                muxed_ops.append(op0)
            elif isinstance(op0, MultiplexedFlag):
                new_op = MultiplexedFlag(
                    np.concatenate([op0.data[0], op1.data[0]]),
                    np.concatenate([op0.data[1], op1.data[1]]),
                    [control] + op0.wires,
                )
                muxed_ops.append(new_op)
            elif isinstance(op0, qml.RZ):
                new_op = MultiplexedFlag(
                    np.array([op0.data[0], op1.data[0]]),
                    np.zeros(2,),
                    [control] + op0.wires,
                )
                muxed_ops.append(new_op)
            elif isinstance(op0, qml.RY):
                new_op = MultiplexedFlag(
                    np.zeros(2,),
                    np.array([op0.data[0], op1.data[0]]),
                    [control] + op0.wires,
                )
                muxed_ops.append(new_op)
            else:
                raise NotImplementedError(f"Can't multiplex {op0} and {op1}.")

    return muxed_ops
