import pennylane as qml
import numpy as np
from scipy.linalg import cossin

def ops_to_mat(ops, wire_order):
    return qml.matrix(qml.tape.QuantumScript(ops), wire_order=wire_order)


def aiii_kak(u, p, q, validate):
    if validate:
        assert u.shape == (p + q, p + q)
    if p == 0 or q == 0:
        return u, np.eye(p + q), np.eye(p + q)
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(u, p=p, q=p, swap_sign=True, separate=False)

    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)
        f[:, :p] = np.roll(f[:, :p], q - p, axis=1)
        f[:p] = np.roll(f[:p], q - p, axis=0)

    if validate:
        dim = u.shape[0]
        r = min(p, q)
        s = max(p, q)
        assert np.allclose(k1[:p, p:], 0.0) and np.allclose(k1[p:, :p], 0.0)
        assert np.allclose(k1 @ k1.conj().T, np.eye(dim))
        assert np.allclose(k2[:p, p:], 0.0) and np.allclose(k2[p:, :p], 0.0)
        assert np.allclose(k2 @ k2.conj().T, np.eye(dim))

        assert np.allclose(f[r:s, r:s], np.eye(s - r))
        assert np.allclose(np.diag(np.diag(f[:r, :r])), f[:r, :r])
        assert np.allclose(f[:r, :r], f[s:, s:])
        assert np.allclose(np.diag(np.diag(f[:r, s:])), f[:r, s:])
        assert np.allclose(f[:r, s:], -f[s:, :r])
        assert np.allclose(f[r:s, :r], 0.0)
        assert np.allclose(f[r:s, s:], 0.0)
        assert np.allclose(f[:r, r:s], 0.0)
        assert np.allclose(f[s:, r:s], 0.0)
        assert np.allclose(k1 @ f @ k2, u), f"\n{k1}\n{f}\n{k2}\n{k1 @ f @ k2}\n{u}"

    return k1, f, k2

