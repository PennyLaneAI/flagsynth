"""This module implements a pure recursive cosine sine decomposition (CSD) for orthogonal
matrices, leading to a parameter-optimal synthesis. It is not optimized to reduce the number of
entangling gates."""

import pennylane as qml
import numpy as np
from .linalg import real_csd

def ortho_synth(matrix: np.ndarray, wires: list, validate:bool=False) -> list:
    groups = [("K", [matrix])]
    for i, w in enumerate(wires):
        new_groups = []
        for group in groups:
            if group[0] == "K":
                # Group is a group of gates
                k0 = []
                k1 = []
                theta_y = []
                for mat in group[1]:
                    _k00, _k01, _theta_y, _k10, _k11 = real_csd(mat, validate)
                    k0.extend([_k00, _k01])
                    theta_y.append(_theta_y)
                    k1.extend([_k10, _k11])
                new_groups += [("K", k0), ("A", np.concatenate(theta_y), w), ("K", k1)]
            else:
                # Group is a list of angles, just preserve it
                new_groups.append(group)
        groups = new_groups

    if validate:
        assert all(all(m.shape==(1, 1) and np.isclose(m[0][0], 1.) for m in group[1]) for group in groups if group[0]=="K")

    circuit = [
        qml.SelectPauliRot(group[1], [w for w in wires if w!=group[2]], group[2], rot_axis="Y")
        for group in reversed(groups) if group[0]=="A"
    ]
    return circuit


