# Rot-Opt-Synth
Unitary synthesis with the optimal number of rotation gates as presented in
*Parameter-optimal unitary synthesis with flag decompositions* by
Kottmann et al. [arXiv:unknown.id](https://arxiv.org/abs/unknown.id)

## Installation

```
git clone git@github.com:XanaduAI/rotoptsynth.git
cd rotoptsynth
pip install .
```

## Main functionality

The primary functionality of this package consists of the functions `rotoptsynth.po_qsd`,
implementing the parameter-optimal Quantum Shannon Decomposition (PO-QSD)
into the {Clifford+Rot} gate set, and `todo`, decomposing into QROMs and adders onto a resource
phase gradient state.
These functions are written to be used with [PennyLane](pennylane.ai) and return sequences of
PennyLane gates. They respect PennyLane's queuing system and thus can be used as-is
in `qml.QNode`s.

### Examples

Let's look at an example usage of `rotoptsynth.po_qsd` and validate that it correctly implements
a (randomly sampled) target matrix:

```python3
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import rotoptsynth as ros

# Pick some system size and generate a target with that size.
n = 3
U = unitary_group.rvs(2**n, random_state=129)
wires = list(range(n))
```
```pycon
>>> implemented_matrix = qml.matrix(ros.po_qsd, wire_order=wires)(U, wires)
>>> np.allclose(implemented_matrix, U)
True
```

We may also look at the synthesized circuit:
```pycon
>>> print(qml.draw(ros.po_qsd)(U, wires))
0: ──RZ(-0.75)────────────────────────────────────────────────────────────╭X──RZ(0.57)─╭X ···
1: ──RZ(0.94)───RY(1.72)─╭●──RZ(10.22)──RY(-1.99)─╭●──RZ(-1.00)──RY(1.12)─│────────────╰● ···
2: ──RZ(3.70)───RY(1.45)─╰Z──RZ(11.37)──RY(1.02)──╰Z──RZ(7.12)───RY(2.14)─╰●───────────── ···

0: ··· ──RZ(0.76)─╭X──RZ(-0.27)──RY(-0.26)─────────────────────────────────────────────────╭X ···
1: ··· ───────────│───RZ(10.04)──RY(2.20)──╭●──RZ(2.76)───RY(-2.12)─╭●──RZ(8.52)──RY(0.92)─│─ ···
2: ··· ───────────╰●──RZ(10.28)──RY(1.72)──╰Z──RZ(10.24)──RY(0.60)──╰Z──RZ(9.45)──RY(0.97)─╰● ···

0: ··· ──RY(1.54)─╭X──RY(1.07)─╭X──RY(-0.12)─╭X─────────RZ(-1.35)──────────────────────── ···
1: ··· ───────────╰●───────────│─────────────╰●─────────RZ(11.75)──RY(2.14)─╭●──RZ(13.38) ···
2: ··· ────────────────────────╰●──RZ(0.80)───RY(0.90)──────────────────────╰Z──RZ(2.14)─ ···

0: ··· ───────────────────────────────────╭X──RZ(1.28)─╭X──RZ(0.93)─╭X──RZ(2.13)─┤  
1: ··· ──RY(-1.84)─╭●──RZ(8.85)──RY(1.61)─│────────────╰●───────────│──╭U(M0)────┤  
2: ··· ──RY(0.78)──╰Z──RZ(9.84)──RY(0.71)─╰●────────────────────────╰●─╰U(M0)────┤
```

We see the fully decomposed structure of the PO-QSD due to the target gate set {Clifford+Rot},
except for the two-qubit `qml.QubitUnitary` at the end, denoted as `U(M0)`. It can be decomposed
with the standard technique for such unitaries, which is readily available in PennyLane.


