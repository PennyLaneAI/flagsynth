# Rot-Opt-Synth
Unitary synthesis with the optimal number of rotation gates as presented in
*Parameter-optimal unitary synthesis with flag decompositions* by
Kottmann et al. [arXiv:unknown.id](https://arxiv.org/abs/unknown.id)

## Installation

```
git clone git@github.com:XanaduAI/flagsynth.git
cd flagsynth
pip install .
```

## Main functionality

The primary functionality of this package consists of the functions `flagsynth.po_qsd`,
implementing the parameter-optimal Quantum Shannon Decomposition (PO-QSD)
into the {Clifford+Rot} gate set, and `flagsynth.recursive_flag_decomp`, decomposing into 
QROMs and adders onto a resource phase gradient state.
These functions are written to be used with [PennyLane](pennylane.ai) and return sequences of
PennyLane gates. They respect PennyLane's queuing system and thus can be used as-is
in `qml.QNode`s.

### Examples

Let's look at an example usage of `flagsynth.po_qsd` and validate that it correctly implements
a (randomly sampled) target matrix:

```python3
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
import flagsynth as fs

# Pick some system size and generate a target with that size.
n = 3
U = unitary_group.rvs(2**n, random_state=129)
wires = list(range(n))
```
```pycon
>>> implemented_matrix = qml.matrix(fs.po_qsd, wire_order=wires)(U, wires)
>>> np.allclose(implemented_matrix, U)
True
```

We may also look at the synthesized circuit:
```pycon
>>> print(qml.draw(fs.po_qsd)(U, wires))
0: ──RZ(-0.75)────────────────────────────────────────────────────────────╭X──RZ(0.57)─╭X ···
1: ──RZ(0.94)───RY(1.72)─╭●──RZ(10.22)──RY(-1.99)─╭●──RZ(-1.00)──RY(1.12)─│────────────╰● ···
2: ──RZ(3.70)───RY(1.45)─╰Z──RZ(11.37)──RY(1.02)──╰Z──RZ(7.12)───RY(2.14)─╰●───────────── ···

0: ··· ──RZ(0.76)─╭X──RZ(-0.27)──RY(-0.26)─────────────────────────────────────────────────╭X ···
1: ··· ───────────│───RZ(10.04)──RY(2.20)──╭●──RZ(2.76)───RY(-2.12)─╭●──RZ(8.52)──RY(0.92)─│─ ···
2: ··· ───────────╰●──RZ(10.28)──RY(1.72)──╰Z──RZ(10.24)──RY(0.60)──╰Z──RZ(9.45)──RY(0.97)─╰● ···

0: ··· ──RY(1.54)─╭X──RY(1.07)─╭X──RY(-0.12)─╭X─────────RZ(-1.35)──────────────────────── ···
1: ··· ───────────╰●───────────│─────────────╰●─────────RZ(11.75)──RY(2.14)─╭●──RZ(13.38) ···
2: ··· ────────────────────────╰●──RZ(0.80)───RY(0.90)──────────────────────╰Z──RZ(2.14)─ ···

0: ··· ───────────────────────────────────╭X──RZ(1.28)─╭X──RZ(0.93)─╭X──RZ(2.13)─────────── ···
1: ··· ──RY(-1.84)─╭●──RZ(8.85)──RY(1.61)─│────────────╰●───────────│───RZ(11.17)──RY(1.50) ···
2: ··· ──RY(0.78)──╰Z──RZ(9.84)──RY(0.71)─╰●────────────────────────╰●──RZ(3.32)───RY(1.13) ···

0: ··· ─────────────────────────────────────────────────────────────────────── ···
1: ··· ──RZ(3.55)─╭X──RZ(0.17)─╭●───────────╭X──RZ(2.50)───RY(2.00)──RZ(10.12) ···
2: ··· ──RZ(1.75)─╰●──RY(1.83)─╰X──RY(2.16)─╰●──RZ(10.81)──RY(1.47)──RZ(10.77) ···

0: ··· ─╭GlobalPhase(-1.02)─┤
1: ··· ─├GlobalPhase(-1.02)─┤
2: ··· ─╰GlobalPhase(-1.02)─┤
```

We see the fully decomposed structure of the PO-QSD due to the target gate set {Clifford+Rot}.

Alternatively, we may want to decompose to multiplexed single-qubit flags, which can
then be implemented efficiently with ``QROM``s, ``Adder``s and phase gradient resources states,
realizing the ``RZ`` and ``RY`` rotations of the flags at the same time or sequentially (depending
on desired work qubit usage). See the paper for details.

```pycon
>>> n = 4
>>> wires = list(range(n))
>>> U = unitary_group.rvs(2**n, random_state=81512)
>>> print(qml.draw(fs.recursive_flag_decomp, show_matrices=False)(U, wires))
0: ─╭◑─╭◑─╭◑─╭◑─╭◑─╭◑─╭◑─╭⚑─╭◑─╭◑─╭◑─╭◑─╭◑─╭◑─╭◑─╭U(M0)─┤  
1: ─├◑─├◑─├◑─├⚑─├◑─├◑─├◑─├◑─├◑─├◑─├◑─├⚑─├◑─├◑─├◑─├U(M0)─┤  
2: ─├◑─├⚑─├◑─├◑─├◑─├⚑─├◑─├◑─├◑─├⚑─├◑─├◑─├◑─├⚑─├◑─├U(M0)─┤  
3: ─╰⚑─╰◑─╰⚑─╰◑─╰⚑─╰◑─╰⚑─╰◑─╰⚑─╰◑─╰⚑─╰◑─╰⚑─╰◑─╰⚑─╰U(M0)─┤  
```

Note the trailing ``U(M0)`` which denotes a ``qml.DiagonalQubitUnitary`` operation, i.e.,
a diagonal on all qubits. It can be decomposed similarly to a multiplexed ``RZ`` gate.

## Usage with PennyLane

Until the functionality is merged into PennyLane itself, ``flagsynth`` registers the
main functionalities as decompositions of ``qml.QubitUnitary``, using 
the graph-based decomposition system:

```python
n = 3
qml.decomposition.enable_graph()

gate_set = {"RZ", "RY", "CNOT", "CZ", "GlobalPhase"}

@qml.decompose(gate_set=gate_set)
@qml.qnode(qml.device("lightning.qubit", wires=n))
def my_qnode(U):
    qml.QubitUnitary(U, wires=range(n))
    return qml.expval(qml.X(1))

U = unitary_group.rvs(2**n, random_state=81212)
```
```pycon
>>> print(my_qnode(U))
-0.1703878223917237
```

## License

See LICENSE file.
