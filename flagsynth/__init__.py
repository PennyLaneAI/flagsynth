"""This packaged contains functionality for unitary parameter-optimal synthesis of unitaries
to either the {Clifford+Rot} gate set or multiplexed single-qubit flags / rotations to be
decomposed further into ``QROM``s, and ``Adder``s onto a phase gradient state.

This code accompanies the preprint "Parameter-optimal unitary synthesis with flag decompositions"
by Kottmann et al. `arXiv:2603.20376 <arxiv.org/abs/2603.20376>`__.
"""

from .asymmetric_decomp import asymmetric_decomp
from .linalg import balance_diagonal, csd, de_mux, mottonen, re_and_de_mux
from .sdm import sdm
from .multiplexed_flag import MultiplexedFlag
from .ortho_synth import ortho_synth
from .recursive_flag_decomp import (
    decompose_mux_single_qubit_flags,
    mux_multi_qubit_decomp,
    mux_ops,
    one_qubit_flag_decomp,
    recursive_flag_decomp,
    recursive_flag_decomp_cliff_rz,
    two_qubit_flag_decomp,
)
