"""Python package for parameter-optimal unitary synthesis techniques."""
import jax
jax.config.update("jax_enable_x64", True)

from .asym_decomp import asymmetric_two_qubit_decomp
from .diag_decomps import attach_multiplexer_node, diag_decomp
from .full_decomp import rot_opt_synth
from .utils import aiii_kak, count_clifford, ops_to_mat, count_rotation_angles
from .validation import disable_validation, enable_validation, validation_enabled
