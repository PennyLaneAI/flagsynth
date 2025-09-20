import numpy as np

def toggle_validation():
    """A closure that toggles the validation of synthesis inputs and outputs on and off."""

    _VALIDATION = True

    def enable():
        """
        A global toggle for enabling the validation of synthesis inputs and outputs.
        When activated, validates user inputs and computed outputs of RotOptSynth synthesis
        techniques, at the cost of reduced performance.

        Validation is enabled by default.
        """

        nonlocal _VALIDATION
        _VALIDATION = True

    def disable() -> None:
        """
        A global toggle for disabling the validation of synthesis inputs and outputs.
        When activated, validates user inputs and computed outputs of RotOptSynth synthesis
        techniques, at the cost of reduced performance.

        Validation is enabled by default.
        """

        nonlocal _VALIDATION
        _VALIDATION = False

    def status() -> bool:
        """
        Return whether the validation of synthesis inputs and outputs is enabled.

        Validation is enabled by default.
        """

        nonlocal _VALIDATION
        return _VALIDATION

    return enable, disable, status

enable_validation, disable_validation, validation_enabled = toggle_validation()

def is_unitary(arr):
    return np.allclose(arr.conj().T @ arr, np.eye(len(arr)))

def is_orthogonal(arr):
    return np.allclose(arr.T @ arr, np.eye(len(arr)))

def is_symmetric(arr):
    return np.allclose(arr, arr.T)

def has_unit_determinant(arr):
    return np.isclose(np.linalg.det(arr), 1.)

def is_block_diagonal(arr, first_block_size):
    return (
        np.allclose(arr[:first_block_size, first_block_size:], 0.)
        and np.allclose(arr[first_block_size:, :first_block_size], 0.)
    )
