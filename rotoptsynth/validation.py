"""This module contains a toggle for validation within RotOptSynth's functions, as well as
some basic validation functions for matrices."""

import numpy as np


def toggle_validation():
    """A closure that toggles the validation of synthesis inputs and outputs on and off."""

    _validate = True

    def enable():
        """
        A global toggle for enabling the validation of synthesis inputs and outputs.
        When activated, validates user inputs and computed outputs of RotOptSynth synthesis
        techniques, at the cost of reduced performance.

        Validation is enabled by default.
        """

        nonlocal _validate
        _validate = True

    def disable() -> None:
        """
        A global toggle for disabling the validation of synthesis inputs and outputs.
        When activated, validates user inputs and computed outputs of RotOptSynth synthesis
        techniques, at the cost of reduced performance.

        Validation is enabled by default.
        """

        nonlocal _validate
        _validate = False

    def status() -> bool:
        """
        Return whether the validation of synthesis inputs and outputs is enabled.

        Validation is enabled by default.
        """

        nonlocal _validate
        return _validate

    return enable, disable, status


enable_validation, disable_validation, validation_enabled = toggle_validation()


def _is_square(arr):
    """Return whether an array is 2D and square."""
    return len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]


def is_unitary(arr):
    """Return whether an array represents a unitary matrix."""
    return _is_square(arr) and np.allclose(arr.conj().T @ arr, np.eye(len(arr)))


def is_orthogonal(arr):
    """Return whether an array represents an orthogonal matrix."""
    return _is_square(arr) and np.allclose(arr.T @ arr, np.eye(len(arr)))


def is_symmetric(arr):
    """Return whether an array represents a symmetric matrix."""
    return _is_square(arr) and np.allclose(arr, arr.T)


def has_unit_determinant(arr):
    """Return whether an array has unit determinant."""
    return np.isclose(np.linalg.det(arr), 1.0)


def is_block_diagonal(arr, first_block_size):
    """Return whether an array is block-diagonal with two blocks, where the first
    block has size ``first_block_size``."""
    return np.allclose(arr[:first_block_size, first_block_size:], 0.0) and np.allclose(
        arr[first_block_size:, :first_block_size], 0.0
    )
