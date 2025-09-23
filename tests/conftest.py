import pytest
import rotoptsynth as ros


@pytest.fixture(scope="function")
def with_validation():
    """Run test with validation."""
    was_enabled = ros.validation_enabled()
    if not was_enabled:
        ros.enable_validation()
        try:
            yield
        finally:
            ros.disable_validation()
    else:
        yield


@pytest.fixture(scope="function")
def without_validation():
    """Run test without validation."""
    was_enabled = ros.validation_enabled()
    if was_enabled:
        ros.disable_validation()
        try:
            yield
        finally:
            ros.enable_validation()
    else:
        yield
