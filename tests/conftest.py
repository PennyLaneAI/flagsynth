import pytest
import rotoptsynth as ros

@pytest.fixture(scope="function")
def enable_disable_validation():
    """enable and disable validation around a test."""
    ros.enable_validation()
    try:
        yield
    finally:
        ros.disable_validation()

@pytest.fixture(scope="function")
def disable_validation():
    """disable validation in a test."""
    ros.disable_validation()
