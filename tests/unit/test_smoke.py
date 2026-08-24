"""Smoke tests for the package scaffold."""

from calibrated_reliability import __version__


def test_version() -> None:
    """The package exposes a semantic project version."""
    assert __version__ == "0.1.0"
