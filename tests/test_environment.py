"""Packaging smoke check; this does not establish algorithm correctness."""

from importlib.metadata import distribution
from pathlib import Path

import ml_principles


def test_distribution_installs_importable_package() -> None:
    """Catch missing package discovery or an unavailable editable installation."""
    installed = distribution("ml-from-first-principles")
    assert installed.metadata["Name"] == "ml-from-first-principles"
    assert ml_principles.__file__ is not None
    assert Path(ml_principles.__file__).is_file()
