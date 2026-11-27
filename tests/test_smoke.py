"""Smoke test: the package is installed and its metadata is readable."""

import importlib
from importlib.metadata import version


def test_package_is_importable_and_has_a_version() -> None:
    """The package imports and its distribution reports a version."""
    importlib.import_module("invoice_recognition")

    dist_version = version("invoice-recognition")
    assert dist_version
    assert dist_version[0].isdigit()
