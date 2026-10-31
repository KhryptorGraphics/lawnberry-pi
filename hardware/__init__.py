"""Package root for the hardware model and control tooling.

Present so ``python -m hardware.control.scenarios`` and ``hardware.control.*`` imports work
from the repository root (``pyproject.toml`` sets ``pythonpath = ["."]`` but only packages
``backend*``). This is not an installed distribution package.
"""
