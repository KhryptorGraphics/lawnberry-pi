"""HardwareDriver ABC defining lifecycle methods for all drivers.

Implements T021 requirements: async init/start/stop/health_check with
minimal typing and docstrings. Drivers for sensors, motors, GPS, etc. must
inherit from this base class to ensure consistent behavior and safety.
"""

from __future__ import annotations

import abc
import os
from typing import Any

# Pi 5's RP1 GPIO bank is /dev/gpiochip4 (chip 0 is root-only and unopenable by the
# service user → lgpio "can not open gpiochip"); Pi 4 uses chip 0. Probe 4 then 0.
# Override with LAWNBERRY_GPIOCHIP=<comma-list>, e.g. "0" on a Pi 4.
GPIOCHIP_ENV = "LAWNBERRY_GPIOCHIP"


def open_gpiochip(lgpio: Any) -> int:
    """Open the usable GPIO bank via lgpio, returning its handle.

    Probes candidate chip numbers (Pi 5 chip 4 first, then chip 0) so the same
    code works across Pi models and udev layouts. Raises the last error if none
    open, matching lgpio.gpiochip_open() semantics for callers that catch it.
    """
    override = os.environ.get(GPIOCHIP_ENV)
    chips = [int(c) for c in override.split(",") if c.strip()] if override else [4, 0]
    last_err: Exception = RuntimeError("no gpiochip candidates")
    for chip in chips:
        try:
            return lgpio.gpiochip_open(chip)
        except Exception as e:  # lgpio raises on missing/inaccessible chip
            last_err = e
    raise last_err


class HardwareDriver(abc.ABC):
    """Abstract base class for hardware drivers.

    Lifecycle:
    - initialize(): allocate resources, open device handles
    - start(): begin streaming or periodic operations
    - stop(): cease operations and release as needed
    - health_check(): return quick health snapshot
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config: dict[str, Any] = config or {}
        self.initialized: bool = False
        self.running: bool = False

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize hardware resources."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Start active operations (if applicable)."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop operations and release transient resources (keep handles if needed)."""

    @abc.abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Return a health snapshot suitable for /health. Must be non-blocking."""
