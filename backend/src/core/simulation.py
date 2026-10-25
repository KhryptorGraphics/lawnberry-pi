"""The ONE reader of the ``SIM_MODE`` environment variable (audit SAF-07, BL-06).

Simulation weakens every safety check in the process: an absent output reads ``simulated`` and
counts as a confirmed stop, GPIO is never touched, credentials and the posture log are relaxed.
A flag that decides that must be parsed once, strictly, and read the same way everywhere. Before
this module the safety stack accepted ``1``/``true``/``True`` while the posture check and the
navigation service compared against ``"1"``/``"0"`` only, so ``SIM_MODE=true`` (a ``.env`` typo
over the unit's ``SIM_MODE=0``) made the stop paths report simulated confirmations while
navigation took its real-hardware branches in the same process.

The rule, fail closed:

* the value ``1`` (surrounding whitespace ignored) is simulation, and nothing else is;
* unset, empty, ``0`` and ``false``/``no``/``off`` (any case) are real hardware;
* any other value (``true``, ``yes``, ``2``, a typo) is **not** simulation either, and is logged
  once at WARNING with what to write instead, so a bench user who typed ``SIM_MODE=true`` finds
  out why the mower now tries to open its hardware.

Nothing else in ``backend/`` may read ``SIM_MODE`` from the environment:
``tests/unit/test_simulation_mode_parser.py`` fails on any such read. Import
:func:`is_simulation_mode`. The module imports the standard library only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: The environment variable this module owns.
SIM_MODE_ENV = "SIM_MODE"
#: The only value that means simulation.
SIMULATION_TOKEN = "1"
#: Values that say, in so many words, "real hardware" (the systemd unit ships ``0``).
HARDWARE_TOKENS = frozenset({"", "0", "false", "no", "off"})

_warned: set[str] = set()


def parse_sim_mode(raw: str | None) -> tuple[bool, bool]:
    """``(simulated, recognised)`` for a raw ``SIM_MODE`` value.

    ``simulated`` is true only for ``1``. ``recognised`` is false for a value that is neither
    ``1`` nor an explicit hardware token: it is treated as not simulation, and reported.
    """
    value = (raw or "").strip()
    if value == SIMULATION_TOKEN:
        return True, True
    return False, value.lower() in HARDWARE_TOKENS


def is_simulation_mode(environ: Mapping[str, str] | None = None) -> bool:
    """True iff ``SIM_MODE`` is exactly ``1`` in ``environ`` (default: ``os.environ``)."""
    raw = (os.environ if environ is None else environ).get(SIM_MODE_ENV)
    simulated, recognised = parse_sim_mode(raw)
    if not recognised and raw not in _warned:
        _warned.add(raw)
        logger.warning(
            "%s=%r is not a recognised value: treating it as NOT simulation (real hardware, "
            "real safety checks). Write %s=1 for simulation or %s=0 for hardware.",
            SIM_MODE_ENV,
            raw,
            SIM_MODE_ENV,
            SIM_MODE_ENV,
        )
    return simulated


__all__ = [
    "HARDWARE_TOKENS",
    "SIMULATION_TOKEN",
    "SIM_MODE_ENV",
    "is_simulation_mode",
    "parse_sim_mode",
]
