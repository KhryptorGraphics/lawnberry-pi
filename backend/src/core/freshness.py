"""Monotonic ages for readings that only carry a wall-clock stamp (audit SAF-05, US-02).

ToF, ultrasonic and GNSS readings are stamped with the host's wall clock when they are made
(``datetime.now(UTC)``). Judging how old one is as ``wall_now - stamp`` is right only while the
wall clock never steps: if chrony/NTP, a GPS discipline or a manual ``date`` moves it back after
a reading was taken, the stamp lies in the future, the age is negative, and clamping a negative
age to 0 (the old code) keeps a dead sensor's last value "fresh" until wall time catches up,
however long the source has been silent.

:class:`ReceiptAges` fixes that by reading the wall clock ONCE per reading, when the reading is
first seen (producer and consumer are milliseconds apart, so a step between them is the
exception), converting that age into a time on a monotonic clock, and remembering it per
source. The same stamp seen again (the sensor interface replays ``last_reading`` when the
receiver goes quiet) returns the remembered time, so the reading keeps ageing on the monotonic
clock whatever the wall clock does afterwards. A stamp that cannot be aged (missing, garbled,
or from the future when first seen) is **unknown, never fresh**: the caller must treat the
source as having no usable data.

The module imports the standard library only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


def parse_stamp(stamp: Any) -> datetime | None:
    """An aware datetime from a ``datetime`` or an ISO-8601 string; ``None`` if unusable.

    A naive datetime is read as UTC, the convention of every model in this code base.
    """
    if isinstance(stamp, str):
        try:
            stamp = datetime.fromisoformat(stamp)
        except ValueError:
            return None
    if not isinstance(stamp, datetime):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp


class ReceiptAges:
    """Per-source memory of when a stamped reading was first seen (see the module docstring).

    ``clock`` is the monotonic clock the returned times are on (the consumer's own clock, so
    ``consumer_clock() - taken_at(...)`` is the age).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        # key -> (stamp last seen, its monotonic time, or None for "cannot be aged")
        self._seen: dict[str, tuple[datetime, float | None]] = {}

    def taken_at(
        self, key: str, stamp: Any, wall_now: datetime, now: float | None = None
    ) -> float | None:
        """Monotonic time at which the measurement stamped ``stamp`` was taken, or ``None``.

        ``now`` is the current time on the monotonic clock (default: the clock itself).
        ``None`` means the age is unknowable (no usable stamp, or a stamp from the future when
        the reading was first seen) and stays ``None`` for that stamp even after wall time
        catches up with it: only a different stamp (a new reading) can be judged afresh. A reading
        with no usable stamp is ``None`` too and leaves what is remembered untouched, so the old
        stamp that comes back after it (a replay) is still aged from when it was first seen.
        """
        when = parse_stamp(stamp)
        if when is None:
            return None  # what is remembered for ``key`` stays: a bare reading forgets nothing
        known = self._seen.get(key)
        if known is not None and known[0] == when:
            return known[1]
        wall_age = (wall_now - when).total_seconds()
        taken = None if wall_age < 0.0 else (self._clock() if now is None else now) - wall_age
        self._seen[key] = (when, taken)
        return taken

    def forget(self, key: str | None = None) -> None:
        """Drop what is remembered for ``key`` (all keys when omitted)."""
        if key is None:
            self._seen.clear()
        else:
            self._seen.pop(key, None)


__all__ = ["ReceiptAges", "parse_stamp"]
