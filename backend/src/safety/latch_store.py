"""Durable E-stop and interlock latches (audit SAF-16).

The latches are in-memory flags, and the unit runs with ``Restart=always`` (10 s): a crash, an
out-of-memory kill or an operator's ``systemctl restart`` used to release every one of them
silently, ten seconds after the stop, with nobody at the machine. This module makes them
survive a restart.

What is saved (one small JSON file in the data directory, ``emergency_latch.json``):

* ``process_estop``      the process-wide latch the E-stop routes and the orchestrator set;
* ``tractor_estop``      the tractor's own E-stop latch (watchdog, deadman escalation, its route);
* ``tractor_interlock``  the blade interlock latch, with its reason (tilt);
* ``robohat_hold``       the differential platform's stop hold.

The file always mirrors the live latches: every stop and every clear rewrites it, and it is
removed when nothing is latched. At start-up :func:`restore` re-latches whatever the file
names, before anything can actuate; the latch then needs the same authenticated explicit
clear as any other (``POST /control/emergency_clear``, ``POST /tractor/clear-emergency``).
Authorization is never saved: a restarted tractor always starts revoked. A file that exists
but cannot be read or understood fails CLOSED: every latch is assumed set.

Writing is atomic and durable (temp file in the same directory created ``0600``, ``flush``,
``fsync``, ``os.replace``, ``fsync`` of the directory) and happens on ONE dedicated writer
thread, so a slow disk never blocks the event loop that carries the deadman and the stop
itself, and writes land in the order they were requested (a clear can never be overtaken by
the stop's older snapshot). A write that fails is logged at CRITICAL and reported to the
caller (``PersistResult``); the in-memory latch holds regardless.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..services.robohat_service import RoboHATService
    from ..services.tractor_service import TractorControlService

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
FILE_NAME = "emergency_latch.json"
#: Overrides the whole path (tests, unusual layouts); default ``$LAWNBERRY_DATA_DIR/FILE_NAME``.
ENV_FILE = "LAWNBERRY_LATCH_FILE"

PROCESS_ESTOP = "process_estop"
TRACTOR_ESTOP = "tractor_estop"
TRACTOR_INTERLOCK = "tractor_interlock"
ROBOHAT_HOLD = "robohat_hold"
LATCH_NAMES = (PROCESS_ESTOP, TRACTOR_ESTOP, TRACTOR_INTERLOCK, ROBOHAT_HOLD)


def latch_path() -> Path:
    """Where the latches are saved: ``$LAWNBERRY_LATCH_FILE``, else ``<data dir>/FILE_NAME``."""
    explicit = os.getenv(ENV_FILE)
    if explicit:
        return Path(explicit)
    return Path(os.getenv("LAWNBERRY_DATA_DIR", "./data")) / FILE_NAME


@dataclass(frozen=True)
class PersistResult:
    """Outcome of one save: ``ok`` is False when the latch could NOT be made durable."""

    ok: bool
    latched: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class LatchSet:
    """What a previous process left latched (``unreadable`` set: assume everything is)."""

    latches: dict[str, str | None] = field(default_factory=dict)
    unreadable: str | None = None

    def __bool__(self) -> bool:
        return bool(self.latches) or self.unreadable is not None

    def has(self, name: str) -> bool:
        """Is ``name`` latched (an unreadable file latches every name)?"""
        return self.unreadable is not None or name in self.latches


# --------------------------------------------------------------------------- live state


def collect_live(
    tractor: TractorControlService | None = None, robohat: RoboHATService | None = None
) -> dict[str, str | None]:
    """The latches that are set right now: ``{name: reason or None}``. Never raises.

    ``tractor``/``robohat`` default to the process singletons; a service that is calling this
    about itself passes itself, so a stop on an instance that is not the singleton is still
    what gets saved. A source that cannot be read counts as latched (fail closed).
    """
    from ..core.globals import _safety_state

    live: dict[str, str | None] = {}
    try:
        if _safety_state.get("emergency_stop_active", False):
            live[PROCESS_ESTOP] = None
    except Exception:
        live[PROCESS_ESTOP] = "state unreadable"
    try:
        if tractor is None:
            from ..services.tractor_service import get_tractor_service

            tractor = get_tractor_service()
        if tractor.enabled:
            if tractor.state.emergency_stop_active:
                live[TRACTOR_ESTOP] = None
            if tractor.state.interlock_latched:
                live[TRACTOR_INTERLOCK] = tractor.state.latch_reason
    except Exception as exc:
        logger.exception("Tractor latch state unreadable: saving it as latched")
        live[TRACTOR_ESTOP] = f"state unreadable: {type(exc).__name__}"
    try:
        if robohat is None:
            from ..services.robohat_service import get_robohat_service

            robohat = get_robohat_service()
        if robohat is not None and robohat.emergency_latched:
            live[ROBOHAT_HOLD] = None
    except Exception as exc:
        logger.exception("RoboHAT latch state unreadable: saving it as latched")
        live[ROBOHAT_HOLD] = f"state unreadable: {type(exc).__name__}"
    return live


# ------------------------------------------------------------------------------- writing

_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="latch-store")
_last_error: str | None = None
_state_lock = threading.Lock()


def last_error() -> str | None:
    """The last save failure (cleared by the next success), for status displays."""
    with _state_lock:
        return _last_error


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # a filesystem that cannot open a directory has nothing to sync
    try:
        os.fsync(fd)
    except OSError:
        logger.warning("fsync of %s failed", directory, exc_info=True)
    finally:
        os.close(fd)


def _write_file(path: Path, live: dict[str, str | None]) -> None:
    """Make ``path`` hold exactly ``live`` (removed when it is empty). Raises on failure."""
    if not live:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        _fsync_dir(path.parent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "latches": live,
    }
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".emergency_latch-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _save(path: Path, live: dict[str, str | None]) -> PersistResult:
    global _last_error
    try:
        _write_file(path, live)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with _state_lock:
            _last_error = error
        logger.critical(
            "EMERGENCY LATCH NOT SAVED to %s (%s): a restart would release %s",
            path,
            error,
            sorted(live) or "nothing (the clear was not recorded either)",
        )
        return PersistResult(False, tuple(sorted(live)), error)
    with _state_lock:
        _last_error = None
    return PersistResult(True, tuple(sorted(live)))


def request_persist(
    tractor: TractorControlService | None = None, robohat: RoboHATService | None = None
) -> Future[PersistResult]:
    """Save the live latches; returns at once (the write runs on the writer thread).

    The snapshot is taken NOW, on the calling thread (so it is exactly the state the caller
    just set or cleared), the write is queued behind every earlier one. Never raises: a
    failure is in the returned future's ``PersistResult``.
    """
    path = latch_path()
    try:
        live = collect_live(tractor, robohat)
    except Exception as exc:  # collect_live does not raise; defence for a future edit
        logger.exception("latch snapshot failed")
        done: Future[PersistResult] = Future()
        done.set_result(PersistResult(False, (), f"snapshot failed: {type(exc).__name__}: {exc}"))
        return done
    return _writer.submit(_save, path, live)


async def wait(pending: Future[PersistResult], timeout: float = 5.0) -> PersistResult:
    """Await a save without ever raising (a late write keeps going on its own thread)."""
    try:
        async with asyncio.timeout(timeout):
            return await asyncio.shield(asyncio.wrap_future(pending))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}" or "timed out"
        logger.critical("EMERGENCY LATCH save not confirmed: %s", error)
        return PersistResult(False, (), error)


async def persist(
    tractor: TractorControlService | None = None,
    robohat: RoboHATService | None = None,
    timeout: float = 5.0,
) -> PersistResult:
    """Save the live latches and wait for the write. Never raises."""
    return await wait(request_persist(tractor, robohat), timeout)


# ------------------------------------------------------------------------------- reading


def load() -> LatchSet:
    """What the file says is latched. A missing file is nothing; an unreadable one is everything."""
    path = latch_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LatchSet()
    except Exception as exc:
        return LatchSet(unreadable=f"{path} cannot be read: {type(exc).__name__}: {exc}")
    try:
        data: Any = json.loads(raw)
        if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
            raise ValueError(f"not a version {FORMAT_VERSION} latch record")
        latches = data.get("latches")
        if not isinstance(latches, dict):
            raise ValueError("'latches' is not an object")
        clean: dict[str, str | None] = {}
        for name, reason in latches.items():
            if not isinstance(name, str) or not (reason is None or isinstance(reason, str)):
                raise ValueError(f"malformed latch entry {name!r}")
            clean[name] = reason
    except Exception as exc:
        return LatchSet(unreadable=f"{path} is corrupt: {type(exc).__name__}: {exc}")
    return LatchSet(clean)


def restore(
    tractor: TractorControlService | None = None, robohat: RoboHATService | None = None
) -> LatchSet:
    """Re-latch what the previous process left latched. Call once, before anything actuates.

    Sets the process-wide latch, the tractor's E-stop / blade interlock latch and (when the
    service exists) the RoboHAT hold, logs every latch at CRITICAL and returns what was
    found. Nothing is cleared here and authorization is not restored. Never raises.
    """
    from ..core.globals import _safety_state

    try:
        found = load()
    except Exception as exc:  # load() does not raise; fail closed if it ever does
        found = LatchSet(unreadable=f"latch restore failed: {type(exc).__name__}: {exc}")
    if not found:
        return found
    why = found.unreadable or ", ".join(
        f"{name}{f' ({reason})' if reason else ''}"
        for name, reason in sorted(found.latches.items())
    )
    logger.critical(
        "EMERGENCY LATCH RESTORED from %s: %s. The machine stays stopped until an operator "
        "clears it (POST /api/v2/control/emergency_clear and "
        "POST /api/v2/tractor/clear-emergency).",
        latch_path(),
        why,
    )
    if found.has(PROCESS_ESTOP):
        _safety_state["emergency_stop_active"] = True
    try:
        if tractor is not None:
            tractor.restore_latches(
                estop=found.has(TRACTOR_ESTOP) or found.has(PROCESS_ESTOP),
                interlock=(
                    found.latches.get(TRACTOR_INTERLOCK) or "restored"
                    if found.has(TRACTOR_INTERLOCK)
                    else None
                ),
                why=why,
            )
    except Exception:
        logger.exception("Restoring the tractor latch failed")
    try:
        if robohat is not None and (found.has(ROBOHAT_HOLD) or found.has(PROCESS_ESTOP)):
            robohat.restore_emergency_hold()
    except Exception:
        logger.exception("Restoring the RoboHAT hold failed")
    return found


__all__ = [
    "ENV_FILE",
    "FILE_NAME",
    "LATCH_NAMES",
    "LatchSet",
    "PersistResult",
    "collect_live",
    "last_error",
    "latch_path",
    "load",
    "persist",
    "request_persist",
    "restore",
    "wait",
]
