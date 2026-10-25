"""Process-wide record of the safety stops that end an autonomous run (audit SAF-01).

A *stop event* is any safety function that stopped the machine on its own: the E-stop routes
and the orchestrator, the watchdog trip and the command-deadman escalation (both end in the
tractor's ``emergency_stop``), a tilt latch, a lock-out or a park, the RoboHAT stop hold, an
operator's revoke of the authorization. Each one calls :func:`record_stop` at the moment it
latches, synchronously and before it waits for anything.

Two things use it:

* **The epoch** (:func:`epoch`) counts the events. A mission or AI run captures it when it
  starts and compares it before every command it sends; a run that sees it move has been
  stopped by something other than itself and must end -- whatever the latches say by then.
  That is what makes "a clear can never re-activate a run" hold even for a loop that
  survived the stop (a long await, a failed abort): the clear releases the latch, never the
  run's right to command.
* **The listeners** (:func:`add_listener`) are told at once, so an autonomous run is
  aborted (mission marked ``aborted``, navigation mode back to ``idle``) in the same event-loop
  turn as the stop instead of one control tick later. They run synchronously and must be
  cheap and never block: a stop is never delayed by its listeners, and a listener that raises
  is logged and skipped. A bound method is held weakly, so a service that goes away is not
  kept alive (or called) by this registry.

Nothing here moves an actuator.
"""

from __future__ import annotations

import inspect
import logging
import weakref
from collections.abc import Callable

logger = logging.getLogger(__name__)

#: ``listener(reason, source)``: why the machine was stopped and who asked for it.
StopListener = Callable[[str, str], None]

_epoch = 0
_listeners: list[Callable[[], StopListener | None]] = []


def epoch() -> int:
    """How many safety stops this process has recorded (a run compares before and after)."""
    return _epoch


def add_listener(listener: StopListener) -> None:
    """Call ``listener(reason, source)`` on every stop event (idempotent per listener)."""
    if any(ref() == listener for ref in _listeners):
        return
    if inspect.ismethod(listener):
        _listeners.append(weakref.WeakMethod(listener))
    else:
        _listeners.append(lambda: listener)


def remove_listener(listener: StopListener) -> None:
    """Stop calling ``listener`` (a listener that was never added is ignored)."""
    _listeners[:] = [ref for ref in _listeners if ref() not in (None, listener)]


def record_stop(reason: str, source: str = "") -> int:
    """A safety function just stopped the machine: bump the epoch and tell every listener.

    Synchronous and exception-free by contract: it is called from the stop paths themselves
    (before their first await), so it must neither wait nor fail. Returns the new epoch.
    """
    global _epoch
    _epoch += 1
    for ref in list(_listeners):
        listener = ref()
        if listener is None:
            _listeners[:] = [r for r in _listeners if r is not ref]
            continue
        try:
            listener(reason, source)
        except Exception:
            logger.exception("stop listener %r failed (stop: %s from %s)", listener, reason, source)
    return _epoch


__all__ = ["StopListener", "add_listener", "epoch", "record_stop", "remove_listener"]
