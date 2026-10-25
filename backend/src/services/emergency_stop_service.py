"""All-platform emergency stop and actuator parking with truthful results.

``execute_emergency_stop`` fans out to every stop path that exists in this
process -- the tractor service (levers neutral, throttle idle, blade PTO and
starter relays off), the RoboHAT drive/blade controller and the IBT-4 blade
service -- concurrently and independently, so one failing or hung path cannot
delay or skip another. The report says what each output *actually* did; it
never claims a stop that was not observed.

Output status vocabulary:

- ``stopped``      a real hardware path confirmed the stop: the firmware echoed
                   the command, the actuator command was applied, or, for the
                   IBT-4, both GPIO inputs were driven low and read back low.
                   None of these observe the machine itself (IBT-4 power stage,
                   blade motor, PTO); the output ``detail`` says what was seen
- ``simulated``    SIM_MODE: no hardware by design; counts as confirmed so SIM
                   stays testable
- ``unconfirmed``  the command was issued but its effect could not be confirmed
                   (firmware did not echo it, or the driver is a flag-only stub)
- ``failed``       attempted and failed (``error`` attached)
- ``not_present`` / ``not_enabled``  nothing to stop (absent / platform
                   disabled); reported ``simulated`` in SIM_MODE

Aggregates: ``motors_stopped`` (``blade_disabled``) is ``True`` only if at least
one drive (blade) output is confirmed (``stopped``/``simulated``) and none is
``failed`` or ``unconfirmed``. ``complete`` additionally requires that no output
anywhere failed. Absent hardware therefore never reads as "stopped" outside
SIM_MODE.

``park_actuators`` is the same fan-out for shutdown: it brings the actuators to
their resting state (blade off, levers neutral, throttle idle) without latching
an E-stop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.globals import _blade_state, _safety_state
from ..core.simulation import is_simulation_mode
from ..safety import latch_store, stop_events

logger = logging.getLogger(__name__)

#: Ceiling per stop path. A path that exceeds it is reported as failed but is
#: *not* cancelled: a half-run safing sequence is worse than a slow one.
PATH_TIMEOUT_S = 5.0

# Stops still driving the outputs (audit SEC-12). ``_stops_in_flight`` counts orchestrator calls
# between the latch and the report; ``_stop_paths`` holds every stop path task that has not
# finished, which includes a path that overran ``PATH_TIMEOUT_S`` and keeps running in the
# background. ``_stop_epoch`` counts stops started. The general clear (``api/safety.py``) is
# refused while ``stop_in_flight()`` is true and abandons itself if the epoch moves under it:
# a clear racing a stop used to release the RoboHAT hold the stop had just set, while the
# stop's own report still said latched and complete.
_stops_in_flight = 0
_stop_epoch = 0
_stop_paths: set[asyncio.Future[Any]] = set()


def stop_in_flight() -> bool:
    """Is an emergency stop still driving any output?"""
    return _stops_in_flight > 0 or bool(_stop_paths)


def stop_epoch() -> int:
    """How many emergency stops have started in this process (a clear compares before/after)."""
    return _stop_epoch


STOPPED = "stopped"
SIMULATED = "simulated"
UNCONFIRMED = "unconfirmed"
FAILED = "failed"
NOT_PRESENT = "not_present"
NOT_ENABLED = "not_enabled"

_CONFIRMED = {STOPPED, SIMULATED}
_ABSENT = {NOT_PRESENT, NOT_ENABLED}

# Which outputs of each path count towards motors_stopped / blade_disabled.
_DRIVE_OUTPUTS = {"tractor": {"left_lever", "right_lever", "throttle"}, "robohat": {"drive"}}
_BLADE_OUTPUTS = {"tractor": {"blade_pto"}, "robohat": {"blade"}, "blade_service": {"blade"}}


def _output(status: str, *, detail: str | None = None, error: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"status": status}
    if detail:
        out["detail"] = detail
    if error:
        out["error"] = error
    return out


def _absent(status: str, why: str) -> dict[str, Any]:
    """A path with nothing to stop: ``simulated`` in SIM_MODE, else ``status``."""
    if is_simulation_mode():
        return {
            "status": SIMULATED,
            "outputs": {},
            "detail": f"{why} (SIM_MODE: simulated)",
            "absent": True,
        }
    return {"status": status, "outputs": {}, "detail": why, "absent": True}


def _failed_path(error: str) -> dict[str, Any]:
    return {"status": FAILED, "outputs": {}, "error": error}


def _path_status(outputs: dict[str, dict[str, Any]]) -> str:
    statuses = {o["status"] for o in outputs.values()}
    for worst in (FAILED, UNCONFIRMED):
        if worst in statuses:
            return worst
    return SIMULATED if statuses == {SIMULATED} else STOPPED


def _tractor_path(result: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    for name, info in result.get("subsystems", {}).items():
        if not info["ok"]:
            outputs[name] = _output(FAILED, error=info["error"])
        else:
            outputs[name] = _output(SIMULATED if info["simulated"] else STOPPED)
    return {"status": _path_status(outputs), "outputs": outputs}


def _robohat_output(outcome: str, what: str) -> dict[str, Any]:
    if outcome == "acked":
        return _output(STOPPED)
    if outcome == "unacked":
        return _output(
            UNCONFIRMED,
            detail=f"{what} command written but the firmware did not echo it "
            "(firmware in RC mode, or no response)",
        )
    return _output(FAILED, error=f"{what} command could not be written to the serial port")


def _robohat_path(result: dict[str, Any]) -> dict[str, Any]:
    outputs = {
        "drive": _robohat_output(result["drive"], "neutral-PWM"),
        "blade": _robohat_output(result["blade"], "blade=off"),
    }
    return {"status": _path_status(outputs), "outputs": outputs}


def _blade_service_path(result: dict[str, Any]) -> dict[str, Any]:
    """Map a ``BladeService`` stop/park report to an output status.

    ``stopped`` needs a hardware-backed driver whose off write was CONFIRMED: ``off`` is True
    only after every write succeeded and both inputs read back low. A driver that was
    initialised but did not confirm it, or whose setup failed (``gpio_error``, e.g. GPIO25
    busy), is ``failed`` (attempted and failed), not absent. ``not_present`` therefore only
    describes an UNARMED service that was never initialised: an armed one (``ibt4.gpio_enabled``)
    is initialised by the stop itself (``BladeService``), so it ends up ``stopped`` or ``failed``.
    """
    error = result.get("error")
    gpio_error = result.get("gpio_error")
    if error or (result["initialized"] and not result["off"]):
        output = _output(
            FAILED, error=error or gpio_error or "blade driver refused the off command"
        )
    elif not result["initialized"] and gpio_error:
        output = _output(
            FAILED,
            error=f"IBT-4 GPIO setup failed, the blade output was NOT driven: {gpio_error}",
        )
    elif not result["initialized"]:
        return _absent(NOT_PRESENT, "IBT-4 blade service never initialised (nothing to switch off)")
    elif result["hardware_backed"]:
        pins = result.get("pins") or {}
        gpio = (
            f"GPIO{pins['in1']}/{pins['in2']}"
            if "in1" in pins and "in2" in pins
            else "the IBT-4 inputs"
        )
        output = _output(
            STOPPED,
            detail=f"{gpio} driven low and read back low; "
            "the IBT-4 power stage and the blade motor are not observed",
        )
    elif is_simulation_mode():
        output = _output(SIMULATED)
    else:
        output = _output(
            UNCONFIRMED,
            detail="IBT-4 GPIO adapter is a flag-only stub: no hardware effect, "
            "the blade output was NOT driven",
        )
    outputs = {"blade": output}
    return {"status": _path_status(outputs), "outputs": outputs}


def _start_path(
    factory: Callable[[], Awaitable[dict[str, Any]]], *, track: bool = False
) -> asyncio.Future[dict[str, Any]]:
    """Schedule one stop path as its own task; ``track`` registers it as a stop still in flight.

    A path is a task of its own, not a part of the caller: cancelling the caller cannot cancel
    it. ``track`` keeps ``stop_in_flight()`` true until the path finishes, even past its
    timeout. Synchronous on purpose (no ``await``): see ``_run_all``.
    """
    task = asyncio.ensure_future(factory())
    if track:
        _stop_paths.add(task)
        task.add_done_callback(_stop_paths.discard)
    return task


async def _await_path(
    name: str, task: asyncio.Future[dict[str, Any]], timeout_s: float
) -> tuple[str, dict[str, Any]]:
    """Wait for one stop path; never raises, never cancels a half-run safing sequence.

    ``asyncio.wait`` does not cancel its futures (unlike ``wait_for``), so a path that overruns
    ``timeout_s`` keeps running to completion in the background and is reported as failed.
    """
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    if not done:
        task.add_done_callback(
            lambda t: logger.warning(
                "Safing path %s finished late: %s",
                name,
                "failed" if t.cancelled() or t.exception() else "ok",
            )
        )
        logger.critical("Safing path %s did not finish within %.1fs", name, timeout_s)
        return name, _failed_path(f"did not finish within {timeout_s:.1f}s (still running)")
    try:
        return name, task.result()
    except Exception as exc:
        logger.exception("Safing path %s raised", name)
        return name, _failed_path(f"{type(exc).__name__}: {exc}")


async def _tractor(kind: str, source: str) -> dict[str, Any]:
    from . import tractor_service

    tractor = tractor_service.get_tractor_service()
    if not tractor.enabled:
        return _absent(NOT_ENABLED, tractor.not_enabled_text())
    result = await (tractor.emergency_stop(source) if kind == "stop" else tractor.park(source))
    return _tractor_path(result)


async def _robohat(kind: str, source: str) -> dict[str, Any]:
    from . import robohat_service

    robohat = robohat_service.get_robohat_service()
    if robohat is None or not (robohat.running and robohat.status.serial_connected):
        return _absent(NOT_PRESENT, "RoboHAT serial controller not connected")
    result = await (robohat.emergency_stop_detail() if kind == "stop" else robohat.park())
    if result.get("error") == "serial not connected":
        return _absent(NOT_PRESENT, "RoboHAT serial controller not connected")
    return _robohat_path(result)


async def _blade_service(kind: str, source: str) -> dict[str, Any]:
    from . import blade_service

    service = blade_service.get_blade_service()
    result = await (service.emergency_stop() if kind == "stop" else service.park())
    return _blade_service_path(result)


def _summarize(paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def confirmed(names: dict[str, set[str]]) -> bool:
        any_ok = False
        for path_name, outputs in names.items():
            path = paths.get(path_name)
            if path is None:
                continue
            if path["status"] == FAILED and not path["outputs"]:
                return False  # the whole path blew up or hung: its outputs are unknown
            for out_name in outputs:
                out = path["outputs"].get(out_name)
                if out is None:
                    # SIM_MODE absent paths carry no outputs but count as simulated.
                    any_ok = any_ok or (path.get("absent") and path["status"] == SIMULATED)
                    continue
                if out["status"] in (FAILED, UNCONFIRMED):
                    return False
                any_ok = any_ok or out["status"] in _CONFIRMED
        return any_ok

    faults: list[str] = []
    for path_name, path in paths.items():
        if path.get("error"):
            faults.append(f"{path_name}: {path['status']}: {path['error']}")
        for out_name, out in path["outputs"].items():
            if out["status"] in (FAILED, UNCONFIRMED):
                msg = out.get("error") or out.get("detail") or ""
                faults.append(f"{path_name}.{out_name}: {out['status']}: {msg}".rstrip(": "))
    motors = confirmed(_DRIVE_OUTPUTS)
    blade = confirmed(_BLADE_OUTPUTS)
    any_failed = any(
        o["status"] == FAILED for p in paths.values() for o in p["outputs"].values()
    ) or any(p["status"] == FAILED and not p["outputs"] for p in paths.values())
    return {
        "motors_stopped": motors,
        "blade_disabled": blade,
        "complete": motors and blade and not any_failed,
        "subsystems": paths,
        "faults": faults,
    }


async def _run_all(kind: str, source: str, path_timeout_s: float) -> dict[str, Any]:
    track = kind == "stop"
    # Every path is scheduled in THIS loop turn, before the first await. Handing the coroutines to
    # ``gather`` instead wrapped them in tasks that only start on the next turn, so a caller
    # cancelled right there (a dropped request, a lifespan teardown) cancelled them before they
    # ran: no stop path executed although the latch was already set (audit BL-08).
    started = (
        ("tractor", _start_path(lambda: _tractor(kind, source), track=track)),
        ("robohat", _start_path(lambda: _robohat(kind, source), track=track)),
        ("blade_service", _start_path(lambda: _blade_service(kind, source), track=track)),
    )
    results = await asyncio.gather(
        *(_await_path(name, task, path_timeout_s) for name, task in started)
    )
    return _summarize(dict(results))


async def execute_emergency_stop(
    source: str = "api", *, path_timeout_s: float = PATH_TIMEOUT_S
) -> dict[str, Any]:
    """Latch the general E-stop and drive every stop path that exists to safe.

    Sets the process-wide latch first (``emergency_stop_active``), then returns
    ``{"emergency_stop_active": True, "motors_stopped": bool, "blade_disabled":
    bool, "complete": bool, "subsystems": {...}, "faults": [...]}``. The latch is
    always set; the booleans say what was actually confirmed. Never raises.
    """
    global _stops_in_flight, _stop_epoch
    _safety_state["emergency_stop_active"] = True
    _blade_state["active"] = False
    _stop_epoch += 1
    _stops_in_flight += 1
    try:
        # Every autonomous run ends with the stop, in this same event-loop turn (audit SAF-01):
        # a clear of the latch below must never re-activate a mission or an AI loop. The latch
        # is queued for the disk before the slow fan-out so a restart cannot release it
        # (audit SAF-16).
        stop_events.record_stop("emergency_stop", source)
        latch_store.request_persist()
        report = await _run_all("stop", source, path_timeout_s)
    finally:
        _stops_in_flight -= 1
    # The final state (the RoboHAT hold the stop has just set) is what a restart must restore.
    saved = await latch_store.persist()
    report["emergency_stop_active"] = True
    report["latch_persisted"] = saved.ok
    report["latch_error"] = saved.error
    if report["complete"]:
        logger.warning("Emergency stop (%s): all outputs confirmed", source)
    else:
        logger.critical("Emergency stop (%s) NOT fully confirmed: %s", source, report["faults"])
    return report


async def park_actuators(
    source: str = "shutdown", *, path_timeout_s: float = PATH_TIMEOUT_S
) -> dict[str, Any]:
    """Bring every actuator to its resting state without latching an E-stop.

    Same fan-out and report shape as :func:`execute_emergency_stop` (minus the
    latch): blade off, levers neutral, throttle idle. Never raises.
    """
    report = await _run_all("park", source, path_timeout_s)
    if report["complete"]:
        logger.info("Actuators parked (%s)", source)
    else:
        logger.critical("Actuator park (%s) NOT fully confirmed: %s", source, report["faults"])
    return report


__all__ = [
    "execute_emergency_stop",
    "park_actuators",
    "stop_epoch",
    "stop_in_flight",
    "PATH_TIMEOUT_S",
]
