"""One fail-closed collector for the operator-facing safety block.

Every status surface (``/api/v2/status``, ``/api/v2/ws/status``, ``/api/v2/dashboard/status``,
``/api/v2/dashboard/metrics`` and ``/api/v1/status``) reports the same ``safety_status`` object
built here from the real safety state, never from literals:

* the process-wide E-stop latch (``core.globals._safety_state``);
* the tractor service (E-stop latch, blade-interlock latch, lock-out, park);
* the RoboHAT drive controller's software E-stop hold;
* the IBT-4 blade service's E-stop latch;
* the navigation motion gate (obstacle guard and GNSS monitor holds);
* the robot-state interlock list (tilt, obstacle, ...).

Each source is read independently inside its own ``try``: a source that cannot be read is
reported as the interlock ``state_unreadable:<source>`` and the blade path is then reported as
not ok. The collector never raises and never touches hardware: every accessor it calls is a
process singleton whose construction opens no bus, and every attribute it reads is a plain
in-memory flag, so it is cheap enough for 5 Hz per WebSocket client.

Interlock vocabulary (``safety_interlocks``, stable order, no duplicates): the robot-state
``InterlockType`` values first (``tilt_detected``, ``obstacle_detected``, ...), then
``emergency_stop``, ``tractor_emergency_stop``, ``blade_interlock:<reason>``,
``tractor_lockout``, ``tractor_parked``, ``robohat_emergency_hold``, ``blade_service_latched``,
``obstacle_hold``, ``gnss_hold`` and ``state_unreadable:<source>``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Obstacle-guard reasons that are guard faults rather than a sighted obstacle.
_NOT_A_SIGHTING = ("no obstacle sources configured", "obstacle guard failed")


def _unreadable(source: str, interlocks: list[str]) -> None:
    logger.warning("Safety status: %s unreadable; reporting fail-closed", source, exc_info=True)
    interlocks.append(f"state_unreadable:{source}")


def collect_safety_status() -> dict[str, Any]:
    """The current safety block: the five legacy keys, from the real safety state."""
    interlocks: list[str] = []
    estop = False
    tilt = False
    obstacle = False
    blade_inhibited = False

    # --- robot-state interlocks (the pre-existing tilt/obstacle mapping, kept exactly) ---
    try:
        from ..core.robot_state_manager import get_robot_state_manager
        from ..models.safety_interlock import InterlockState, InterlockType

        active = list(get_robot_state_manager().get_state().active_interlocks)
        detected = {i.interlock_type for i in active if i.state == InterlockState.ACTIVE}
        tilt = InterlockType.TILT_DETECTED in detected
        obstacle = bool(
            detected & {InterlockType.OBSTACLE_DETECTED, InterlockType.ULTRASONIC_OBSTACLE}
        )
        estop = InterlockType.EMERGENCY_STOP in detected
        blade_inhibited = bool(detected)
        interlocks.extend(
            i.interlock_type.value if hasattr(i, "interlock_type") else str(i) for i in active
        )
    except Exception:
        _unreadable("robot_state", interlocks)
        blade_inhibited = True

    # --- process-wide E-stop latch ---
    try:
        from ..core.globals import _safety_state

        if bool(_safety_state.get("emergency_stop_active", False)):
            estop = True
            interlocks.append("emergency_stop")
    except Exception:
        _unreadable("process_latch", interlocks)
        blade_inhibited = True

    # --- tractor service ---
    try:
        from ..services.tractor_service import get_tractor_service

        tractor = get_tractor_service()
        enabled = bool(tractor.enabled)
        lockout = tractor.lockout_reason
        parked = bool(tractor.parked)
        state = tractor.state
        if enabled or lockout is not None:
            if state.emergency_stop_active:
                estop = True
                interlocks.append("tractor_emergency_stop")
            if state.interlock_latched:
                reason = state.latch_reason or "unknown"
                interlocks.append(f"blade_interlock:{reason}")
                tilt = tilt or reason == "tilt"
                blade_inhibited = True
        if lockout is not None:
            estop = True
            interlocks.append("tractor_lockout")
        if parked:
            interlocks.append("tractor_parked")
            blade_inhibited = True
    except Exception:
        _unreadable("tractor", interlocks)
        blade_inhibited = True

    # --- RoboHAT drive controller (None until the lifespan builds it) ---
    try:
        from ..services.robohat_service import get_robohat_service

        robohat = get_robohat_service()
        if robohat is not None and bool(robohat.emergency_latched):
            estop = True
            interlocks.append("robohat_emergency_hold")
    except Exception:
        _unreadable("robohat", interlocks)
        blade_inhibited = True

    # --- IBT-4 blade service ---
    try:
        from ..services.blade_service import get_blade_service

        if bool(get_blade_service().emergency_latched):
            estop = True
            interlocks.append("blade_service_latched")
    except Exception:
        _unreadable("blade_service", interlocks)
        blade_inhibited = True

    # --- navigation motion gate: obstacle guard and GNSS monitor holds ---
    try:
        from ..services.navigation_service import NavigationService

        gate = NavigationService.get_instance().motion_gate()
        if gate.hold:
            reasons = [str(r) for r in gate.reasons]
            gnss = [r for r in reasons if r.startswith("gnss")]
            other = [r for r in reasons if not r.startswith("gnss")]
            if gnss:
                interlocks.append("gnss_hold")
            if other or not gnss:
                interlocks.append("obstacle_hold")
                obstacle = obstacle or any(
                    "obstacle" in r.lower() and not r.lower().startswith(_NOT_A_SIGHTING)
                    for r in other
                )
            blade_inhibited = True
    except Exception:
        _unreadable("navigation_gate", interlocks)
        blade_inhibited = True

    blade_inhibited = blade_inhibited or estop or bool(interlocks)
    return {
        "emergency_stop_active": estop,
        "tilt_detected": tilt,
        "obstacle_detected": obstacle,
        "blade_safety_ok": not blade_inhibited,
        "safety_interlocks": list(dict.fromkeys(interlocks)),
    }


def _tractor_blade_engaged() -> bool:
    from ..services.tractor_service import get_tractor_service

    tractor = get_tractor_service()
    return tractor is not None and bool(tractor.state.blade_engaged)


def _ibt4_blade_may_be_on() -> bool:
    from ..services.blade_service import get_blade_service

    return bool(get_blade_service().blade_may_be_on())


def _robohat_blade_commanded_on() -> bool:
    from ..services.robohat_service import get_robohat_service

    robohat = get_robohat_service()  # None until the lifespan builds it
    return robohat is not None and bool(robohat.blade_commanded_on)


def collect_blade_active() -> bool:
    """Is a blade (possibly) running? Conservative: anything that cannot say "off" counts as on.

    Three blades, each read on its own: the tractor's PTO relay (``blade_engaged``), the IBT-4
    (the two pads READ now, not the command cache: a pad forced high, a failed readback, lines
    that were never claimed and an unconfirmed write all read "may be on") and the RoboHAT's
    blade, the deployed differential platform's (the last ``blade=`` line written to the
    firmware). A source that cannot be read at all counts as "may be on" too (audit HW-04).
    """
    for name, source in (
        ("tractor", _tractor_blade_engaged),
        ("ibt4", _ibt4_blade_may_be_on),
        ("robohat", _robohat_blade_commanded_on),
    ):
        try:
            if source():
                return True
        except Exception:
            logger.warning(
                "blade_active: %s unreadable; reporting the blade as possibly on",
                name,
                exc_info=True,
            )
            return True
    return False
