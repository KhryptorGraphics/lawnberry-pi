"""Status API exposing robot state over REST and WebSocket.

Implements T024 minimal contract: GET /api/v2/status returns current state
and a WebSocket at /api/v2/ws/status streams updates at ~5Hz.
"""

from __future__ import annotations
# ruff: noqa: I001

import asyncio
import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .routers.auth import admit_websocket
from .rest import websocket_hub
from ..core.robot_state_manager import get_robot_state_manager
from ..core.ws_policy import SEND_TIMEOUT_S, ws_registry
from ..models.safety_interlock import InterlockState, InterlockType


router = APIRouter()


@router.get("/api/v2/status")
async def get_status_v2():
    """Return current robot state in a simplified schema for contract tests."""
    mgr = get_robot_state_manager()
    st = mgr.get_state()
    # The named flags must agree with the interlock list below: an interlock is "detected"
    # while ACTIVE (a cleared-pending-ack one is still listed, but no longer detected).
    detected = {i.interlock_type for i in st.active_interlocks if i.state == InterlockState.ACTIVE}
    return {
        "battery_percentage": st.battery.percentage,
        "navigation_state": str(st.navigation_mode),
        "safety_status": {
            "emergency_stop_active": False,
            "tilt_detected": InterlockType.TILT_DETECTED in detected,
            "obstacle_detected": bool(
                detected & {InterlockType.OBSTACLE_DETECTED, InterlockType.ULTRASONIC_OBSTACLE}
            ),
            "blade_safety_ok": True,
            "safety_interlocks": [
                (i.interlock_type.value if hasattr(i, "interlock_type") else str(i))
                for i in st.active_interlocks
            ],
        },
        "motor_status": "idle",
        "last_updated": st.last_updated.isoformat(),
    }


STATUS_CADENCE_HZ = 5.0


@router.websocket("/api/v2/ws/status")
async def ws_status(websocket: WebSocket):
    """Stream robot state at approximately 5Hz (read-only; see ``admit_websocket``).

    Every socket reads the hub's shared telemetry sample (``WebSocketHub.current_telemetry``):
    it used to run its own full sensor read per tick, so a dozen sockets multiplied the load on
    the event loop that also carries the E-stop handler (E-stop POST 4 ms -> 560 ms). A client
    that stops reading is dropped after ``SEND_TIMEOUT_S`` instead of parking a task for ever.
    """
    admission = await admit_websocket(websocket)
    await websocket.accept()
    mgr = get_robot_state_manager()
    last_sample: dict | None = None
    try:
        while True:
            telemetry = await websocket_hub.current_telemetry(max_age_s=1.0 / STATUS_CADENCE_HZ)
            if telemetry is not last_sample:  # one state update per new sample, not per socket
                mgr.update_from_telemetry(telemetry)
                last_sample = telemetry
            st = mgr.get_state()
            async with asyncio.timeout(SEND_TIMEOUT_S):
                await websocket.send_json(
                    {
                        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                        "battery_percentage": st.battery.percentage,
                        "navigation_state": str(st.navigation_mode),
                        "position": st.position.model_dump(),
                    }
                )
            await asyncio.sleep(1.0 / STATUS_CADENCE_HZ)
    except (WebSocketDisconnect, TimeoutError, RuntimeError, OSError):
        return
    finally:
        ws_registry.release(admission.slot)
