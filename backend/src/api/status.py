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
from ..core.safety_status import collect_safety_status
from ..core.ws_policy import SEND_TIMEOUT_S, ws_registry


router = APIRouter()


@router.get("/api/v2/status")
async def get_status_v2():
    """Return current robot state in a simplified schema for contract tests."""
    mgr = get_robot_state_manager()
    st = mgr.get_state()
    return {
        "battery_percentage": st.battery.percentage,
        "navigation_state": str(st.navigation_mode),
        "safety_status": collect_safety_status(),
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

    The socket is bound to its login (``core/ws_policy.py``): when the session ends it is closed
    from outside with 1008, even before ``accept`` below, and the next send (or the accept)
    fails, which ends this handler through the ``except`` and hands the slot back in ``finally``
    (``accept`` is inside the ``try`` for that: a slot taken here is always given back).
    """
    admission = await admit_websocket(websocket)
    last_sample: dict | None = None
    try:
        await websocket.accept()
        mgr = get_robot_state_manager()
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
                        "safety_status": collect_safety_status(),
                        "position": st.position.model_dump(),
                    }
                )
            await asyncio.sleep(1.0 / STATUS_CADENCE_HZ)
    except (WebSocketDisconnect, TimeoutError, RuntimeError, OSError):
        return
    finally:
        ws_registry.release(admission.slot)
