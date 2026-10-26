"""Contract test (T054): geofence enforcement through the navigation debug API.

FR-030: when the reported position is outside the active geofence the robot must go to
``EMERGENCY_STOP`` and ``/api/v2/nav/status`` must report ``geofence.inside == false``.

Exercises ``POST /api/v2/debug/geofence`` + ``POST /api/v2/debug/gps/inject`` +
``GET /api/v2/nav/status`` (``backend/src/api/navigation.py``), including the
ray-casting test in ``backend/src/nav/geoutils.py``. What it does *not* prove is any
actuator effect: a breach sets ``RobotState.navigation_mode`` only.
"""

from __future__ import annotations

import pytest

# Right triangle: A(top-left) - B(top-right) - C(bottom-right); the hypotenuse is A-C.
FENCE = {
    "geofence_id": "test",
    "boundary": [
        {"latitude": 37.422, "longitude": -122.084},  # A
        {"latitude": 37.422, "longitude": -122.083},  # B
        {"latitude": 37.421, "longitude": -122.083},  # C
    ],
}
INSIDE = {"latitude": 37.4218, "longitude": -122.0833, "accuracy_m": 3.0}
OUTSIDE = {"latitude": 37.420, "longitude": -122.082, "accuracy_m": 3.0}


async def _set_fence(client) -> None:
    resp = await client.post("/api/v2/debug/geofence", json=FENCE)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def _inject(client, position: dict) -> None:
    resp = await client.post("/api/v2/debug/gps/inject", json=position)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def _set_mode(client, mode: str) -> None:
    resp = await client.post("/api/v2/nav/mode", json={"mode": mode})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "mode": mode}


async def _nav_status(client) -> dict:
    resp = await client.get("/api/v2/nav/status")
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_position_outside_boundary_triggers_stop(api_client, nav_state):
    await _set_fence(api_client)
    await _set_mode(api_client, "AUTONOMOUS")

    # Control: a fix inside the fence leaves the robot running.
    await _inject(api_client, INSIDE)
    status = await _nav_status(api_client)
    assert status["geofence"] == {"active": True, "inside": True}
    assert status["mode"] == "AUTONOMOUS"

    # A fix outside the fence stops it, and the position shown is the offending fix.
    await _inject(api_client, OUTSIDE)
    status = await _nav_status(api_client)
    assert status["geofence"] == {"active": True, "inside": False}
    assert status["mode"] == "EMERGENCY_STOP"
    assert status["position"] == {
        "latitude": OUTSIDE["latitude"],
        "longitude": OUTSIDE["longitude"],
        "accuracy_m": OUTSIDE["accuracy_m"],
    }

    # /api/v2/status serves the same RobotState; it must agree.
    overview = await api_client.get("/api/v2/status")
    assert overview.status_code == 200
    assert overview.json()["navigation_state"] == "EMERGENCY_STOP"


@pytest.mark.asyncio
async def test_geofence_breach_stays_latched_when_position_returns_inside(api_client, nav_state):
    await _set_fence(api_client)
    await _set_mode(api_client, "AUTONOMOUS")
    await _inject(api_client, OUTSIDE)
    assert (await _nav_status(api_client))["mode"] == "EMERGENCY_STOP"

    # Re-entering the fence must not silently resume autonomous operation.
    await _inject(api_client, INSIDE)
    status = await _nav_status(api_client)
    assert status["geofence"] == {"active": True, "inside": True}
    assert status["mode"] == "EMERGENCY_STOP"


@pytest.mark.asyncio
async def test_fence_activated_while_already_outside_stops_immediately(api_client, nav_state):
    await _set_mode(api_client, "AUTONOMOUS")
    await _inject(api_client, OUTSIDE)

    # No fence yet: nothing to enforce, and no verdict is reported.
    status = await _nav_status(api_client)
    assert status["geofence"] == {"active": False, "inside": None}
    assert status["mode"] == "AUTONOMOUS"

    # Activating the fence is evaluated at once, without waiting for the next GPS fix.
    await _set_fence(api_client)
    status = await _nav_status(api_client)
    assert status["geofence"] == {"active": True, "inside": False}
    assert status["mode"] == "EMERGENCY_STOP"


@pytest.mark.parametrize(
    ("label", "latitude", "longitude", "inside"),
    [
        ("vertex A counts as inside", 37.422, -122.084, True),
        ("midpoint of an edge counts as inside", 37.4215, -122.0835, True),
        ("interior point", 37.4216, -122.0832, True),
        ("about 1 m beyond the right edge", 37.4215, -122.08299, False),
        ("on the far side of the hypotenuse", 37.421, -122.084, False),
        ("just north of the top edge", 37.42201, -122.0835, False),
    ],
)
@pytest.mark.asyncio
async def test_geofence_boundary_classification(
    api_client, nav_state, label, latitude, longitude, inside
):
    await _set_fence(api_client)
    await _set_mode(api_client, "AUTONOMOUS")
    await _inject(api_client, {"latitude": latitude, "longitude": longitude, "accuracy_m": 1.0})

    status = await _nav_status(api_client)
    assert status["geofence"]["inside"] is inside, label
    assert status["mode"] == ("AUTONOMOUS" if inside else "EMERGENCY_STOP"), label
