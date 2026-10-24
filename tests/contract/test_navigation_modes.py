"""Contract test (T056): navigation mode transitions.

FR-032: the robot supports MANUAL <-> AUTONOMOUS <-> EMERGENCY_STOP. Exercises
``POST /api/v2/nav/mode`` and the three places the mode is read back:
``GET /api/v2/nav/status``, ``GET /api/v2/status`` and the ``RobotState`` itself
(``backend/src/api/navigation.py``, ``backend/src/api/status.py``).

Scope: this is the ``RobotState.navigation_mode`` enum (IDLE / MANUAL / AUTONOMOUS /
EMERGENCY_STOP). It is a different enum from the one ``NavigationService`` uses for
missions (AUTO / AI / PAUSED ..., audit N-31), and ``POST /api/v2/control/emergency`` does not
write it, so the E-stop route is not asserted here (``tests/unit/test_emergency_stop_service.py``
covers the E-stop fan-out).
"""

from __future__ import annotations

import pytest

from backend.src.models.robot_state import NavigationMode


async def _read_modes(client) -> tuple[str, str]:
    """Return the mode as served by /nav/status and by /status."""
    nav = await client.get("/api/v2/nav/status")
    overview = await client.get("/api/v2/status")
    assert nav.status_code == 200
    assert overview.status_code == 200
    return nav.json()["mode"], overview.json()["navigation_state"]


@pytest.mark.asyncio
async def test_navigation_mode_transitions(api_client, nav_state):
    assert await _read_modes(api_client) == ("IDLE", "IDLE")
    assert nav_state.navigation_mode is NavigationMode.IDLE

    # Every hop of MANUAL <-> AUTONOMOUS <-> EMERGENCY_STOP, in both directions where the
    # contract names them; each one must be accepted and then read back everywhere.
    path = [
        "MANUAL",
        "AUTONOMOUS",
        "MANUAL",
        "AUTONOMOUS",
        "EMERGENCY_STOP",
        "MANUAL",
        "EMERGENCY_STOP",
        "MANUAL",
    ]
    for mode in path:
        resp = await api_client.post("/api/v2/nav/mode", json={"mode": mode})
        assert resp.status_code == 200, mode
        assert resp.json() == {"ok": True, "mode": mode}
        assert await _read_modes(api_client) == (mode, mode)
        assert nav_state.navigation_mode == mode


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_mode", ["BOGUS", "AUTO", "manual", ""])
async def test_unknown_mode_is_rejected_and_leaves_the_mode_alone(api_client, nav_state, bad_mode):
    ok = await api_client.post("/api/v2/nav/mode", json={"mode": "AUTONOMOUS"})
    assert ok.status_code == 200

    resp = await api_client.post("/api/v2/nav/mode", json={"mode": bad_mode})
    assert resp.status_code == 422
    assert "mode" in str(resp.json()["detail"])

    # "AUTO" belongs to the mission enum, not this one: it must not be coerced either.
    assert await _read_modes(api_client) == ("AUTONOMOUS", "AUTONOMOUS")
