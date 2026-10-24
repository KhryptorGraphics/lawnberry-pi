"""Contract test (T055): waypoint arrival detection through the navigation debug API.

FR-031: given a waypoint and simulated position updates, ``/api/v2/nav/status`` reports
the distance to the waypoint and whether it has been reached, using the *waypoint's own*
``arrival_threshold_m``. Exercises ``POST /api/v2/nav/waypoints``,
``POST /api/v2/debug/gps/inject`` and ``GET /api/v2/nav/status``
(``backend/src/api/navigation.py``, haversine in ``backend/src/nav/geoutils.py``).

One degree of latitude is 111.195 km on the 6 371 km sphere ``haversine_m`` uses, so the
offsets below are 1e-5 deg = 1.11 m, 2.5e-5 deg = 2.78 m and 5e-5 deg = 5.56 m.
"""

from __future__ import annotations

import pytest

WP_LAT = 37.4220
WP_LON = -122.0841
M_PER_1E5_DEG = 1.1119  # metres per 1e-5 degree of latitude


def _waypoint(threshold_m: float, waypoint_id: str = "wp1") -> dict:
    return {
        "waypoints": [
            {
                "waypoint_id": waypoint_id,
                "latitude": WP_LAT,
                "longitude": WP_LON,
                "arrival_threshold_m": threshold_m,
            }
        ]
    }


async def _set_waypoint(client, threshold_m: float) -> None:
    resp = await client.post("/api/v2/nav/waypoints", json=_waypoint(threshold_m))
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "count": 1}


async def _fix_at_offset(client, north_1e5_deg: float) -> dict:
    """Inject a fix ``north_1e5_deg`` x 1e-5 degrees north of the waypoint; return its status."""
    fix = {
        "latitude": WP_LAT + north_1e5_deg * 1e-5,
        "longitude": WP_LON,
        "accuracy_m": 0.5,
    }
    resp = await client.post("/api/v2/debug/gps/inject", json=fix)
    assert resp.status_code == 200
    status = await client.get("/api/v2/nav/status")
    assert status.status_code == 200
    return status.json()["waypoint"]


@pytest.mark.asyncio
async def test_waypoint_arrival_detection(api_client, nav_state):
    await _set_waypoint(api_client, threshold_m=2.0)

    # The placeholder's case: a fix exactly on the waypoint.
    on_waypoint = await _fix_at_offset(api_client, 0.0)
    assert on_waypoint["reached"] is True
    assert on_waypoint["current_id"] == "wp1"
    assert on_waypoint["queue_len"] == 1
    assert on_waypoint["distance_m"] == pytest.approx(0.0, abs=1e-6)

    # 1.11 m away: inside the 2.0 m threshold.
    near = await _fix_at_offset(api_client, 1.0)
    assert near["reached"] is True
    assert near["distance_m"] == pytest.approx(1 * M_PER_1E5_DEG, abs=0.01)

    # 2.78 m away: outside it. Arrival is recomputed from every fix, not latched.
    close_but_not_there = await _fix_at_offset(api_client, 2.5)
    assert close_but_not_there["reached"] is False
    assert close_but_not_there["distance_m"] == pytest.approx(2.5 * M_PER_1E5_DEG, abs=0.01)

    far = await _fix_at_offset(api_client, 5.0)
    assert far["reached"] is False
    assert far["current_id"] == "wp1"
    assert far["distance_m"] == pytest.approx(5 * M_PER_1E5_DEG, abs=0.01)


@pytest.mark.asyncio
async def test_arrival_uses_the_waypoints_own_threshold(api_client, nav_state):
    # The same 2.78 m offset is "not there" for a 2 m waypoint and "arrived" for a 4 m one.
    await _set_waypoint(api_client, threshold_m=2.0)
    assert (await _fix_at_offset(api_client, 2.5))["reached"] is False

    await _set_waypoint(api_client, threshold_m=4.0)
    assert (await _fix_at_offset(api_client, 2.5))["reached"] is True


@pytest.mark.asyncio
async def test_no_arrival_is_reported_before_any_fix(api_client, nav_state):
    await _set_waypoint(api_client, threshold_m=2.0)

    waypoint = (await api_client.get("/api/v2/nav/status")).json()["waypoint"]
    assert not waypoint["reached"]  # False or unknown, but never a claimed arrival
    assert waypoint["distance_m"] is None
    assert waypoint["queue_len"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold_m", [0.0, 0.05, 10.5, -1.0])
async def test_out_of_range_arrival_threshold_is_rejected(api_client, nav_state, threshold_m):
    resp = await api_client.post("/api/v2/nav/waypoints", json=_waypoint(threshold_m))
    assert resp.status_code == 422
    assert "arrival_threshold_m" in str(resp.json()["detail"])

    # The rejected request must not have queued anything.
    status = (await api_client.get("/api/v2/nav/status")).json()
    assert status["waypoint"]["queue_len"] == 0
