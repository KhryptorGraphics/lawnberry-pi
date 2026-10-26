"""Contract test (T042): ``GET /api/v2/fusion/state`` serves a fused pose with honest quality.

FR-025: the fused position is within 1 m (``quality == "good"`` means a position standard
deviation under 1 m, see ``SimpleEKF.get_state``) and the heading is continuous. The
filter maths itself is covered by ``tests/unit/test_ekf.py``; this file covers what the
route adds (``backend/src/api/fusion.py``):

- in SIM_MODE the route feeds the filter a synthetic slow turn and, on every fifth wall-clock
  second, a GPS fix at (0, 0) and an IMU heading of 90 deg. The tests pin that clock so
  they never depend on timing luck;
- outside SIM_MODE nothing feeds the filter, and the route must say so (quality "poor");
- the reported quality is downgraded when the mean sensor-health score is poor.

There are no endpoints to inject IMU or odometry data and ``/api/v2/debug/gps/inject``
does not reach the filter, so a test cannot choose the measurements; the obstacle-fusion
contract of audit work package WP-P2-06 does not exist yet and is not covered here.
"""

from __future__ import annotations

import pytest

# int(t) % 5 == 0 makes the SIM branch of the route fuse GPS (0, 0) and heading 90.
FIX_SECOND = 1_000_005.0
NO_FIX_SECOND = 1_000_001.0


class _PinnedClock:
    """Stands in for the ``time`` module inside ``backend.src.api.fusion`` only."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def fusion_filter(monkeypatch, sim_env):
    """A fresh EKF + sensor-health monitor behind the route, with the route's clock pinned."""
    from backend.src.api import fusion as fusion_api
    from backend.src.fusion.ekf import SimpleEKF
    from backend.src.fusion.sensor_health import SensorHealthMonitor

    clock = _PinnedClock(NO_FIX_SECOND)
    monkeypatch.setattr(fusion_api, "_ekf", SimpleEKF())
    monkeypatch.setattr(fusion_api, "_health", SensorHealthMonitor())
    monkeypatch.setattr(fusion_api, "time", clock)
    return clock, fusion_api


async def _fused_state(client) -> dict:
    resp = await client.get("/api/v2/fusion/state")
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_sensor_fusion_outputs_fused_position_with_accuracy(api_client, fusion_filter):
    clock, _ = fusion_filter

    # Dead reckoning alone (no GPS or IMU fix yet) must not be reported as a good estimate.
    unfixed = await _fused_state(api_client)
    assert unfixed["quality"] == "poor"
    assert unfixed["uncertainty"]["position_std_m"] > 5.0
    assert unfixed["sources"] == ["odometry"]

    # Once GPS and IMU fixes arrive the estimate converges to them.
    clock.now = FIX_SECOND
    headings = []
    for _ in range(4):
        fused = await _fused_state(api_client)
        headings.append(fused["heading"])

    assert fused["quality"] == "good"
    assert fused["uncertainty"]["position_std_m"] < 1.0  # FR-025: within 1 m
    assert fused["position"]["x"] == pytest.approx(0.0, abs=0.1)
    assert fused["position"]["y"] == pytest.approx(0.0, abs=0.1)
    assert fused["heading"] == pytest.approx(90.0, abs=1.0)
    assert fused["sources"] == ["gps", "imu", "odometry"]

    # Heading continuity: after the first correction it moves by well under a degree per update.
    steps = [abs(b - a) for a, b in zip(headings[1:], headings[2:], strict=False)]
    assert steps and max(steps) < 1.0

    # Uncertainty keeps shrinking as fixes accumulate.
    after_one_more = await _fused_state(api_client)
    assert after_one_more["uncertainty"]["position_std_m"] < fused["uncertainty"]["position_std_m"]


@pytest.mark.asyncio
async def test_poor_sensor_health_downgrades_the_fused_quality(api_client, fusion_filter):
    clock, fusion_api = fusion_filter
    clock.now = FIX_SECOND
    for _ in range(3):
        fused = await _fused_state(api_client)
    assert fused["quality"] == "good"

    # The filter is still confident, but a failing GPS drags the mean sensor score <= 0.7.
    fusion_api._health.set_quality("gps", 0.2)
    degraded = await _fused_state(api_client)
    assert degraded["uncertainty"]["position_std_m"] < 1.0
    assert degraded["quality"] == "degraded"
    assert degraded["sensor_quality"]["gps"] == 0.2

    # Recovery is reflected too.
    fusion_api._health.set_quality("gps", 0.9)
    assert (await _fused_state(api_client))["quality"] == "good"


@pytest.mark.asyncio
async def test_unfed_filter_outside_sim_mode_reports_poor_quality(
    api_client, fusion_filter, monkeypatch
):
    clock, _ = fusion_filter
    clock.now = FIX_SECOND  # would fuse a fix if the SIM branch were taken
    monkeypatch.setenv("SIM_MODE", "0")

    for _ in range(3):
        fused = await _fused_state(api_client)

    # Nothing outside SIM_MODE feeds the filter, so there is no estimate to trust.
    assert fused["quality"] == "poor"
    assert fused["uncertainty"]["position_std_m"] > 5.0
    assert fused["uncertainty"]["heading_std_deg"] > 90.0
