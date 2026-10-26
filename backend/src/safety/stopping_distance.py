"""Speed-, latency- and sensor-uncertainty-dependent stopping-distance model (WP-P4-02).

Pure arithmetic, no I/O and no clock, so the thresholds the navigation loop and the
actuation chokepoint use can be unit-tested at their boundaries.

The model answers one question: *how far ahead must the path be known to be free for
the mower to be allowed to travel at speed ``v``?*

    stop_distance(v) = v * (t_react + t_sensor_age + t_actuate)   # moving while we notice and act
                     + v^2 / (2 * a)                              # then coasting to a halt
                     + sensor_uncertainty + margin

* ``t_react``   - compute and command-issue latency (``reaction_latency_s``)
* ``t_sensor_age`` - how old the range reading already is when it is used. It is *not* a
  parameter: callers pass the measured age as ``extra_latency_s`` so an old reading
  automatically costs speed instead of being trusted at face value.
* ``t_actuate`` - command issued until the levers are physically neutral
  (``actuation_latency_s``; the constitution budgets 100 ms to issue and 500 ms to settle,
  ``docs/tractor-acceptance-criteria.md`` section 2).
* ``a`` - deceleration once neutral (``deceleration_mps2``). A hydrostatic drive brakes by
  itself only, and the repository's field protocol warns the machine creeps after neutral.
* ``sensor_uncertainty`` and ``margin`` - range error, mount offset and a safety allowance.

``slow_distance(v) = stop_distance(v) + slow_zone``: the band in which speed is already being
tapered. The inverse, :meth:`StoppingDistanceModel.max_speed_for_range`, is what the
navigation loop and the tractor service apply as a speed cap.

**EVERY DEFAULT IS UNVALIDATED.** They are conservative engineering guesses so the stop path
exists and fails safe; none is a measurement. They are replaced only from the measured
stopping distance at several speeds and slopes (audit M-43) and a reviewer's signature, and
``validated`` is flipped by that same change, never by a code edit. Slope is not modelled
separately: ``deceleration_mps2`` must be the worst case (downhill, wet grass) that was
measured.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models.safety_limits import StoppingDistanceLimits


def _finite_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return max(0.0, value)


class StoppingDistanceModel:
    """Closed-form stop/slow thresholds and their inverse (see the module docstring)."""

    def __init__(self, params: StoppingDistanceLimits | None = None) -> None:
        self.params = params or StoppingDistanceLimits()

    @property
    def validated(self) -> bool:
        """False until the parameters come from measurement (audit M-43)."""
        return self.params.validated

    @property
    def full_lever_speed_mps(self) -> float:
        return self.params.full_lever_speed_mps

    # ------------------------------------------------------------------ thresholds

    def stop_distance(self, speed_mps: float, *, extra_latency_s: float = 0.0) -> float:
        """Free distance (m) needed to stop from ``speed_mps``; strictly increasing in speed."""
        v = _finite_non_negative("speed_mps", speed_mps)
        extra = _finite_non_negative("extra_latency_s", extra_latency_s)
        p = self.params
        latency = p.reaction_latency_s + p.actuation_latency_s + extra
        return v * latency + (v * v) / (2.0 * p.deceleration_mps2) + self._floor_m()

    def slow_distance(self, speed_mps: float, *, extra_latency_s: float = 0.0) -> float:
        """Range (m) below which speed ``speed_mps`` is no longer allowed to be held."""
        return (
            self.stop_distance(speed_mps, extra_latency_s=extra_latency_s) + self.params.slow_zone_m
        )

    def hold_distance(self, *, extra_latency_s: float = 0.0) -> float:
        """Free range (m) at which even the slowest useful speed is no longer permitted."""
        return self.slow_distance(self.params.min_motion_speed_mps, extra_latency_s=extra_latency_s)

    # --------------------------------------------------------------------- inverse

    def speed_for_stop_distance(self, distance_m: float, *, extra_latency_s: float = 0.0) -> float:
        """Largest speed (m/s) whose :meth:`stop_distance` fits in ``distance_m`` (0 if none).

        Exact inverse of :meth:`stop_distance` for ``distance_m`` above its zero-speed value.
        """
        distance = float(distance_m)
        if math.isnan(distance):
            raise ValueError("distance_m must not be NaN")
        extra = _finite_non_negative("extra_latency_s", extra_latency_s)
        avail = distance - self._floor_m()
        if avail <= 0.0:
            return 0.0
        if math.isinf(avail):
            return math.inf
        p = self.params
        a = p.deceleration_mps2
        at = a * (p.reaction_latency_s + p.actuation_latency_s + extra)
        # v = -at + sqrt(at^2 + 2*a*avail), written without the cancellation that loses the
        # small-distance end of the curve when ``at`` is large.
        return (2.0 * a * avail) / (at + math.sqrt(at * at + 2.0 * a * avail))

    def max_speed_for_range(self, free_range_m: float, *, extra_latency_s: float = 0.0) -> float:
        """Speed cap (m/s) for ``free_range_m`` of known-free path ahead.

        The slow zone is kept in reserve, so a machine that obeys the cap rides the edge of
        the slow band and is never inside the stopping envelope. Capped at the full-lever
        speed. Below ``min_motion_speed_mps`` the answer is 0.0: the machine is held rather
        than allowed to creep.
        """
        p = self.params
        v = self.speed_for_stop_distance(
            float(free_range_m) - p.slow_zone_m, extra_latency_s=extra_latency_s
        )
        v = min(v, p.full_lever_speed_mps)
        return 0.0 if v < p.min_motion_speed_mps else v

    def lever_fraction(self, speed_mps: float) -> float:
        """Lever deflection (0..1) that commands ``speed_mps`` (linear, UNVALIDATED mapping)."""
        return max(0.0, min(1.0, speed_mps / self.params.full_lever_speed_mps))

    def speed_of_lever(self, lever: float) -> float:
        """Forward ground speed (m/s) a lever deflection commands (0 for neutral/reverse)."""
        return max(0.0, float(lever)) * self.params.full_lever_speed_mps

    # --------------------------------------------------------------------- helpers

    def _floor_m(self) -> float:
        """Stop distance at zero speed: the uncertainty and the margin."""
        return self.params.sensor_uncertainty_m + self.params.margin_m

    def table(self, speeds_mps: Iterable[float], *, extra_latency_s: float = 0.0) -> list[dict]:
        """``[{speed, stop_m, slow_m}]`` rows for logs and the operator documentation."""
        return [
            {
                "speed_mps": float(v),
                "stop_m": round(self.stop_distance(v, extra_latency_s=extra_latency_s), 3),
                "slow_m": round(self.slow_distance(v, extra_latency_s=extra_latency_s), 3),
            }
            for v in speeds_mps
        ]


__all__ = ["StoppingDistanceModel"]
