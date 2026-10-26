"""A NaN or infinite model output must never become a drive command (audit round 3, new finding).

``ActionPrediction`` clamped its fields with ``max(lo, min(hi, x))``. ``min(1.0, nan)`` is ``1.0``,
so a model run that produced NaN steered full right at full throttle with confidence 1.0 -- the
loudest possible "go" from the quietest possible failure.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SIM_MODE", "1")

from backend.src.models.action_prediction import ActionConfidence, ActionPrediction

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def _is_neutral(p: ActionPrediction) -> bool:
    return (
        p.steering == 0.0
        and p.throttle == 0.0
        and p.confidence == 0.0
        and p.blade is False
        and p.safety_override is True
        and p.confidence_level is ActionConfidence.UNCERTAIN
    )


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("field", ["steering", "throttle", "confidence"])
def test_a_non_finite_field_makes_the_whole_prediction_neutral(field, bad):
    values = {"steering": 0.4, "throttle": 0.8, "confidence": 0.95}
    values[field] = bad

    p = ActionPrediction(blade=True, **values)

    assert _is_neutral(p), p
    assert p.to_motor_commands() == {"left_speed": 0.0, "right_speed": 0.0, "blade_enabled": False}


def test_finite_values_are_still_clamped_into_range_and_not_overridden():
    p = ActionPrediction(steering=-3.0, throttle=7.0, blade=True, confidence=2.0)

    assert (p.steering, p.throttle, p.confidence) == (-1.0, 1.0, 1.0)
    assert p.blade is True and p.safety_override is False

    low = ActionPrediction(steering=0.2, throttle=-1.0, blade=False, confidence=-0.5)
    assert (low.throttle, low.confidence) == (0.0, 0.0)


def test_numpy_scalars_are_checked_too():
    p = ActionPrediction(
        steering=np.float32("nan"), throttle=np.float32(0.9), blade=True, confidence=np.float32(1)
    )
    assert _is_neutral(p)
