"""A NaN model output stands the platform down; it is not dropped as "low confidence" (NAV-07).

``ActionPrediction`` turns a non-finite control output or confidence into "neutral, zero confidence,
blade off, overridden". On paper that is a stop. In ``apply_ai_prediction`` the confidence gate
came first, dropped the zero-confidence prediction and returned before anything was commanded: the
last tick's command, and the PTO a normal prediction had engaged, stayed in force for as long as
the model emitted NaN (the levers came back only through the command deadman), and the run stayed
"running".

The tractor is the real service over the stallable PCA9685, the differential platform the real
RoboHAT service over a fake serial line. Nothing here touches hardware.
"""

from __future__ import annotations

import math

import pytest
from _mission_rig import driving, new_nav, robohat_platform
from _tractor_fakes import LEFT, NEUTRAL, RIGHT, mowing_tractor

from backend.src.models import NavigationMode
from backend.src.models.action_prediction import ActionPrediction
from backend.src.services.navigation_service import MAX_CONSECUTIVE_UNUSABLE_PREDICTIONS

NAN = float("nan")


def _good(blade: bool = True) -> ActionPrediction:
    return ActionPrediction(steering=0.0, throttle=0.8, blade=blade, confidence=0.9)


def _broken() -> ActionPrediction:
    return ActionPrediction(steering=NAN, throttle=NAN, blade=True, confidence=NAN)


def _ai_run(monkeypatch, tractor):
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    return nav


def test_the_prediction_says_it_is_unusable_and_a_finite_one_does_not():
    assert _broken().unusable is True
    assert ActionPrediction(steering=math.inf, throttle=0.5, blade=True, confidence=0.9).unusable
    assert _good().unusable is False
    low = ActionPrediction(steering=0.1, throttle=0.5, blade=True, confidence=0.1)
    assert low.unusable is False, "low confidence is a statement of the model, not a broken run"


async def test_a_nan_prediction_switches_the_blade_off_and_the_levers_neutral_at_once(monkeypatch):
    tractor, chip = mowing_tractor()
    nav = _ai_run(monkeypatch, tractor)
    assert await nav.apply_ai_prediction(_good()) is True
    assert tractor.state.blade_engaged and tractor.blade_pto.state and tractor.state.moving

    applied = await nav.apply_ai_prediction(_broken())

    assert applied is False, "nothing was applied"
    assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False, (
        "the PTO a normal prediction engaged stayed on under a NaN model"
    )
    assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0
    assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT], (
        "neutral is on the wire, not left to the command deadman"
    )
    assert nav.navigation_state.navigation_mode == NavigationMode.AI, "one bad tick is a hold"
    await tractor.shutdown()


async def test_the_run_goes_on_after_a_hold_when_the_model_recovers(monkeypatch):
    tractor, _chip = mowing_tractor()
    nav = _ai_run(monkeypatch, tractor)
    await nav.apply_ai_prediction(_good())

    for _ in range(MAX_CONSECUTIVE_UNUSABLE_PREDICTIONS - 1):
        await nav.apply_ai_prediction(_broken())
    assert nav.navigation_state.navigation_mode == NavigationMode.AI
    assert tractor.state.blade_engaged is False

    assert await nav.apply_ai_prediction(_good()) is True  # the model is back
    assert tractor.state.blade_engaged is True and tractor.state.moving, "the hold was released"

    # the run of bad ticks was broken by the good one: a fresh run of 4 does not end the run
    for _ in range(MAX_CONSECUTIVE_UNUSABLE_PREDICTIONS - 1):
        await nav.apply_ai_prediction(_broken())
    assert nav.navigation_state.navigation_mode == NavigationMode.AI
    await tractor.shutdown()


async def test_a_model_that_keeps_emitting_nan_ends_the_run(monkeypatch):
    tractor, chip = mowing_tractor()
    nav = _ai_run(monkeypatch, tractor)
    await nav.apply_ai_prediction(_good())

    for _ in range(MAX_CONSECUTIVE_UNUSABLE_PREDICTIONS):
        await nav.apply_ai_prediction(_broken())

    assert nav.navigation_state.navigation_mode == NavigationMode.IDLE, "the run is over"
    assert nav.navigation_state.target_velocity == 0.0
    assert tractor.state.blade_engaged is False
    assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
    assert tractor.state.emergency_stop_active is False, "a broken model is not an E-stop latch"
    # an ended run applies nothing more
    assert await nav.apply_ai_prediction(_good()) is False
    assert tractor.state.blade_engaged is False
    await tractor.shutdown()


async def test_a_nan_prediction_on_the_differential_platform_sends_blade_off_before_neutral(
    monkeypatch,
):
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = _ai_run(monkeypatch, tractor)
    assert await nav.apply_ai_prediction(_good()) is True
    assert driving(fw.sent), "a normal prediction drives the platform"
    fw.sent.clear()

    assert await nav.apply_ai_prediction(_broken()) is False

    assert "blade=off" in fw.sent and "pwm,1500,1500" in fw.sent
    assert fw.sent.index("blade=off") < fw.sent.index("pwm,1500,1500"), "blade first, then neutral"
    await svc.shutdown()


@pytest.mark.parametrize("confidence", [0.0, 0.2, 0.49, 0.9])
async def test_a_safety_override_stops_the_machine_whatever_the_models_confidence(
    monkeypatch, confidence
):
    tractor, chip = mowing_tractor()
    nav = _ai_run(monkeypatch, tractor)
    await nav.apply_ai_prediction(_good())
    stop = ActionPrediction(
        steering=0.0, throttle=0.0, blade=False, confidence=confidence, safety_override=True
    )

    assert await nav.apply_ai_prediction(stop) is True  # a low-confidence tick used to drop it

    assert tractor.state.emergency_stop_active is True
    assert tractor.state.blade_engaged is False and chip.regs == NEUTRAL
    await tractor.shutdown()
