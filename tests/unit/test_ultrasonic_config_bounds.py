"""``ultrasonic:`` limits in ``hardware.yaml`` against what the MCU firmware accepts, and the rest.

The host configuration may only offer the MCU what it will accept: a window or a gap the firmware
NAKs would make every handshake end in ``config_rejected`` (fail closed, but no obstacle sensing at
all), and one the host forbids that the firmware allows would only shrink what M-40 can measure.
Every bound here is the edge of a range, so each is tested on both sides.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as host
from backend.src.models.hardware_config import UltrasonicConfig
from tests.unit._usonic_fakes import CORE
from tests.unit._usonic_fw_fakes import RecordingHal


def accepted(**cfg) -> bool:
    try:
        UltrasonicConfig(**cfg)
    except ValidationError:
        return False
    return True


def mcu_accepts(window_ms: int, gap_ms: int) -> bool:
    """Does the firmware core apply this CONFIG (CONFIG_ACK) or refuse it (NAK)?"""
    hal = RecordingHal(n=4, start_ms=1000)
    core = CORE.Core(hal, 4, info_period_ms=100_000)
    core.start()
    config = host.McuConfig(
        enable_mask=0b1111, echo_window_ms=window_ms, gap_ms=gap_ms, quiet_after=8
    )
    hal.queue_rx(host.encode_frame(host.FrameType.CONFIG, 0, host.build_config(config)))
    for _ in range(5):
        core.poll()
    acks = hal.messages(host.FrameType.CONFIG_ACK)
    naks = hal.messages(host.FrameType.NAK)
    assert len(acks) + len(naks) == 1
    return bool(acks)


# Everything else is chosen so the cross-field rules (the echo of max_range_cm must fit the
# window, stale_after_ms must exceed one cycle) never decide: only the field under test does.
SHORT_RANGE = {"max_range_cm": 100.0, "stale_after_ms": 5000}


@pytest.mark.parametrize("window_ms", [9, 10, 11, 99, 100, 101])
def test_the_echo_window_the_host_allows_is_exactly_the_one_the_firmware_accepts(window_ms):
    host_ok = accepted(echo_window_ms=window_ms, **SHORT_RANGE)
    assert host_ok == (CORE.WINDOW_MIN_MS <= window_ms <= CORE.WINDOW_MAX_MS)
    assert host_ok == mcu_accepts(window_ms, gap_ms=30)


@pytest.mark.parametrize("gap_ms", [4, 5, 6, 499, 500, 501])
def test_the_gap_the_host_allows_is_exactly_the_one_the_firmware_accepts(gap_ms):
    host_ok = accepted(inter_sensor_gap_ms=gap_ms, **SHORT_RANGE)
    assert host_ok == (CORE.GAP_MIN_MS <= gap_ms <= CORE.GAP_MAX_MS)
    assert host_ok == mcu_accepts(30, gap_ms)


def test_the_firmware_limits_are_the_ones_the_host_documents():
    assert (CORE.WINDOW_MIN_MS, CORE.WINDOW_MAX_MS) == (10, 100)
    assert (CORE.GAP_MIN_MS, CORE.GAP_MAX_MS) == (5, 500)


@pytest.mark.parametrize(
    ("field", "value", "ok"),
    [
        ("link_timeout_ms", 99, False),
        ("link_timeout_ms", 100, True),
        ("link_timeout_ms", 10_000, True),
        ("link_timeout_ms", 10_001, False),
        ("handshake_timeout_s", 0.0, False),
        ("handshake_timeout_s", 0.001, True),
        ("handshake_timeout_s", 30.0, True),
        ("handshake_timeout_s", 30.1, False),
        ("handshake_timeout_s", math.nan, False),
        ("handshake_timeout_s", math.inf, False),
        ("speed_of_sound_mps", 0.5, False),
        ("speed_of_sound_mps", 299.9, False),
        ("speed_of_sound_mps", 300.0, False),
        ("speed_of_sound_mps", 300.1, True),
        ("speed_of_sound_mps", 369.9, True),
        ("speed_of_sound_mps", 370.0, False),
        ("baudrate", 1199, False),
        ("baudrate", 1200, True),
    ],
    ids=lambda v: str(v),
)
def test_each_limit_refuses_a_value_just_outside_and_takes_one_just_inside(field, value, ok):
    assert accepted(**{field: value}) is ok


def test_the_defaults_that_the_driver_and_the_firmware_agree_on():
    cfg = UltrasonicConfig()
    assert (cfg.echo_window_ms, cfg.inter_sensor_gap_ms) == (30, 30)
    assert (cfg.speed_of_sound_mps, cfg.baudrate) == (343.2, 115200)
    assert (cfg.stale_after_ms, cfg.link_timeout_ms, cfg.handshake_timeout_s) == (700, 1500, 2.0)
    assert (cfg.min_range_cm, cfg.max_range_cm, cfg.alive_window_s) == (25.0, 400.0, 10.0)
    assert cfg.near_hold_s == 30.0
