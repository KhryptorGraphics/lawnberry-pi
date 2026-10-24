"""``config/tractor.yaml`` is validated strictly; NaN/inf never become full scale (SF-11, SF-12).

SF-11: an empty ``default_hold_ms:`` raised ``TypeError`` from the constructor, so
``get_tractor_service()`` raised on every call and the unauthenticated E-stop route answered 500;
``enabled: "false"`` (a string) became ``True``; a hold had no upper bound. SF-12: NaN clamps to
full scale in ``ServoActuator.microseconds``. Also here: the shipped config really reaches the
service and the PCA9685 driver (the review's surviving mutant ``PCA9685Driver({})`` and the
tautological ``test_service_pca9685_config_matches_yaml``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from _tractor_fakes import FakeGpiozero, mowing_tractor
from fastapi import HTTPException

from backend.src.api import rest
from backend.src.core.globals import _client_emergency
from backend.src.drivers.actuators import ServoActuator, ServoCalibration
from backend.src.services import tractor_service
from backend.src.services.tractor_config import TractorConfigError
from backend.src.services.tractor_service import TractorControlService

SHIPPED = yaml.safe_load(Path("config/tractor.yaml").read_text())
NAN, INF = float("nan"), float("inf")


def test_the_shipped_config_reaches_the_service():
    svc = TractorControlService(config=SHIPPED)
    deadman = SHIPPED["command_deadman"]

    assert svc.default_hold_s == deadman["default_hold_ms"] / 1000.0 == 0.5
    assert svc.max_hold_s == deadman["max_hold_ms"] / 1000.0 == 2.0
    assert svc.starter.gpio == SHIPPED["relays"]["starter"]["gpio"]
    assert svc.blade_pto.gpio == SHIPPED["relays"]["blade_pto"]["gpio"]
    assert svc.starter_pulse_ms == SHIPPED["relays"]["starter"]["pulse_ms"]
    for name in ("left_lever", "throttle", "right_lever"):
        cal = getattr(svc, name).cal
        want = SHIPPED["actuators"][name]
        assert (cal.channel, cal.us_min, cal.us_max) == (
            want["channel"],
            want["us_min"],
            want["us_max"],
        )
    assert svc.enabled is SHIPPED["enabled"]


async def test_the_pca9685_config_reaches_the_driver():
    """The shipped address/bus/frequency must be the driver's, not its defaults."""
    shipped = TractorControlService()  # the real config/tractor.yaml
    health = await shipped._pca9685.health_check()
    pca = SHIPPED["pca9685"]
    assert health["address"] == hex(pca["address"]) == "0x41"
    assert health["bus"] == pca["bus"] and health["frequency_hz"] == float(pca["frequency_hz"])

    custom = TractorControlService(
        config={"pca9685": {"address": 0x42, "bus": 3, "frequency_hz": 60}}
    )
    health = await custom._pca9685.health_check()
    assert (health["address"], health["bus"], health["frequency_hz"]) == ("0x42", 3, 60.0)


BAD_CONFIGS = [
    ({"enabled": "false"}, "enabled"),
    ({"enabled": 1}, "enabled"),
    ({"command_deadman": {"default_hold_ms": None}}, "default_hold_ms"),
    ({"command_deadman": {"default_hold_ms": NAN}}, "default_hold_ms"),
    ({"command_deadman": {"default_hold_ms": "500"}}, "default_hold_ms"),
    ({"command_deadman": {"max_hold_ms": 1e9}}, "max_hold_ms"),
    ({"command_deadman": {"default_hold_ms": 900, "max_hold_ms": 100}}, "max_hold_ms"),
    ({"command_deadman": 500}, "command_deadman"),
    ({"pca9685": {"address": "0x41"}}, "address"),
    ({"pca9685": {"frequency_hz": INF}}, "frequency_hz"),
    ({"actuators": {"left_lever": {"channel": 16}}}, "channel"),
    ({"actuators": {"left_lever": {"channel": 1}}}, "different PCA9685 channels"),
    ({"actuators": {"left_lever": {"us_min": 2000, "us_max": 1000}}}, "us_min"),
    ({"actuators": {"left_lever": {"us_max": 9000}}}, "us_max"),
    ({"relays": {"starter": {"gpio": 99}}}, "gpio"),
    ({"relays": {"starter": {"gpio": 26}}}, "different GPIO"),
    ({"relays": {"starter": {"pulse_ms": 600000}}}, "pulse_ms"),
    ({"relays": {"blade_pto": {"active_high": "yes"}}}, "active_high"),
    ({"interlocks": {"require_blade_off_to_start": "false"}}, "require_blade_off_to_start"),
    ({"actuation_supervision": {"interval_ms": 0}}, "interval_ms"),
]


@pytest.mark.parametrize(("cfg", "needle"), BAD_CONFIGS)
def test_an_invalid_config_is_an_explicit_error_not_a_coercion(cfg, needle):
    with pytest.raises(TractorConfigError) as err:
        TractorControlService(config=cfg)
    assert needle in str(err.value)


def test_every_problem_is_reported_at_once():
    cfg = {
        "enabled": "no",
        "command_deadman": {"max_hold_ms": NAN},
        "relays": {"starter": {"gpio": 99}},
    }
    with pytest.raises(TractorConfigError) as err:
        TractorControlService(config=cfg)
    assert len(err.value.problems) == 3, err.value.problems


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"enabled": True},
        {"command_deadman": {"default_hold_ms": 80, "max_hold_ms": 300}},
        {"command_deadman": None},
        {"pca9685": {"address": 0x41, "bus": 1, "frequency_hz": 50}},
        {"relays": {"starter": {"gpio": 5, "pulse_ms": 800}, "blade_pto": {"gpio": 26}}},
        {"actuators": {"throttle": {"channel": 1, "us_min": 1100, "us_max": 1900}}},
    ],
)
def test_valid_configs_still_load(cfg):
    assert isinstance(TractorControlService(config=cfg), TractorControlService)


async def test_a_broken_config_file_does_not_take_the_estop_route_down(monkeypatch, tmp_path):
    from backend.src.main import app

    # The usual "commented out" typo: the value is empty.
    (tmp_path / "tractor.yaml").write_text("command_deadman:\n  default_hold_ms:\n")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(tractor_service, "_tractor_service", None)
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FakeGpiozero(monkeypatch)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        stop = await c.post("/api/v2/tractor/emergency-stop")
        blade = await c.post("/api/v2/tractor/blade", json={"engaged": True})
        state = await c.get("/api/v2/tractor/state")

    assert stop.status_code == 200, "the E-stop route must never answer 500"
    assert stop.json()["status"] == "not_enabled" and "default_hold_ms" in stop.json()["detail"]
    assert blade.json()["status"] == "rejected"
    assert state.json()["actuator_ok"] is False
    assert "default_hold_ms" in state.json()["actuator_fault"]
    assert gpio.devices == [], "a pin from a file that cannot be trusted is not driven"
    svc = tractor_service.get_tractor_service()
    assert svc.enabled is False and svc.lockout_reason and "default_hold_ms" in svc.config_error
    assert (await svc.clear_emergency())["status"] == "rejected"  # a lock-out needs a restart


def test_a_config_file_that_is_not_a_mapping_is_an_explicit_error(monkeypatch, tmp_path):
    (tmp_path / "tractor.yaml").write_text("- not\n- a mapping\n")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    with pytest.raises(TractorConfigError, match="top level must be a mapping"):
        TractorControlService()


# --------------------------------- SF-12: NaN / inf ---------------------------------


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_a_servo_refuses_a_non_finite_command_instead_of_clamping_it_to_full_scale(bad):
    servo = ServoActuator("left_lever", ServoCalibration(channel=0))
    with pytest.raises(ValueError, match="non-finite"):
        servo.microseconds(bad)
    with pytest.raises(ValueError, match="non-finite"):
        servo.command(bad)
    assert servo.value == 0.0 and servo.us == 1500, "a refused command changes nothing"


async def test_non_finite_levers_and_throttle_are_rejected_before_the_chip():
    svc, pca = mowing_tractor()
    for refused in (
        await svc.set_levers(NAN, 0.5),
        await svc.set_levers(0.5, INF),
        await svc.set_left_lever(NAN),
        await svc.set_right_lever(-INF),
        await svc.set_throttle(NAN),
    ):
        assert refused["status"] == "rejected" and "non-finite" in refused["reason"], refused
    assert pca.log == [] and svc.state.left_lever == 0.0 and svc.state.throttle == 0.0
    await svc.shutdown()


@pytest.mark.parametrize(
    "patch",
    [
        {"vector": {"linear": NAN, "angular": 0.0}},
        {"vector": {"linear": 0.5, "angular": INF}},
        {"max_speed_limit": NAN},
    ],
)
async def test_the_drive_route_refuses_non_finite_input(monkeypatch, patch):
    monkeypatch.setattr(rest, "_resolve_manual_session", lambda sid: {"principal": "test"})
    monkeypatch.setattr(rest, "_emergency_until", 0.0)
    _client_emergency.clear()
    request = SimpleNamespace(headers={}, state=SimpleNamespace())
    cmd = {"session_id": "s", "vector": {"linear": 0.5, "angular": 0.0}, "duration_ms": 100}

    with pytest.raises(HTTPException) as err:
        await rest.control_drive_v2({**cmd, **patch}, request)

    assert err.value.status_code == 422 and "finite" in err.value.detail
