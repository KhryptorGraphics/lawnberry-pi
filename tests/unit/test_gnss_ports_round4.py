"""Which serial ports the GNSS driver may open (audit GPS-12, SAF-S1 and their mutants).

The RoboHAT MM1 and the ultrasonic MCU are USB CDC ACM ports next to a USB GNSS receiver, and a
second reader on a tty steals the bytes its owner is waiting for. The driver knows them from
configuration (environment, the settings profile, ``hardware.yaml``), from identity (vendor id,
descriptor) and from the lock the owner holds. Each of those is tested alone here, because each
is the only guard on the day the others are missing.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from _gnss_fakes import Port, install_serial

from backend.src.drivers.sensors import gps_driver
from backend.src.drivers.sensors.gps_driver import (
    GPSDriver,
    SerialPortInfo,
    ports_owned_by_other_drivers,
    preferred_gps_device,
)
from backend.src.models.sensor_data import GpsMode

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """No ambient port names, and a settings profile / config directory of the test's own."""
    for var in (
        "GPS_DEVICE",
        "GPS_EXCLUDE_DEVICES",
        "ROBOHAT_SERIAL_PORT",
        "ROBOHAT_PORT",
        "LAWN_ROBOHAT_PORT",
        "LAWN_SETTINGS_DIR",
        "LAWN_CONFIG_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.chdir(tmp_path)  # no ./config/default.json of this repository
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(config_dir(tmp_path, ultrasonic_port=None)))
    monkeypatch.setattr(gps_driver, "_warned_ownership_config", False, raising=False)


def config_dir(tmp_path: Path, *, ultrasonic_port: str | None) -> Path:
    """A copy of the shipped hardware.yaml / limits.yaml with ``ultrasonic.port`` set."""
    target = tmp_path / "cfg"
    target.mkdir(exist_ok=True)
    shutil.copy(REPO / "config" / "limits.yaml", target / "limits.yaml")
    text = (REPO / "config" / "hardware.yaml").read_text(encoding="utf-8")
    marker = " port: null"
    assert text.count(marker) == 1, "the shipped ultrasonic block changed: adapt this helper"
    port = "null" if ultrasonic_port is None else ultrasonic_port
    (target / "hardware.yaml").write_text(text.replace(marker, f" port: {port}"), encoding="utf-8")
    return target


# --------------------------------------------------------------------------------- who owns a port


def info(device: str, vid: int | None = None, description: str = "") -> SerialPortInfo:
    return SerialPortInfo(device=device, vid=vid, description=description)


def test_a_raspberry_pi_rp2040_vendor_id_alone_marks_a_port_as_another_boards():
    owned = ports_owned_by_other_drivers([info("/dev/ttyACM9", 0x2E8A, "generic serial")])
    assert "/dev/ttyACM9" in owned


@pytest.mark.parametrize(
    "description",
    [
        "Robo HAT MM1",
        "Robo_HAT MM1",
        "robohat",
        "Adafruit CircuitPython CDC data",
        "Circuit-Python",
        "RP2040-Zero",
        "Raspberry Pi Pico",
        "Cytron Maker Pi",
        "LawnBerry ultrasonic",
    ],
)
def test_a_descriptor_that_names_another_lawnberry_board_alone_marks_it(description):
    owned = ports_owned_by_other_drivers([info("/dev/ttyACM9", None, description)])
    assert "/dev/ttyACM9" in owned


@pytest.mark.parametrize(
    ("vid", "description"),
    [(0x1546, "u-blox GNSS receiver"), (0x0403, "FT232R USB UART"), (None, "CP2102 USB to UART")],
)
def test_a_receiver_or_an_adapter_is_not_another_drivers_port(vid, description):
    assert ports_owned_by_other_drivers([info("/dev/ttyUSB0", vid, description)]) == set()


@pytest.mark.parametrize("name", ["ROBOHAT_SERIAL_PORT", "ROBOHAT_PORT", "LAWN_ROBOHAT_PORT"])
def test_every_variable_that_names_the_robohat_port_counts(monkeypatch, name):
    monkeypatch.setenv(name, "/dev/ttyACM6")
    assert "/dev/ttyACM6" in ports_owned_by_other_drivers([])


def test_the_exclude_list_is_comma_separated_and_tolerates_blanks(monkeypatch):
    monkeypatch.setenv("GPS_EXCLUDE_DEVICES", " /dev/ttyACM4 ,,/dev/ttyUSB3 ")
    assert ports_owned_by_other_drivers([]) >= {"/dev/ttyACM4", "/dev/ttyUSB3"}


def test_a_port_named_through_a_symlink_is_excluded_by_its_real_path(monkeypatch, tmp_path):
    """The by-id path of the RoboHAT and the ttyACM node a USB scan finds are one device."""
    real = tmp_path / "ttyACM7"
    real.write_text("")
    link = tmp_path / "usb-RoboHAT-if00"
    link.symlink_to(real)
    monkeypatch.setenv("ROBOHAT_SERIAL_PORT", str(link))
    owned = ports_owned_by_other_drivers([])
    assert str(real) in owned and str(link) not in owned


def test_the_ultrasonic_port_of_hardware_yaml_is_another_drivers_port(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "LAWNBERRY_CONFIG_DIR", str(config_dir(tmp_path, ultrasonic_port="/dev/ttyACM8"))
    )
    assert "/dev/ttyACM8" in ports_owned_by_other_drivers([])


def test_a_hardware_yaml_with_no_ultrasonic_port_names_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(config_dir(tmp_path, ultrasonic_port=None)))
    assert ports_owned_by_other_drivers([]) == set()


def test_the_robohat_port_of_the_settings_profile_is_the_robohats(monkeypatch, tmp_path):
    """``robohat_service`` honours ``hardware.robohat_port`` of the settings profile, so that port
    is the RoboHAT's whatever the environment says."""
    settings = tmp_path / "settings"
    settings.mkdir()
    (settings / "default.json").write_text(
        json.dumps({"hardware": {"robohat_port": "/dev/ttyACM3"}})
    )
    monkeypatch.setenv("LAWN_SETTINGS_DIR", str(settings))
    assert "/dev/ttyACM3" in ports_owned_by_other_drivers([])


@pytest.mark.parametrize("variable", ["LAWN_SETTINGS_DIR", "LAWN_CONFIG_DIR", None])
@pytest.mark.parametrize("name", ["default.json", "settings.json"])
def test_the_settings_profile_is_read_from_the_places_robohat_service_reads_it(
    monkeypatch, tmp_path, variable, name
):
    """One source of truth for where the RoboHAT's port is written: the same files in the same
    order as ``services.robohat_service`` (a mismatch would exclude a port nobody uses and probe
    the one the RoboHAT does)."""
    from backend.src.services import robohat_service

    base = tmp_path / "profile"
    base.mkdir()
    (base / name).write_text(json.dumps({"hardware": {"robohat_port": "/dev/ttyACM5"}}))
    if variable is None:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / name).write_text((base / name).read_text())
    else:
        monkeypatch.setenv(variable, str(base))
    assert gps_driver._profile_robohat_port() == robohat_service._read_profile_robohat_port()
    assert gps_driver._profile_robohat_port() == "/dev/ttyACM5"


def test_a_settings_profile_without_a_robohat_port_names_nothing(monkeypatch, tmp_path):
    settings = tmp_path / "settings"
    settings.mkdir()
    for body in ('{"hardware": {"robohat_port": null}}', "not json", '{"hardware": 3}', "[]"):
        (settings / "default.json").write_text(body)
        monkeypatch.setenv("LAWN_SETTINGS_DIR", str(settings))
        assert gps_driver._profile_robohat_port() == ""


# -------------------------------------------------------------------------------- preferred device


def test_the_u_blox_port_that_is_another_drivers_is_not_preferred(monkeypatch):
    ports = [info("/dev/ttyACM1", 0x1546, "u-blox A"), info("/dev/ttyACM2", 0x1546, "u-blox B")]
    assert preferred_gps_device(GpsMode.F9P_USB, ports) == "/dev/ttyACM1"
    monkeypatch.setenv("GPS_EXCLUDE_DEVICES", "/dev/ttyACM1")
    assert preferred_gps_device(GpsMode.F9P_USB, ports) == "/dev/ttyACM2"
    monkeypatch.setenv("GPS_EXCLUDE_DEVICES", "/dev/ttyACM1,/dev/ttyACM2")
    assert preferred_gps_device(GpsMode.F9P_USB, ports) is None


def test_only_a_u_blox_port_is_preferred_for_a_usb_receiver():
    ports = [info("/dev/ttyACM0", None, "generic"), info("/dev/ttyUSB0", 0x0403, "FTDI")]
    assert preferred_gps_device(GpsMode.F9P_USB, ports) is None


def test_a_uart_receiver_prefers_the_first_uart_that_exists_and_is_not_owned(monkeypatch):
    exists = {"/dev/ttyAMA0", "/dev/serial0", "/dev/ttyS0"}
    monkeypatch.setattr(gps_driver.os.path, "exists", lambda p: p in exists)
    assert preferred_gps_device(GpsMode.LC29H_UART, []) == "/dev/ttyAMA0"
    monkeypatch.setenv("GPS_EXCLUDE_DEVICES", "/dev/ttyAMA0")
    assert preferred_gps_device(GpsMode.LC29H_UART, []) == "/dev/serial0"
    exists.discard("/dev/serial0")
    assert preferred_gps_device(GpsMode.LC29H_UART, []) == "/dev/ttyS0"


# ---------------------------------------------------------------------------------- what is probed


def fake_ports(monkeypatch, found, *, acm=(), usb=()):
    """``/dev/ttyACM*`` and ``/dev/ttyUSB*`` as this machine would list them, ``found`` as pyserial
    enumerates them; every device a driver may open is a silent port (so the probe fails)."""
    ports = {d: Port("silent") for d, _v, _t in found}
    ports.update({d: Port("silent") for d in (*acm, *usb)})
    serial = install_serial(monkeypatch, ports, found=found)

    def fake_glob(pattern):
        if "ACM" in pattern:
            return list(acm)
        return list(usb) if "USB" in pattern else []

    monkeypatch.setattr(gps_driver.glob, "glob", fake_glob)
    monkeypatch.setattr(gps_driver, "_UART_DEVICES", ())
    monkeypatch.setattr(gps_driver, "_USB_SERIAL_DEVICES", ())
    return serial


def driver(**cfg) -> GPSDriver:
    return GPSDriver({"mode": GpsMode.LC29H_UART, "uart_device": "/dev/null0", **cfg})


def test_the_device_named_by_the_environment_is_probed_first(monkeypatch):
    fake_ports(monkeypatch, [("/dev/ttyACM4", 0x1546, "u-blox")], acm=["/dev/ttyACM4"])
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyUSB7")
    usable, skipped = driver()._candidates()
    assert usable[0] == "/dev/ttyUSB7" and skipped == []


def test_a_u_blox_port_is_a_candidate_by_its_vendor_id_even_if_the_node_is_not_listed(monkeypatch):
    fake_ports(monkeypatch, [("/dev/ttyACM4", 0x1546, "u-blox")])  # no ttyACM* on this machine
    usable, _skipped = driver()._candidates()
    assert "/dev/ttyACM4" in usable


def test_a_device_named_twice_is_probed_once(monkeypatch):
    serial = fake_ports(monkeypatch, [])
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyAMA0")
    candidates, _ = driver(uart_device="/dev/ttyAMA0")._candidates()
    assert candidates.count("/dev/ttyAMA0") == 1 and serial.opens == []


def test_a_port_another_driver_owns_is_skipped_and_reported(monkeypatch):
    fake_ports(
        monkeypatch,
        [("/dev/ttyACM0", 0x2E8A, "RP2040"), ("/dev/ttyACM1", 0x1546, "u-blox")],
        acm=["/dev/ttyACM0", "/dev/ttyACM1"],
    )
    usable, skipped = driver()._candidates()
    assert "/dev/ttyACM0" in skipped and "/dev/ttyACM0" not in usable
    assert "/dev/ttyACM1" in usable


def test_an_acm_device_nobody_enumerated_is_not_probed(monkeypatch):
    """pyserial listed no identity for it: it may be the RoboHAT or the ultrasonic MCU."""
    fake_ports(
        monkeypatch, [("/dev/ttyACM1", 0x1546, "u-blox")], acm=["/dev/ttyACM1", "/dev/ttyACM7"]
    )
    usable, skipped = driver()._candidates()
    assert "/dev/ttyACM7" not in usable and "/dev/ttyACM7" not in skipped
    assert "/dev/ttyACM1" in usable


def test_an_enumerated_acm_device_that_belongs_to_nobody_is_probed(monkeypatch):
    fake_ports(monkeypatch, [("/dev/ttyACM5", 0x10C4, "CP210x")], acm=["/dev/ttyACM5"])
    usable, _skipped = driver()._candidates()
    assert "/dev/ttyACM5" in usable


def test_with_no_enumeration_no_acm_device_is_probed_but_the_named_ones_are(monkeypatch):
    """pyserial could not list the ports: nothing can be checked, so an ``ACM`` device found by
    globbing may be any board. A device the operator names is theirs to name."""
    fake_ports(monkeypatch, [], acm=["/dev/ttyACM1"], usb=["/dev/ttyUSB4"])
    monkeypatch.setattr(
        gps_driver, "_enumerate_serial_ports", lambda: ([info("/dev/ttyACM1", 0x10C4, "x")], False)
    )
    usable, _ = driver()._candidates()
    assert "/dev/ttyACM1" not in usable
    assert "/dev/ttyUSB4" in usable  # a USB-serial adapter is no board of ours
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyACM1")
    usable, _ = driver()._candidates()
    assert usable[0] == "/dev/ttyACM1"


def test_with_unreadable_configuration_no_unidentified_acm_device_is_probed_and_it_is_said(
    monkeypatch, tmp_path, caplog
):
    """``ultrasonic.port`` is unknown when ``hardware.yaml`` cannot be read: the identity checks
    are then the only guard, and an enumerated device they cannot place is left alone."""
    fake_ports(monkeypatch, [("/dev/ttyACM5", 0x10C4, "CP210x")], acm=["/dev/ttyACM5"])
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(
        "LAWNBERRY_CONFIG_DIR", str(empty)
    )  # no limits.yaml: strict outside SIM_MODE
    with caplog.at_level(logging.WARNING):
        usable, _ = driver()._candidates()
        driver()._candidates()
    assert "/dev/ttyACM5" not in usable
    said = [
        r for r in caplog.records if "hardware configuration could not be read" in r.getMessage()
    ]
    assert len(said) == 1  # once, not on every probe
    # and with readable configuration the same device is a candidate
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(config_dir(tmp_path, ultrasonic_port=None)))
    usable, _ = driver()._candidates()
    assert "/dev/ttyACM5" in usable


def test_an_open_that_fails_is_tried_once_not_once_per_baud_rate(monkeypatch):
    serial = fake_ports(monkeypatch, [])
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyUSB9")  # no such port: the open raises
    d = driver()
    d.PROBE_WINDOW_S = 0.05
    assert d._open_port() is False
    assert [p for p, _b, _k in serial.opens].count("/dev/ttyUSB9") == 1


def test_every_baud_rate_is_tried_on_a_port_that_opens_but_is_silent(monkeypatch):
    serial = fake_ports(monkeypatch, [("/dev/ttyACM5", 0x10C4, "CP210x")], acm=["/dev/ttyACM5"])
    d = driver()
    d.PROBE_WINDOW_S = 0.05
    assert d._open_port() is False
    bauds = [b for p, b, _k in serial.opens if p == "/dev/ttyACM5"]
    assert bauds == d._baud_candidates and len(bauds) == 3
    assert d._next_probe_at > 0.0  # and it is not repeated at once
