"""scripts/ultrasonic_mcu_probe.py: the bench tool for audit measurement M-40.

The owner will rely on its CSV to decide what a healthy JSN-SR04T does in open air and where the
blind zone ends, so the file must carry the raw echo width, the status name and the timing, and the
tool must exit non-zero when nothing arrived.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from pathlib import Path

from tests.unit._usonic_fakes import FakeClock, FakeMcuFactory, SensorModel

spec = importlib.util.spec_from_file_location(
    "scripts.ultrasonic_mcu_probe", Path("scripts/ultrasonic_mcu_probe.py")
)
probe = importlib.util.module_from_spec(spec)
sys.modules["scripts.ultrasonic_mcu_probe"] = probe
spec.loader.exec_module(probe)


def run_probe(tmp_path, sensors, seconds=3.0, **kw):
    clock = FakeClock()
    factory = FakeMcuFactory(clock, sensors=sensors)
    out = io.StringIO()
    path = tmp_path / "m40.csv"
    code = probe.run(
        "/dev/ttyACM-usonic", seconds, str(path),
        serial_factory=factory, clock=clock.now, sleep=clock.advance, out=out, **kw,
    )  # fmt: skip
    return code, path, out.getvalue(), factory


def test_probe_logs_raw_echo_widths_with_the_status_name_per_sensor(tmp_path):
    sensors = [
        SensorModel(distance_cm=100.0),
        SensorModel(mode="silent"),
        SensorModel(mode="no_fall"),
        SensorModel(distance_cm=12.0),
    ]
    code, path, out, factory = run_probe(tmp_path, sensors)
    assert code == 0
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert set(rows[0]) == set(probe.CSV_FIELDS)
    by_sensor = {}
    for r in rows:
        by_sensor.setdefault(r["sensor"], []).append(r)
    assert set(by_sensor) == {"outer_left", "inner_left", "inner_right", "outer_right"}
    assert {r["status"] for r in by_sensor["outer_left"]} == {"OK"}
    assert {r["echo_us"] for r in by_sensor["outer_left"]} == {"5828"}  # raw, not rounded to cm
    assert {r["status"] for r in by_sensor["inner_left"]} == {"NO_EDGE"}
    assert {r["echo_us"] for r in by_sensor["inner_left"]} == {"0"}
    assert {r["status"] for r in by_sensor["inner_right"]} == {"NO_FALL"}
    assert {r["echo_us"] for r in by_sensor["outer_right"]} == {"699"}  # a 12 cm target: blind zone
    assert [int(r["fails"]) for r in by_sensor["inner_left"]][:3] == [1, 2, 3]
    assert "link=active" in out and "readings=" in out
    assert factory.current.closed  # the probe leaves the port released


def test_probe_exits_nonzero_and_says_why_when_no_mcu_answers(tmp_path):
    clock = FakeClock()
    out = io.StringIO()

    def no_device(path, baud):
        raise OSError("no such device")

    code = probe.run(
        "/dev/ttyACM-usonic", 1.0, str(tmp_path / "x.csv"),
        serial_factory=no_device, clock=clock.now, sleep=clock.advance, out=out,
    )  # fmt: skip
    assert code == 2
    assert "link=disconnected" in out.getvalue() and "open_failed" in out.getvalue()
    assert "Noneus" not in out.getvalue()
    assert (tmp_path / "x.csv").read_text().strip() == ",".join(probe.CSV_FIELDS)


def test_probe_cli_requires_the_port():
    import pytest

    with pytest.raises(SystemExit):
        probe.main([])
