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

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto
from tests.unit._usonic_fakes import FakeClock, FakeMcuFactory, SensorModel
from tests.unit.test_jsn_sr04t_driver import MCU_BASE_MS, ScriptedPort

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


LOCKED = (
    "Could not exclusively lock port /dev/ttyACM-usonic: "
    "[Errno 11] Resource temporarily unavailable"
)


def test_probe_says_in_plain_words_that_another_process_holds_the_port(tmp_path):
    # The driver opens the data port exclusively, so with the backend (or another probe) running a
    # second opener fails with open_failed. The probe must say what that means, once, and keep the
    # exit code and the link= line unchanged.
    clock = FakeClock()
    out = io.StringIO()

    def locked(*_args, **_kwargs):
        raise OSError(LOCKED)

    code = probe.run(
        "/dev/ttyACM-usonic", 3.0, str(tmp_path / "x.csv"),
        serial_factory=locked, clock=clock.now, sleep=clock.advance, out=out,
    )  # fmt: skip
    text = out.getvalue()
    assert code == 2
    assert text.count("holds it") == 1  # said once, not on every retry
    assert "exclusively" in text and "backend" in text and "another probe" in text
    assert "stop it first" in text and "lawnberry-backend" in text
    assert LOCKED in text  # the driver's own reason is still shown
    last = text.strip().splitlines()[-1]
    assert last.startswith("link=disconnected fault=open_failed: ") and last.endswith("readings=0")
    assert (tmp_path / "x.csv").read_text().strip() == ",".join(probe.CSV_FIELDS)


def test_probe_keeps_retrying_and_works_once_the_port_is_free(tmp_path):
    clock = FakeClock()
    factory = FakeMcuFactory(clock, sensors=[SensorModel(distance_cm=100.0)] * 4)
    refusals = [2]

    def busy_then_free(*args, **kwargs):
        if refusals[0] > 0:
            refusals[0] -= 1
            raise OSError(LOCKED)
        return factory(*args, **kwargs)

    out = io.StringIO()
    code = probe.run(
        "/dev/ttyACM-usonic", 3.0, str(tmp_path / "x.csv"),
        serial_factory=busy_then_free, clock=clock.now, sleep=clock.advance, out=out,
    )  # fmt: skip
    assert code == 0
    assert out.getvalue().count("holds it") == 1
    assert "link=active" in out.getvalue()


def test_probe_prints_no_port_hint_when_the_port_opens(tmp_path):
    code, _path, out, _factory = run_probe(tmp_path, [SensorModel(distance_cm=100.0)] * 4)
    assert code == 0 and "holds it" not in out


def test_probe_cli_requires_the_port():
    import pytest

    with pytest.raises(SystemExit):
        probe.main([])


# ---- the CSV is the raw frame log (US3-10) -----------------------------------------------------


def rows_of(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_probe_logs_the_frames_the_driver_refuses_and_says_what_it_did_with_each(tmp_path):
    clock = FakeClock()
    base = FakeMcuFactory(clock, sensors=[SensorModel(distance_cm=100.0)] * 4)

    def duplicating(path, baud):
        port = base(path, baud)
        port.corrupt_next(10**9, lambda frame: frame + frame if frame[3] == 0x83 else frame)
        return port

    out = io.StringIO()
    path = tmp_path / "m40.csv"
    code = probe.run(
        "/dev/ttyACM-usonic", 3.0, str(path),
        serial_factory=duplicating, clock=clock.now, sleep=clock.advance, out=out,
    )  # fmt: skip
    rows = rows_of(path)
    fates = [r["disposition"] for r in rows]
    assert code == 0 and set(fates) == {"accepted", "refused_clock"}
    assert fates.count("accepted") == fates.count("refused_clock"), "every frame was sent twice"
    last = out.getvalue().strip().splitlines()[-1]
    assert f"frames={len(rows)} refused={fates.count('refused_clock')} ignored=0 " in last
    assert last.endswith(f"readings={fates.count('accepted')}")
    assert rows[0]["disposition"] == "accepted" and rows[1]["disposition"] == "refused_clock"
    assert rows[0]["echo_us"] == rows[1]["echo_us"] == "5828"


def scripted_probe(tmp_path, steps, seconds=3.0):
    """The probe against a scripted port: ``steps`` is ``[(seconds since start, push(port))]``."""
    clock = FakeClock()
    port = ScriptedPort()
    pending = sorted(steps, key=lambda s: s[0])

    def sleep(dt):
        clock.advance(dt)
        while pending and pending[0][0] <= clock.now() - 1000.0:
            pending.pop(0)[1](port)

    out = io.StringIO()
    path = tmp_path / "m40.csv"
    code = probe.run(
        "/dev/ttyACM-usonic", seconds, str(path),
        serial_factory=lambda _p, _b: port, clock=clock.now, sleep=sleep, out=out,
    )  # fmt: skip
    return code, rows_of(path), out.getvalue()


def handshake(port):
    port.info()


def ack(port):
    port.ack(proto.McuConfig(enable_mask=0b1111, echo_window_ms=30, gap_ms=30, quiet_after=8))


def test_probe_logs_readings_that_arrive_during_the_handshake_and_marks_them_ignored(tmp_path):
    steps = [
        (0.05, handshake),
        (0.06, lambda p: p.reading(sensor=0, t_ms=MCU_BASE_MS + 60)),  # before the CONFIG echo
        (0.10, ack),
        (0.20, lambda p: p.reading(sensor=0, t_ms=MCU_BASE_MS + 200)),
    ]
    code, rows, out = scripted_probe(tmp_path, steps)
    assert code == 0
    assert [r["disposition"] for r in rows] == ["ignored_link", "accepted"]
    assert [r["mcu_t_ms"] for r in rows] == [str(MCU_BASE_MS + 60), str(MCU_BASE_MS + 200)]
    assert "frames=2 refused=0 ignored=1 " in out and out.strip().endswith("readings=1")


def test_probe_exits_3_when_frames_arrive_but_the_driver_accepts_none(tmp_path):
    steps = [(0.05, handshake), (0.10, ack)] + [
        (0.2 + 0.1 * k, lambda p, k=k: p.reading(sensor=7, t_ms=MCU_BASE_MS + 200 + 100 * k))
        for k in range(10)  # a sensor index the host has no channel for
    ]
    code, rows, out = scripted_probe(tmp_path, steps)
    assert code == 3
    assert {r["disposition"] for r in rows} == {"bad_sensor"} and len(rows) == 10
    assert {r["sensor"] for r in rows} == {"7"}
    assert "frames=10 refused=0 ignored=10 " in out and out.strip().endswith("readings=0")


def test_probe_exits_2_only_when_no_reading_frame_arrived_at_all(tmp_path):
    code, rows, out = scripted_probe(tmp_path, [(0.05, handshake), (0.10, ack)])
    assert code == 2 and rows == []
    assert "frames=0 " in out and out.strip().endswith("readings=0")


def test_probe_can_measure_the_slowest_timings_the_firmware_accepts(tmp_path):
    # M-40 has to be able to look at 100 ms windows and 500 ms gaps: the default stale_after_ms
    # (700 ms) is shorter than their 2.4 s cycle, which the configuration refuses
    code, path, out, factory = run_probe(
        tmp_path, [SensorModel(distance_cm=120.0)] * 4, seconds=8.0, echo_window_ms=100, gap_ms=500
    )
    assert code == 0 and "link=active" in out
    assert (factory.current.core.window_ms, factory.current.core.gap_ms) == (100, 500)
    assert {r["status"] for r in rows_of(path)} == {"OK"}
    assert {r["disposition"] for r in rows_of(path)} == {"accepted"}


@pytest.mark.parametrize(
    "timings", [{"echo_window_ms": 5}, {"gap_ms": 1}, {"echo_window_ms": 1000}],
    ids=lambda t: next(iter(t)),
)  # fmt: skip
def test_probe_says_plainly_that_timings_the_configuration_refuses_are_not_valid(tmp_path, timings):
    code, path, out, factory = run_probe(tmp_path, [SensorModel(distance_cm=120.0)] * 4, **timings)
    assert code == 64 and "not a valid configuration" in out and "Traceback" not in out
    assert factory.opens == [], "nothing was opened"
