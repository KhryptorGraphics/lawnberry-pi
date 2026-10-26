#!/usr/bin/env python3
"""Record raw frames from the ultrasonic MCU for audit measurement M-40 (JSN-SR04T behaviour).

Read-only with respect to the machine: it opens the MCU's USB *data* serial port, listens for the
MCU's own INFO (it writes nothing to a port that has not first sent a valid frame), performs the
normal handshake (CONFIG with the values below) and logs what the sensors report. It actuates
nothing and never touches a Raspberry Pi GPIO pin. Use it with the array on the bench to answer
the questions the audit lists as unmeasured: what a healthy sensor reports in open air (no pulse?
a maximum-length pulse?), the real blind zone (put a flat target at 10, 15, 20, 25, 30 cm), the
beam edge, behaviour on grass / people / pets / low objects, rain, and the cycle timing.

    python scripts/ultrasonic_mcu_probe.py --port /dev/serial/by-id/<mcu data port> \\
        --seconds 60 --csv /tmp/m40_open_air.csv

The CSV is the raw frame log: ONE ROW PER VALID READING FRAME the MCU sent, including the ones the
driver ignored (handshake, a channel it did not enable) or refused (a replay, a timestamp that
does not fit the MCU clock), with what it did in the ``disposition`` column (``accepted``,
``ignored_link``, ``ignored_channel``, ``bad_sensor``, ``refused_clock``). Columns: wall_time_utc,
sensor, status, echo_us, distance_cm_at_343mps, health, fails, cycle, mcu_t_ms, disposition.
``distance_cm_at_343mps`` is only a convenience conversion; keep ``echo_us``.

The last line counts them: ``frames`` (all), ``refused``, ``ignored`` and ``readings`` (accepted).
Exit code 0 if readings were accepted, 2 if the link never delivered a reading frame (see the
printed link state), 3 if reading frames arrived but the driver accepted none of them (the CSV
shows why), 64 if the timing arguments are not a valid configuration.

The driver opens the data port EXCLUSIVELY: while the LawnBerry backend (or another probe) holds it
this one fails with ``open_failed``. It then says so; stop that process first and run it again.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto  # noqa: E402
from backend.src.drivers.sensors.jsn_sr04t_driver import (  # noqa: E402
    Disposition,
    JsnSr04tMcuDriver,
    SerialFactory,
    echo_to_cm,
    open_pyserial,
)
from backend.src.models.hardware_config import STALE_MARGIN_MS, UltrasonicConfig  # noqa: E402

CSV_FIELDS = (
    "wall_time_utc", "sensor", "status", "echo_us", "distance_cm_at_343mps", "health", "fails",
    "cycle", "mcu_t_ms", "disposition",
)  # fmt: skip

OPEN_FAILED_HINT = (
    "cannot open the data port: check the path, and that nothing else holds it. The driver opens "
    "the port exclusively, so the LawnBerry backend or another probe running on it blocks this "
    "one: stop it first (for example 'sudo systemctl stop lawnberry-backend'). Retrying until the "
    "time is up."
)

#: Exit codes (0 = readings were accepted).
EXIT_NO_FRAMES = 2
EXIT_NONE_ACCEPTED = 3
EXIT_BAD_TIMING = 64


def _status_name(code: int) -> str:
    try:
        return proto.EchoStatus(code).name
    except ValueError:
        return f"UNKNOWN_{code}"


def probe_config(port: str, echo_window_ms: int, gap_ms: int) -> UltrasonicConfig:
    """The probe's configuration for the timings under test.

    ``stale_after_ms`` follows the cycle (the configuration refuses one that is shorter than a
    full cycle plus its margin), so slow timings such as ``--echo-window-ms 100 --gap-ms 500`` can
    be measured: the probe judges nothing, it only logs.
    """
    cycle_ms = (echo_window_ms + gap_ms) * len(UltrasonicConfig().channels)
    stale_after_ms = max(700, cycle_ms + 2 * STALE_MARGIN_MS)
    return UltrasonicConfig(
        port=port,
        echo_window_ms=echo_window_ms,
        inter_sensor_gap_ms=gap_ms,
        stale_after_ms=stale_after_ms,
    )


def run(
    port: str,
    seconds: float,
    csv_path: str | None = None,
    *,
    echo_window_ms: int = 30,
    gap_ms: int = 30,
    serial_factory: SerialFactory | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    out: TextIO = sys.stdout,
) -> int:
    try:
        cfg = probe_config(port, echo_window_ms, gap_ms)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "timing"
        print(f"probe: not a valid configuration: {where}: {first['msg']}", file=out)
        return EXIT_BAD_TIMING
    driver = JsnSr04tMcuDriver(
        cfg, serial_factory=serial_factory or open_pyserial, clock=clock, simulate=False
    )
    fates: Counter[Disposition] = Counter()
    with contextlib.ExitStack() as stack:
        writer = None
        if csv_path:
            handle = stack.enter_context(open(csv_path, "w", newline="", encoding="utf-8"))
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
        stack.callback(driver.close)

        def on_reading(r: proto.Reading, wall: datetime, fate: Disposition) -> None:
            fates[fate] += 1
            if writer:
                sensor = cfg.channels[r.sensor] if r.sensor < len(cfg.channels) else r.sensor
                approx = f"{echo_to_cm(r.echo_us, 343.0):.1f}" if r.echo_us else ""
                writer.writerow(
                    [wall.isoformat(), sensor, _status_name(r.status), r.echo_us, approx,
                     r.health, r.fails, r.cycle, r.t_ms, fate.value]
                )  # fmt: skip

        driver.reading_hook = on_reading
        deadline = clock() + seconds
        next_report = clock()
        hinted = False
        while clock() < deadline:
            driver.poll()
            if not hinted and (driver.link_fault or "").startswith("open_failed"):
                hinted = True
                print(f"probe: {OPEN_FAILED_HINT} [{driver.link_fault}]", file=out)
            if clock() >= next_report:
                next_report = clock() + 1.0
                print(_table(driver), file=out)
            sleep(0.005)
        final_state, final_fault = driver.link_state.value, driver.link_fault
    frames = sum(fates.values())
    accepted = fates[Disposition.ACCEPTED]
    refused = fates[Disposition.REFUSED_CLOCK]
    print(
        f"link={final_state} fault={final_fault} frames={frames} refused={refused} "
        f"ignored={frames - accepted - refused} readings={accepted}",
        file=out,
    )
    if accepted:
        return 0
    return EXIT_NONE_ACCEPTED if frames else EXIT_NO_FRAMES


def _table(driver: JsnSr04tMcuDriver) -> str:
    cells = [
        f"{c.sensor_id}={c.status.value}:{c.distance_cm:.1f}cm/"
        f"{'-' if c.echo_us is None else c.echo_us}us"
        for c in driver.snapshot()
    ]
    return f"[{driver.link_state.value}] " + "  ".join(cells)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", required=True, help="the MCU's USB data serial port (NOT the REPL)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--csv", help="write every reading frame to this file")
    ap.add_argument("--echo-window-ms", type=int, default=30)
    ap.add_argument("--gap-ms", type=int, default=30)
    args = ap.parse_args(argv)
    return run(
        args.port, args.seconds, args.csv, echo_window_ms=args.echo_window_ms, gap_ms=args.gap_ms
    )


if __name__ == "__main__":
    sys.exit(main())
