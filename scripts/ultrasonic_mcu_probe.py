#!/usr/bin/env python3
"""Record raw frames from the ultrasonic MCU for audit measurement M-40 (JSN-SR04T behaviour).

Read-only with respect to the machine: it opens the MCU's USB *data* serial port, performs the
normal handshake (HELLO, then CONFIG with the values below), and logs what the sensors report. It
actuates nothing and never touches a Raspberry Pi GPIO pin. Use it with the array on the bench to
answer the questions the audit lists as unmeasured: what a healthy sensor reports in open air (no
pulse? a maximum-length pulse?), the real blind zone (put a flat target at 10, 15, 20, 25, 30 cm),
the beam edge, behaviour on grass / people / pets / low objects, rain, and the cycle timing.

    python scripts/ultrasonic_mcu_probe.py --port /dev/serial/by-id/<mcu data port> \\
        --seconds 60 --csv /tmp/m40_open_air.csv

CSV columns: wall_time_utc, sensor, status, echo_us, distance_cm_at_343mps, health, fails,
cycle, mcu_t_ms. ``distance_cm_at_343mps`` is only a convenience conversion; keep ``echo_us``.
Exit code 0 if readings arrived, 2 if the link never delivered one (see the printed link state).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto  # noqa: E402
from backend.src.drivers.sensors.jsn_sr04t_driver import (  # noqa: E402
    JsnSr04tMcuDriver,
    SerialFactory,
    echo_to_cm,
    open_pyserial,
)
from backend.src.models.hardware_config import UltrasonicConfig  # noqa: E402

CSV_FIELDS = (
    "wall_time_utc", "sensor", "status", "echo_us", "distance_cm_at_343mps", "health", "fails",
    "cycle", "mcu_t_ms",
)  # fmt: skip


def _status_name(code: int) -> str:
    try:
        return proto.EchoStatus(code).name
    except ValueError:
        return f"UNKNOWN_{code}"


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
    cfg = UltrasonicConfig(port=port, echo_window_ms=echo_window_ms, inter_sensor_gap_ms=gap_ms)
    driver = JsnSr04tMcuDriver(
        cfg, serial_factory=serial_factory or open_pyserial, clock=clock, simulate=False
    )
    rows = 0
    with contextlib.ExitStack() as stack:
        writer = None
        if csv_path:
            handle = stack.enter_context(open(csv_path, "w", newline="", encoding="utf-8"))
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
        stack.callback(driver.close)

        def on_reading(r: proto.Reading, wall) -> None:
            nonlocal rows
            rows += 1
            if writer:
                sensor = cfg.channels[r.sensor] if r.sensor < len(cfg.channels) else r.sensor
                approx = f"{echo_to_cm(r.echo_us, 343.0):.1f}" if r.echo_us else ""
                writer.writerow(
                    [wall.isoformat(), sensor, _status_name(r.status), r.echo_us, approx,
                     r.health, r.fails, r.cycle, r.t_ms]
                )  # fmt: skip

        driver.reading_hook = on_reading
        deadline = clock() + seconds
        next_report = clock()
        while clock() < deadline:
            driver.poll()
            if clock() >= next_report:
                next_report = clock() + 1.0
                print(_table(driver), file=out)
            sleep(0.005)
        final_state, final_fault = driver.link_state.value, driver.link_fault
    print(f"link={final_state} fault={final_fault} readings={rows}", file=out)
    return 0 if rows else 2


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
