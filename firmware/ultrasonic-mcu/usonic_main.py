"""Entry point of the ultrasonic MCU firmware (imported by ``code.py``)."""

import usonic_config as cfg
from usonic_core import Core


def build(hal_factory=None):
    """Return ``(hal, core)``.

    While the pins are still TBD (``TRIG_PINS``/``ECHO_PINS`` None) the MCU gets zero sensors: it
    keeps the USB link, the watchdog and the INFO heartbeat alive but fires nothing, and the host
    treats the sensor-count mismatch as a link fault (every channel UNKNOWN).
    """
    if hal_factory is None:
        from usonic_hal_cp import CpHal as hal_factory
    pins_set = cfg.TRIG_PINS is not None and cfg.ECHO_PINS is not None
    hal = hal_factory(
        cfg.TRIG_PINS if pins_set else (),
        cfg.ECHO_PINS if pins_set else (),
        trigger_pulse_us=cfg.TRIGGER_PULSE_US,
        watchdog_s=cfg.WATCHDOG_TIMEOUT_S,
    )
    kwargs = dict(
        fw_version=cfg.FW_VERSION,
        echo_window_ms=cfg.DEFAULT_ECHO_WINDOW_MS,
        gap_ms=cfg.DEFAULT_GAP_MS,
        quiet_after=cfg.DEFAULT_QUIET_AFTER,
        info_period_ms=cfg.INFO_PERIOD_MS,
        partial_timeout_ms=cfg.PARTIAL_FRAME_TIMEOUT_MS,
    )
    try:
        core = Core(hal, hal.sensor_count, firing_order=cfg.FIRING_ORDER, **kwargs)
    except ValueError:
        # A FIRING_ORDER that does not match the sensor count must not stop the MCU.
        core = Core(hal, hal.sensor_count, **kwargs)
    return hal, core


def _reset_after_crash(hal, exc):
    """Say why on the console, then reset the MCU. Never raises; returns only if no reset."""
    try:
        print("usonic: crashed, resetting:", repr(exc))
    except Exception:
        pass
    try:
        hal.reboot()
    except Exception:
        pass


def run(hal_factory=None, iterations=None):
    """Main loop. ``iterations`` is only for tests; on the MCU it runs forever.

    A crash must never leave a silent MCU that is still enumerated on USB. An ``Exception`` out of
    the boot announcement, ``feed_watchdog`` or ``core.poll`` therefore resets the MCU at once
    (``hal.reboot()``): the host sees a new ``boot_id`` and reset reason SOFTWARE and re-handshakes.
    If even that fails, the loop stops feeding the watchdog and spins: the watchdog is armed for as
    long as this function has not returned, so it resets the MCU within ``WATCHDOG_TIMEOUT_S``.
    Not relied on: the watchdog staying armed after ``code.py`` ENDS with an exception, which
    depends on the CircuitPython release (adafruit/circuitpython#7722; unverified for the board).

    ``KeyboardInterrupt``, ``SystemExit`` and the reload exception are not ``Exception`` subclasses
    and propagate: Ctrl-C into the REPL, Ctrl-D and autoreload still stop the program.

    Returns the number of passes made (reachable only with ``iterations``).
    """
    hal, core = build(hal_factory)
    count = 0
    try:
        core.start()
        while iterations is None or count < iterations:
            hal.feed_watchdog()
            core.poll()
            count += 1
    except Exception as exc:
        _reset_after_crash(hal, exc)
        # Reached only if the reset did not happen (on the MCU a reset never returns).
        while iterations is None or count < iterations:
            count += 1
    return count
