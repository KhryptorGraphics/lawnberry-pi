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


def run(hal_factory=None, iterations=None):
    """Main loop. ``iterations`` is only for tests; on the MCU it runs forever."""
    hal, core = build(hal_factory)
    core.start()
    count = 0
    while iterations is None or count < iterations:
        hal.feed_watchdog()
        core.poll()
        count += 1
