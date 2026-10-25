from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Ina3221Config(BaseModel):
    address: int | None = Field(default=None, description="I2C address override for INA3221")
    bus: int | None = Field(default=None, description="I2C bus override for INA3221")
    shunt_ohms_ch1: float | None = Field(
        default=None, description="Override shunt resistance for channel 1 (solar)"
    )
    shunt_ohms_ch2: float | None = Field(
        default=None, description="Override shunt resistance for channel 2"
    )
    shunt_ohms_ch3: float | None = Field(
        default=None, description="Override shunt resistance for channel 3 (battery)"
    )
    shunt_spec_ch1: str | None = Field(
        default=None, description="Channel 1 shunt spec, e.g. '30A/75mV'"
    )
    shunt_spec_ch2: str | None = Field(
        default=None, description="Channel 2 shunt spec, e.g. '5A/50mV'"
    )
    shunt_spec_ch3: str | None = Field(
        default=None, description="Channel 3 shunt spec, e.g. '50A/75mV'"
    )


class VictronBleConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Victron SmartSolar BLE reader")
    device_id: str | None = Field(
        default=None, description="Victron identifier or MAC address for BLE connection"
    )
    device_key: str | None = Field(
        default=None,
        description="Optional combined '<address>@<key>' string accepted by victron-ble CLI",
    )
    encryption_key: str | None = Field(
        default=None,
        description="Victron Instant Readout encryption key; combined with device_id when device_key not provided",  # noqa: E501
    )
    cli_path: str = Field(default="victron-ble", description="victron-ble CLI executable path")
    adapter: str | None = Field(
        default=None, description="Optional BLE adapter (passed as --adapter)"
    )
    read_timeout_s: float = Field(
        default=8.0, description="Timeout for BLE read command in seconds"
    )
    prefer_battery: bool = Field(
        default=False,
        description="Prefer Victron telemetry for battery voltage/current/power when merging with INA3221",  # noqa: E501
    )
    prefer_solar: bool = Field(
        default=False,
        description="Prefer Victron telemetry for solar metrics when merging with INA3221",
    )
    prefer_load: bool = Field(
        default=False,
        description="Prefer Victron telemetry for load current when merging with INA3221",
    )
    sample_limit: int = Field(
        default=1,
        ge=1,
        description="Number of victron-ble frames to collect per poll before returning the most recent",  # noqa: E501
    )


class GPSType(StrEnum):
    ZED_F9P_USB = "zed-f9p-usb"
    ZED_F9P_UART = "zed-f9p-uart"
    NEO_8M_UART = "neo-8m-uart"
    LC29H_DA = "lc29h-da"  # Quectel LC29H(DA) via UART with RTK support


class IMUType(StrEnum):
    BNO085_UART = "bno085-uart"


class MotorControllerType(StrEnum):
    ROBOHAT_RP2040 = "robohat-rp2040"
    L298N = "l298n"


class BladeControllerType(StrEnum):
    ROBOHAT_RP2040 = "robohat-rp2040"
    IBT_4 = "ibt-4"


class Ibt4Pins(BaseModel):
    """BCM GPIO numbers for the IBT-4 inputs (``cutting_system.blade_driver`` in the spec)."""

    in1: int = Field(default=24, ge=0, le=27, description="BCM GPIO wired to IBT-4 IN1")
    in2: int = Field(
        default=25,
        ge=0,
        le=27,
        description=(
            "BCM GPIO wired to IBT-4 IN2. GPIO25 is the HaLow HAT SPI IRQ on the deployed Pi "
            "(C-03, owner action O-5): the kernel holds it, so a claim fails until it is moved"
        ),
    )


class Ibt4Levels(BaseModel):
    """Logic level each IBT-4 input is driven to when the blade is ON (OFF is all inputs low).

    Exactly ONE input must be high: (0, 0) equals OFF and (1, 1) is the brake state of the usual
    half-bridge input scheme, so neither can mean "blade on". OFF (both low) lets the blade
    COAST down on such a bridge; whether a brake is needed is an owner decision.

    UNVERIFIED: the IBT-4 truth table (which input must be high for the blade direction, whether
    PWM or an enable line is needed) has not been read from the fitted board or measured.
    """

    in1: Literal[0, 1] = 1
    in2: Literal[0, 1] = 0

    @model_validator(mode="after")
    def _exactly_one_high(self):
        if self.in1 == self.in2:
            raise ValueError(
                f"ibt4.on_levels in1={self.in1}, in2={self.in2}: exactly one input must be high "
                "(both low is the OFF state, both high is the brake state of a half-bridge; "
                "the fitted board's truth table is unmeasured)"
            )
        return self


class Ibt4Config(BaseModel):
    """IBT-4 blade driver (differential-drive mower platform), ``ibt4:`` in hardware.yaml.

    ``gpio_enabled`` is the arming gate for the real ``lgpio`` implementation. It defaults to
    ``False``: the driver is then a flag-only stub and the E-stop reports the blade output as
    ``unconfirmed``. Turn it on only after the pins are approved (O-5) and the truth table and the
    pin state at process exit are measured. Nothing here has been run on hardware.

    ``deadman_ms`` is the blade ON deadman: a blade ON command holds the blade on for this long,
    and the blade is switched OFF (confirmed by readback) unless the command is repeated before
    it expires. A dead client, a lost connection or a stopped operator UI therefore cannot leave
    the blade running; a client that wants it to keep running re-sends ON about every
    ``deadman_ms / 3``.
    """

    gpio_enabled: bool = Field(default=False)
    pins: Ibt4Pins = Field(default_factory=Ibt4Pins)
    on_levels: Ibt4Levels = Field(default_factory=Ibt4Levels)
    deadman_ms: int = Field(
        default=2000,
        ge=200,
        le=10_000,
        description="Blade ON lasts this long unless repeated; then it is driven OFF",
    )

    @model_validator(mode="after")
    def _distinct_pins(self):
        if self.pins.in1 == self.pins.in2:
            raise ValueError("ibt4.pins.in1 and ibt4.pins.in2 must be different GPIOs")
        return self


class UltrasonicConfig(BaseModel):
    """Four JSN-SR04T behind a dedicated USB microcontroller, ``ultrasonic:`` in hardware.yaml.

    The MCU (``firmware/ultrasonic-mcu/``) owns the TRIG/ECHO timing and talks a framed binary
    protocol over USB serial, so no Raspberry Pi header pin is used. The master switch stays
    ``ultrasonic_enabled`` in ``config/limits.yaml`` (shipped ``false``).

    EVERY numeric threshold below is an UNVALIDATED placeholder until the JSN-SR04T measurements
    (audit M-40: revision and mode, blind zone, beam pattern, range on grass/people/pets, rain)
    are done. The two ``*_is_clear`` policy flags default to the conservative setting: a silent
    or out-of-range channel is UNKNOWN (not clear) unless the owner proves otherwise.
    """

    backend: Literal["mcu_usb", "gpio_hcsr04"] = Field(
        default="mcu_usb",
        description="mcu_usb = four JSN-SR04T via the USB MCU; gpio_hcsr04 = legacy 3x HC-SR04",
    )
    port: str | None = Field(
        default=None,
        description=(
            "USB serial device of the MCU *data* channel (TBD for the owner: prefer a "
            "/dev/serial/by-id path). Never the REPL console port: control bytes in a frame "
            "would interrupt the firmware."
        ),
    )
    baudrate: int = Field(
        default=115200, ge=1200, description="Ignored by USB CDC; kept for pyserial"
    )
    channels: list[str] = Field(
        default_factory=lambda: ["outer_left", "inner_left", "inner_right", "outer_right"],
        description=(
            "Channel labels in MCU sensor order. Labels only: sensor positions, yaw and pitch "
            "are TBD (mount design, M-01)."
        ),
    )
    disabled_channels: list[str] = Field(
        default_factory=list, description="Channels masked off (not fitted); excluded from verdicts"
    )
    echo_window_ms: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Per-sensor window after TRIG for the echo (TBD, M-40)",
    )
    inter_sensor_gap_ms: int = Field(
        default=30, ge=5, le=500, description="Quiet time between two sensors firing (crosstalk)"
    )
    speed_of_sound_mps: float = Field(default=343.2, gt=300.0, lt=370.0)
    min_range_cm: float = Field(
        default=25.0,
        gt=0.0,
        description="Blind zone: an echo closer than this is UNKNOWN (about 25 cm discussed, M-40)",
    )
    max_range_cm: float = Field(
        default=400.0, gt=0.0, description="Longest usable echo (TBD, M-40)"
    )
    stale_after_ms: int = Field(
        default=700, ge=50, description="A channel not refreshed for this long is UNKNOWN"
    )
    link_timeout_ms: int = Field(
        default=1500, ge=100, description="No valid frame from the MCU for this long: link down"
    )
    handshake_timeout_s: float = Field(default=2.0, gt=0.0)
    alive_window_s: float = Field(
        default=10.0,
        gt=0.0,
        description="A channel counts as alive if it produced any echo pulse within this window",
    )
    out_of_range_is_clear: bool = Field(
        default=False,
        description="Treat an echo beyond max range from an alive channel as CLEAR (needs M-40)",
    )
    no_echo_is_clear: bool = Field(
        default=False,
        description=(
            "Treat a silent channel (no echo pulse at all) as CLEAR. A silent line is the same "
            "for open air and for a dead sensor; needs M-40 and independent redundancy"
        ),
    )
    no_echo_requires_alive: bool = Field(
        default=True,
        description=(
            "With no_echo_is_clear, still require a real echo pulse within alive_window_s, so a "
            "sensor that dies stops reading clear. Set false only if M-40 shows a healthy sensor "
            "emits no pulse in open air AND something else covers a dead sensor"
        ),
    )

    @field_validator("channels")
    @classmethod
    def _channels_valid(cls, v: list[str]) -> list[str]:
        if not 1 <= len(v) <= 8:
            raise ValueError("ultrasonic.channels needs 1 to 8 entries")
        if len(set(v)) != len(v):
            raise ValueError("ultrasonic.channels must be unique")
        for name in v:
            if not name or not all(c.isalnum() or c == "_" for c in name):
                raise ValueError(f"invalid ultrasonic channel label {name!r}")
        return v

    @model_validator(mode="after")
    def _consistent(self):
        unknown = set(self.disabled_channels) - set(self.channels)
        if unknown:
            raise ValueError(f"ultrasonic.disabled_channels not in channels: {sorted(unknown)}")
        if self.min_range_cm >= self.max_range_cm:
            raise ValueError("ultrasonic.min_range_cm must be below max_range_cm")
        longest_echo_us = 2.0 * (self.max_range_cm / 100.0) / self.speed_of_sound_mps * 1e6
        if self.echo_window_ms * 1000 < longest_echo_us:
            raise ValueError(
                f"ultrasonic.echo_window_ms={self.echo_window_ms} is shorter than the echo of "
                f"max_range_cm={self.max_range_cm} ({longest_echo_us / 1000:.1f} ms)"
            )
        return self


class HardwareConfig(BaseModel):
    """Declares physical hardware modules present in the system.

    Loaded from config/hardware.yaml at startup (FR-003).
    """

    gps_type: GPSType | None = Field(default=None)
    gps_ntrip_enabled: bool = Field(default=False)
    imu_type: IMUType | None = Field(default=None)
    tof_sensors: list[str] = Field(
        default_factory=list, description="ToF sensor positions e.g. ['left','right']"
    )
    env_sensor: bool = Field(default=False, description="BME280 present")
    power_monitor: bool = Field(default=False, description="INA3221 present")
    motor_controller: MotorControllerType | None = Field(default=None)
    blade_controller: BladeControllerType | None = Field(default=None)
    camera_enabled: bool = Field(default=False)
    # Optional typed ToF configuration (sensors.tof_config in hardware.yaml)
    tof_config: ToFConfig | None = Field(default=None)
    # Optional INA3221 configuration overrides (ina3221 block in hardware.yaml)
    ina3221_config: Ina3221Config | None = Field(default=None)
    # Optional Victron SmartSolar BLE configuration
    victron_config: VictronBleConfig | None = Field(default=None)
    # Optional IBT-4 blade driver configuration (ibt4 block in hardware.yaml)
    ibt4_config: Ibt4Config | None = Field(default=None)
    # Optional JSN-SR04T-over-MCU ultrasonic configuration (ultrasonic block in hardware.yaml)
    ultrasonic_config: UltrasonicConfig | None = Field(default=None)

    @field_validator("gps_ntrip_enabled")
    @classmethod
    def ntrip_requires_rtk_gps(cls, v: bool, info):
        if v:
            gps_type = info.data.get("gps_type")
            rtk_capable = {GPSType.ZED_F9P_USB, GPSType.ZED_F9P_UART, GPSType.LC29H_DA}
            if gps_type not in rtk_capable:
                raise ValueError("NTRIP corrections require RTK-capable GPS (ZED-F9P or LC29H-DA)")
        return v

    @field_validator("motor_controller")
    @classmethod
    def motor_controller_required(cls, v: MotorControllerType | None):
        # System requires a motor controller for motion; allow None for SIM_MODE setups.
        return v


class ToFConfig(BaseModel):
    bus: int | None = Field(default=1)
    left_address: int | None = Field(default=0x29)
    right_address: int | None = Field(default=0x30)
    ranging_mode: str | None = Field(default="better_accuracy")
    left_shutdown_gpio: int | None = Field(default=None)
    right_shutdown_gpio: int | None = Field(default=None)
    timing_budget_us: int | None = Field(default=None)
