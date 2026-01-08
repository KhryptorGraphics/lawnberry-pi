"""HC-SR04 Ultrasonic Sensor Driver for LawnBerry Pi.

Provides async lifecycle and distance readings from 3x HC-SR04 ultrasonic
sensors positioned at front-left, front-center, and front-right.

Safety Requirement (FR-023): Objects detected within 30cm must trigger
immediate obstacle avoidance. This driver only supplies data; enforcement
occurs in safety triggers.

GPIO Pin Assignments (from BEAD-040):
- front_left:   TRIG=GPIO4,  ECHO=GPIO17
- front_center: TRIG=GPIO27, ECHO=GPIO10
- front_right:  TRIG=GPIO11, ECHO=GPIO9

ECHO pins require voltage dividers (5V to 3.3V) - see docs/ultrasonic-wiring-guide.md
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ...core.simulation import is_simulation_mode
from ..base import HardwareDriver

# Lazy import for GPIO libraries
_lgpio = None
_gpiozero = None


def _ensure_gpio():
    """Lazy import GPIO libraries to avoid errors in CI/simulation."""
    global _lgpio, _gpiozero
    if _lgpio is None:
        try:
            import lgpio
            _lgpio = lgpio
        except ImportError:
            pass
    if _gpiozero is None:
        try:
            from gpiozero import DistanceSensor
            _gpiozero = DistanceSensor
        except ImportError:
            pass


@dataclass
class UltrasonicReading:
    """Single ultrasonic sensor reading."""
    sensor_id: str  # "front_left", "front_center", "front_right"
    distance_cm: float  # Distance in centimeters (2-400 typical range)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid: bool = True  # False if reading timed out or out of range


class UltrasonicDriver(HardwareDriver):
    """Driver for 3x HC-SR04 ultrasonic sensors.

    The HC-SR04 works by:
    1. Sending a 10us pulse on TRIG pin
    2. Sensor emits 8 40kHz pulses
    3. ECHO pin goes HIGH when pulse sent, LOW when echo received
    4. Distance = (echo_time * speed_of_sound) / 2

    Range: 2cm - 400cm
    Resolution: ~0.3cm
    Measurement cycle: ~60ms per sensor
    """

    # GPIO pin assignments per BEAD-040 wiring guide
    SENSORS = {
        "front_left": {"trig": 4, "echo": 17},
        "front_center": {"trig": 27, "echo": 10},
        "front_right": {"trig": 11, "echo": 9},
    }

    # Speed of sound in cm/s at 20°C (adjustable for temperature)
    SPEED_OF_SOUND = 34300

    # Timing constants
    TRIG_PULSE_US = 10  # Trigger pulse duration
    TIMEOUT_S = 0.03  # Echo timeout (30ms = ~500cm max)
    MIN_DISTANCE_CM = 2.0
    MAX_DISTANCE_CM = 400.0

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        self._gpio_handle = None
        self._gpiozero_sensors: dict[str, Any] = {}
        self._last_readings: dict[str, UltrasonicReading] = {}
        self._sim_cycle = 0
        self._sim_obstacles: dict[str, float] = {}  # Simulated obstacle distances

    async def initialize(self) -> None:
        """Initialize GPIO pins for ultrasonic sensors."""
        if is_simulation_mode() or os.environ.get("SIM_MODE") == "1":
            # Initialize simulated obstacle positions
            self._sim_obstacles = {
                "front_left": 150.0,
                "front_center": 200.0,
                "front_right": 180.0,
            }
            self.initialized = True
            return

        _ensure_gpio()

        # Try lgpio first (preferred for Pi 5)
        if _lgpio is not None:
            try:
                self._gpio_handle = _lgpio.gpiochip_open(0)
                for sensor_id, pins in self.SENSORS.items():
                    # Set TRIG as output
                    _lgpio.gpio_claim_output(self._gpio_handle, pins["trig"], 0)
                    # Set ECHO as input
                    _lgpio.gpio_claim_input(self._gpio_handle, pins["echo"])
                self.initialized = True
                return
            except Exception as e:
                if self._gpio_handle is not None:
                    try:
                        _lgpio.gpiochip_close(self._gpio_handle)
                    except Exception:
                        pass
                    self._gpio_handle = None

        # Fallback to gpiozero DistanceSensor
        if _gpiozero is not None:
            try:
                for sensor_id, pins in self.SENSORS.items():
                    self._gpiozero_sensors[sensor_id] = _gpiozero(
                        echo=pins["echo"],
                        trigger=pins["trig"],
                        max_distance=4.0,  # 4 meters
                    )
                self.initialized = True
                return
            except Exception as e:
                self._gpiozero_sensors = {}

        # No GPIO library available - allow initialization for testing
        self.initialized = True

    async def start(self) -> None:
        """Start the ultrasonic driver."""
        self.running = True

    async def stop(self) -> None:
        """Stop the ultrasonic driver and release GPIO resources."""
        self.running = False

        # Clean up lgpio
        if self._gpio_handle is not None and _lgpio is not None:
            try:
                for pins in self.SENSORS.values():
                    _lgpio.gpio_free(self._gpio_handle, pins["trig"])
                    _lgpio.gpio_free(self._gpio_handle, pins["echo"])
                _lgpio.gpiochip_close(self._gpio_handle)
            except Exception:
                pass
            self._gpio_handle = None

        # Clean up gpiozero
        for sensor in self._gpiozero_sensors.values():
            try:
                sensor.close()
            except Exception:
                pass
        self._gpiozero_sensors = {}

    async def health_check(self) -> dict[str, Any]:
        """Return driver health status."""
        return {
            "sensor": "ultrasonic_array",
            "sensor_count": len(self.SENSORS),
            "sensors": list(self.SENSORS.keys()),
            "initialized": self.initialized,
            "running": self.running,
            "last_readings": {
                k: {"distance_cm": v.distance_cm, "valid": v.valid}
                for k, v in self._last_readings.items()
            },
            "gpio_backend": self._get_gpio_backend(),
            "simulation": is_simulation_mode() or os.environ.get("SIM_MODE") == "1",
        }

    def _get_gpio_backend(self) -> str:
        """Return which GPIO backend is in use."""
        if is_simulation_mode() or os.environ.get("SIM_MODE") == "1":
            return "simulation"
        if self._gpio_handle is not None:
            return "lgpio"
        if self._gpiozero_sensors:
            return "gpiozero"
        return "none"

    async def read_distance(self, sensor_id: str) -> UltrasonicReading:
        """Read distance from a single ultrasonic sensor.

        Args:
            sensor_id: One of "front_left", "front_center", "front_right"

        Returns:
            UltrasonicReading with distance in cm and validity flag
        """
        if sensor_id not in self.SENSORS:
            return UltrasonicReading(
                sensor_id=sensor_id,
                distance_cm=0.0,
                valid=False,
            )

        if not self.initialized:
            return UltrasonicReading(
                sensor_id=sensor_id,
                distance_cm=0.0,
                valid=False,
            )

        # Simulation mode
        if is_simulation_mode() or os.environ.get("SIM_MODE") == "1":
            reading = self._simulate_reading(sensor_id)
            self._last_readings[sensor_id] = reading
            return reading

        # Real hardware
        reading = await self._read_hardware(sensor_id)
        self._last_readings[sensor_id] = reading
        return reading

    async def read_all(self) -> list[UltrasonicReading]:
        """Read all three ultrasonic sensors.

        Returns:
            List of UltrasonicReading for each sensor
        """
        readings = []
        for sensor_id in self.SENSORS:
            reading = await self.read_distance(sensor_id)
            readings.append(reading)
            # Small delay between readings to avoid interference
            await asyncio.sleep(0.01)
        self._sim_cycle += 1
        return readings

    def _simulate_reading(self, sensor_id: str) -> UltrasonicReading:
        """Generate simulated ultrasonic reading."""
        base_distance = self._sim_obstacles.get(sensor_id, 200.0)

        # Add noise
        noise = random.gauss(0, 2.0)  # ±2cm noise
        distance = base_distance + noise

        # Simulate occasional obstacle approaching
        if self._sim_cycle % 100 == 50 + hash(sensor_id) % 20:
            distance = 25.0 + random.uniform(0, 5)  # Close obstacle

        # Simulate very occasional timeout/invalid reading
        if random.random() < 0.01:
            return UltrasonicReading(
                sensor_id=sensor_id,
                distance_cm=0.0,
                valid=False,
            )

        # Clamp to valid range
        distance = max(self.MIN_DISTANCE_CM, min(self.MAX_DISTANCE_CM, distance))

        return UltrasonicReading(
            sensor_id=sensor_id,
            distance_cm=round(distance, 1),
            valid=True,
        )

    async def _read_hardware(self, sensor_id: str) -> UltrasonicReading:
        """Read from actual HC-SR04 hardware."""
        pins = self.SENSORS[sensor_id]

        # Use gpiozero if available
        if sensor_id in self._gpiozero_sensors:
            try:
                sensor = self._gpiozero_sensors[sensor_id]
                distance_m = sensor.distance
                if distance_m is None:
                    return UltrasonicReading(sensor_id=sensor_id, distance_cm=0.0, valid=False)
                distance_cm = distance_m * 100
                valid = self.MIN_DISTANCE_CM <= distance_cm <= self.MAX_DISTANCE_CM
                return UltrasonicReading(
                    sensor_id=sensor_id,
                    distance_cm=round(distance_cm, 1),
                    valid=valid,
                )
            except Exception:
                return UltrasonicReading(sensor_id=sensor_id, distance_cm=0.0, valid=False)

        # Use lgpio for direct GPIO control
        if self._gpio_handle is not None and _lgpio is not None:
            try:
                trig = pins["trig"]
                echo = pins["echo"]

                # Send trigger pulse
                _lgpio.gpio_write(self._gpio_handle, trig, 1)
                await asyncio.sleep(self.TRIG_PULSE_US / 1_000_000)
                _lgpio.gpio_write(self._gpio_handle, trig, 0)

                # Wait for echo to start (timeout protection)
                start_wait = time.monotonic()
                while _lgpio.gpio_read(self._gpio_handle, echo) == 0:
                    if time.monotonic() - start_wait > self.TIMEOUT_S:
                        return UltrasonicReading(sensor_id=sensor_id, distance_cm=0.0, valid=False)
                    await asyncio.sleep(0.00001)  # 10us poll

                pulse_start = time.monotonic()

                # Wait for echo to end
                while _lgpio.gpio_read(self._gpio_handle, echo) == 1:
                    if time.monotonic() - pulse_start > self.TIMEOUT_S:
                        return UltrasonicReading(sensor_id=sensor_id, distance_cm=0.0, valid=False)
                    await asyncio.sleep(0.00001)

                pulse_end = time.monotonic()

                # Calculate distance
                pulse_duration = pulse_end - pulse_start
                distance_cm = (pulse_duration * self.SPEED_OF_SOUND) / 2

                valid = self.MIN_DISTANCE_CM <= distance_cm <= self.MAX_DISTANCE_CM
                return UltrasonicReading(
                    sensor_id=sensor_id,
                    distance_cm=round(distance_cm, 1),
                    valid=valid,
                )
            except Exception:
                return UltrasonicReading(sensor_id=sensor_id, distance_cm=0.0, valid=False)

        # No GPIO available
        return UltrasonicReading(sensor_id=sensor_id, distance_cm=0.0, valid=False)

    def get_minimum_distance(self) -> float | None:
        """Return the minimum distance from any sensor.

        Returns:
            Minimum distance in cm, or None if no valid readings
        """
        valid_readings = [r for r in self._last_readings.values() if r.valid]
        if not valid_readings:
            return None
        return min(r.distance_cm for r in valid_readings)


__all__ = ["UltrasonicDriver", "UltrasonicReading"]
