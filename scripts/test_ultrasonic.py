#!/usr/bin/env python3
"""Ultrasonic Sensor Test Script (BEAD-042).

Interactive test script for HC-SR04 ultrasonic sensors.
Tests each sensor individually and displays real-time readings.

Usage:
    # Simulation mode (no hardware required)
    SIM_MODE=true python scripts/test_ultrasonic.py

    # Hardware mode (requires wiring per docs/ultrasonic-wiring-guide.md)
    python scripts/test_ultrasonic.py

    # Single sensor test
    python scripts/test_ultrasonic.py --sensor front_center

    # Continuous monitoring
    python scripts/test_ultrasonic.py --continuous --interval 0.2
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.src.drivers.sensors.ultrasonic_driver import UltrasonicDriver, UltrasonicReading


def print_header():
    """Print test script header."""
    print("=" * 60)
    print("HC-SR04 Ultrasonic Sensor Test Script")
    print("=" * 60)
    sim_mode = os.environ.get("SIM_MODE") == "1" or os.environ.get("SIM_MODE") == "true"
    print(f"Mode: {'SIMULATION' if sim_mode else 'HARDWARE'}")
    print("-" * 60)


def print_reading(reading: UltrasonicReading, show_warning: bool = True):
    """Print a single sensor reading with color coding."""
    if not reading.valid:
        status = "\033[91mINVALID\033[0m"  # Red
        distance_str = "---"
    elif reading.distance_cm < 30:
        status = "\033[91mDANGER\033[0m"   # Red - very close
        distance_str = f"{reading.distance_cm:6.1f} cm"
    elif reading.distance_cm < 100:
        status = "\033[93mWARNING\033[0m"  # Yellow - close
        distance_str = f"{reading.distance_cm:6.1f} cm"
    else:
        status = "\033[92mOK\033[0m"       # Green - safe
        distance_str = f"{reading.distance_cm:6.1f} cm"

    sensor_name = reading.sensor_id.replace("_", " ").title()
    print(f"  {sensor_name:15} | {distance_str:12} | {status}")


async def test_single_sensor(driver: UltrasonicDriver, sensor_id: str, count: int = 10):
    """Test a single sensor with multiple readings."""
    print(f"\nTesting sensor: {sensor_id}")
    print("-" * 40)

    valid_count = 0
    distances = []

    for i in range(count):
        reading = await driver.read_distance(sensor_id)
        print_reading(reading)

        if reading.valid:
            valid_count += 1
            distances.append(reading.distance_cm)

        await asyncio.sleep(0.1)

    print("-" * 40)
    print(f"Valid readings: {valid_count}/{count}")
    if distances:
        print(f"Min distance: {min(distances):.1f} cm")
        print(f"Max distance: {max(distances):.1f} cm")
        print(f"Avg distance: {sum(distances)/len(distances):.1f} cm")


async def test_all_sensors(driver: UltrasonicDriver, count: int = 10):
    """Test all three sensors."""
    print("\nTesting all sensors:")
    print("-" * 60)

    for i in range(count):
        print(f"\nReading #{i+1}:")
        readings = await driver.read_all()
        for reading in readings:
            print_reading(reading)
        await asyncio.sleep(0.2)


async def continuous_monitoring(driver: UltrasonicDriver, interval: float = 0.5):
    """Continuous real-time monitoring of all sensors."""
    print("\nContinuous Monitoring (Ctrl+C to stop)")
    print("-" * 60)

    try:
        while True:
            readings = await driver.read_all()

            # Clear line and print readings
            print("\r", end="")
            for reading in readings:
                name = reading.sensor_id.split("_")[1][0].upper()  # L, C, R
                if not reading.valid:
                    print(f"{name}:--- ", end="")
                elif reading.distance_cm < 30:
                    print(f"\033[91m{name}:{reading.distance_cm:3.0f}\033[0m ", end="")
                elif reading.distance_cm < 100:
                    print(f"\033[93m{name}:{reading.distance_cm:3.0f}\033[0m ", end="")
                else:
                    print(f"\033[92m{name}:{reading.distance_cm:3.0f}\033[0m ", end="")

            # Show minimum distance
            min_dist = driver.get_minimum_distance()
            if min_dist is not None:
                print(f"| Min:{min_dist:5.1f}cm ", end="")

            sys.stdout.flush()
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


async def run_health_check(driver: UltrasonicDriver):
    """Display driver health information."""
    health = await driver.health_check()

    print("\nDriver Health Check:")
    print("-" * 40)
    print(f"  Sensor Type: {health['sensor']}")
    print(f"  Sensor Count: {health['sensor_count']}")
    print(f"  Initialized: {health['initialized']}")
    print(f"  Running: {health['running']}")
    print(f"  GPIO Backend: {health['gpio_backend']}")
    print(f"  Simulation: {health['simulation']}")

    if health.get('last_readings'):
        print("\n  Last Readings:")
        for sensor_id, data in health['last_readings'].items():
            valid = "Valid" if data['valid'] else "Invalid"
            print(f"    {sensor_id}: {data['distance_cm']:.1f} cm ({valid})")


async def verify_voltage_dividers(driver: UltrasonicDriver):
    """Quick test to verify ECHO voltage dividers are working.

    If voltage dividers are missing/incorrect, readings will be erratic
    or always show max/min values.
    """
    print("\nVoltage Divider Verification:")
    print("-" * 60)
    print("Testing for stable readings (erratic values may indicate")
    print("missing or incorrect voltage dividers on ECHO pins).")
    print()

    for sensor_id in ["front_left", "front_center", "front_right"]:
        readings = []
        for _ in range(20):
            reading = await driver.read_distance(sensor_id)
            if reading.valid:
                readings.append(reading.distance_cm)
            await asyncio.sleep(0.05)

        if len(readings) < 5:
            print(f"  {sensor_id}: \033[91mFAILED\033[0m - Too few valid readings")
            print("    Check: VCC, GND, TRIG connections")
            continue

        variance = max(readings) - min(readings)
        avg = sum(readings) / len(readings)

        if variance > avg * 0.5:  # More than 50% variance
            print(f"  {sensor_id}: \033[93mWARNING\033[0m - High variance ({variance:.1f}cm)")
            print("    Possible issue with voltage divider or interference")
        elif all(r > 390 for r in readings) or all(r < 5 for r in readings):
            print(f"  {sensor_id}: \033[91mFAILED\033[0m - Readings stuck at limit")
            print("    Check: ECHO voltage divider circuit")
        else:
            print(f"  {sensor_id}: \033[92mPASS\033[0m - Readings stable ({avg:.1f}cm avg)")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test HC-SR04 ultrasonic sensors")
    parser.add_argument("--sensor", type=str, help="Test specific sensor (front_left/center/right)")
    parser.add_argument("--count", type=int, default=10, help="Number of readings (default: 10)")
    parser.add_argument("--continuous", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=float, default=0.5, help="Read interval in seconds")
    parser.add_argument("--health", action="store_true", help="Show health check only")
    parser.add_argument("--verify", action="store_true", help="Verify voltage dividers")
    args = parser.parse_args()

    print_header()

    # Initialize driver
    driver = UltrasonicDriver()
    await driver.initialize()
    await driver.start()

    try:
        if args.health:
            await run_health_check(driver)
        elif args.verify:
            await verify_voltage_dividers(driver)
        elif args.continuous:
            await continuous_monitoring(driver, args.interval)
        elif args.sensor:
            if args.sensor not in UltrasonicDriver.SENSORS:
                print(f"Invalid sensor: {args.sensor}")
                print(f"Valid options: {list(UltrasonicDriver.SENSORS.keys())}")
                return 1
            await test_single_sensor(driver, args.sensor, args.count)
        else:
            await test_all_sensors(driver, args.count)
            await run_health_check(driver)

    finally:
        await driver.stop()

    print("\nTest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
