"""Hardware self-test utilities for on-device validation.

Designed to be safe on CI and devices without hardware enabled:
- Imports optional deps (smbus2, pyserial) lazily inside functions
- Catches permission and missing-device errors
- Returns a structured report instead of raising
"""

from __future__ import annotations

import os
import time
import json
import grp
from typing import Dict, Any, List


EXPECTED_I2C = {
    "bme280": [0x76, 0x77],
    "ina3221": [0x40, 0x41],
    "vl53l0x": [0x29, 0x30],
    "bno085": [0x4A, 0x4B],  # BNO085 IMU
}

SERIAL_CANDIDATES = [
    "/dev/ttyUSB0",
    "/dev/ttyAMA0",  # Pi 5 primary UART for GPS
    "/dev/ttyS0",
    "/dev/serial0",
]

# GPS module candidates (LC29H typically on ttyAMA0 at 115200)
GPS_SERIAL_CONFIGS = [
    {"device": "/dev/ttyAMA0", "baudrate": 115200},  # LC29H default
    {"device": "/dev/ttyUSB0", "baudrate": 115200},  # USB GPS fallback
    {"device": "/dev/ttyACM0", "baudrate": 115200},  # ZED-F9P USB
]

# Ultrasonic GPIO pins (HC-SR04 array)
ULTRASONIC_GPIO_PINS = {
    "front_left": {"trig": 4, "echo": 17},
    "front_center": {"trig": 27, "echo": 10},
    "front_right": {"trig": 11, "echo": 9},
}

# Camera device candidates
CAMERA_DEVICES = {
    "stereo": ["/dev/video0", "/dev/video2"],  # ELP USB stereo camera
    "picamera": None,  # Detected via libcamera
}


def _group_names() -> List[str]:
    try:
        gids = os.getgroups()
        names = []
        for g in gids:
            try:
                names.append(grp.getgrgid(g).gr_name)
            except KeyError:
                continue
        return names
    except Exception:
        return []


def i2c_probe(bus_num: int = 1) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "available": False,
        "bus": f"/dev/i2c-{bus_num}",
        "error": None,
        "present": {},
    }
    dev_path = f"/dev/i2c-{bus_num}"
    if not os.path.exists(dev_path):
        report["error"] = f"missing {dev_path}"
        return report

    try:
        from smbus2 import SMBus  # type: ignore
    except Exception as e:  # ImportError or others
        report["error"] = f"smbus2 unavailable: {e}"
        return report

    try:
        with SMBus(bus_num) as bus:
            report["available"] = True
            # Probe only expected addresses to keep it fast/safe
            present: Dict[str, List[str]] = {}
            for name, addrs in EXPECTED_I2C.items():
                found: List[str] = []
                for addr in addrs:
                    try:
                        # Use read_byte to probe; many devices NACK -> catch
                        bus.read_byte(addr)
                        found.append(hex(addr))
                    except Exception:
                        # Not present or no permission
                        continue
                if found:
                    present[name] = found
            report["present"] = present
    except Exception as e:
        report["error"] = str(e)

    return report


def serial_probe(paths: List[str] | None = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "available": False,
        "candidates": [],
        "opened": None,
        "error": None,
    }
    paths = paths or SERIAL_CANDIDATES
    existing = [p for p in paths if os.path.exists(p)]
    report["candidates"] = existing
    if not existing:
        return report

    try:
        import serial  # type: ignore
    except Exception as e:
        report["error"] = f"pyserial unavailable: {e}"
        return report

    for dev in existing:
        try:
            with serial.Serial(dev, baudrate=9600, timeout=0.2) as ser:  # type: ignore
                # Non-blocking peek
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                _ = ser.read(16)
                report["available"] = True
                report["opened"] = dev
                break
        except Exception:
            continue
    return report


def gps_probe() -> Dict[str, Any]:
    """Probe for LC29H or other GPS module via serial, checking for NMEA output."""
    report: Dict[str, Any] = {
        "available": False,
        "device": None,
        "baudrate": None,
        "nmea_received": False,
        "fix_quality": None,
        "error": None,
    }

    try:
        import serial  # type: ignore
    except Exception as e:
        report["error"] = f"pyserial unavailable: {e}"
        return report

    for cfg in GPS_SERIAL_CONFIGS:
        device = cfg["device"]
        baud = cfg["baudrate"]
        if not os.path.exists(device):
            continue

        try:
            with serial.Serial(device, baudrate=baud, timeout=1.5) as ser:
                ser.reset_input_buffer()
                # Read multiple lines to find NMEA sentence
                data = ser.read(1024).decode("ascii", errors="ignore")
                if "$G" in data:  # NMEA sentences start with $G (GNGGA, GNRMC, etc.)
                    report["available"] = True
                    report["device"] = device
                    report["baudrate"] = baud
                    report["nmea_received"] = True
                    # Try to extract fix quality from GGA sentence
                    for line in data.split("\n"):
                        if "GGA" in line:
                            parts = line.split(",")
                            if len(parts) > 6:
                                try:
                                    report["fix_quality"] = int(parts[6])
                                except ValueError:
                                    pass
                            break
                    break
        except Exception as e:
            continue

    return report


def ultrasonic_probe() -> Dict[str, Any]:
    """Test ultrasonic GPIO pin accessibility (does not require actual sensors)."""
    report: Dict[str, Any] = {
        "available": False,
        "gpio_accessible": False,
        "sensors": {},
        "error": None,
    }

    # Check if GPIO is accessible
    gpio_paths = ["/dev/gpiochip0", "/dev/gpiochip4"]  # Pi 5 uses gpiochip4
    gpio_found = any(os.path.exists(p) for p in gpio_paths)
    if not gpio_found:
        report["error"] = "No GPIO chip found"
        return report

    report["gpio_accessible"] = True

    # Try to import lgpio for Pi 5
    try:
        import lgpio  # type: ignore
        chip = lgpio.gpiochip_open(4)  # Pi 5 uses chip 4
        report["available"] = True

        for sensor_name, pins in ULTRASONIC_GPIO_PINS.items():
            try:
                # Just claim and release to test accessibility
                lgpio.gpio_claim_output(chip, pins["trig"])
                lgpio.gpio_claim_input(chip, pins["echo"])
                lgpio.gpio_write(chip, pins["trig"], 0)
                lgpio.gpio_free(chip, pins["trig"])
                lgpio.gpio_free(chip, pins["echo"])
                report["sensors"][sensor_name] = {"trig": pins["trig"], "echo": pins["echo"], "accessible": True}
            except Exception:
                report["sensors"][sensor_name] = {"trig": pins["trig"], "echo": pins["echo"], "accessible": False}

        lgpio.gpiochip_close(chip)
    except Exception as e:
        # Fallback: check if gpiozero works
        try:
            from gpiozero import Device, DigitalOutputDevice  # type: ignore
            from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore
            Device.pin_factory = LGPIOFactory()
            report["available"] = True
            for sensor_name, pins in ULTRASONIC_GPIO_PINS.items():
                report["sensors"][sensor_name] = {"trig": pins["trig"], "echo": pins["echo"], "accessible": "gpiozero"}
        except Exception as e2:
            report["error"] = f"lgpio: {e}, gpiozero: {e2}"

    return report


def stereo_camera_probe() -> Dict[str, Any]:
    """Probe for ELP USB stereo camera."""
    report: Dict[str, Any] = {
        "available": False,
        "device": None,
        "resolution": None,
        "error": None,
    }

    try:
        import cv2  # type: ignore
    except Exception as e:
        report["error"] = f"OpenCV unavailable: {e}"
        return report

    for device_path in CAMERA_DEVICES["stereo"]:
        if not os.path.exists(device_path):
            continue

        try:
            # Extract device index from path
            idx = int(device_path.split("video")[-1])
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Set stereo resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    if w >= 2560:  # Stereo camera has combined 2560x960
                        report["available"] = True
                        report["device"] = device_path
                        report["resolution"] = f"{w}x{h}"
                cap.release()
                if report["available"]:
                    break
        except Exception:
            continue

    return report


def picamera_probe() -> Dict[str, Any]:
    """Probe for Raspberry Pi Camera 2 via libcamera."""
    report: Dict[str, Any] = {
        "available": False,
        "model": None,
        "resolution": None,
        "error": None,
    }

    # Check via rpicam-hello output
    import subprocess
    try:
        result = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout + result.stderr
        if "imx219" in output.lower() or "imx477" in output.lower() or "ov5647" in output.lower():
            report["available"] = True
            # Extract model
            for model in ["imx219", "imx477", "ov5647"]:
                if model in output.lower():
                    report["model"] = model
                    break
            # Try to extract resolution
            if "3280x2464" in output:
                report["resolution"] = "3280x2464"
            elif "1920x1080" in output:
                report["resolution"] = "1920x1080"
    except FileNotFoundError:
        report["error"] = "rpicam-hello not found"
    except subprocess.TimeoutExpired:
        report["error"] = "rpicam-hello timeout"
    except Exception as e:
        report["error"] = str(e)

    return report


def run_selftest() -> Dict[str, Any]:
    """Run comprehensive hardware self-test for all LawnBerry Pi components."""
    groups = _group_names()
    i2c = i2c_probe(bus_num=1)
    serial = serial_probe()
    gps = gps_probe()
    ultrasonic = ultrasonic_probe()
    stereo = stereo_camera_probe()
    picamera = picamera_probe()

    summary = {
        # I2C sensors
        "i2c_bus_present": i2c.get("available", False),
        "bme280_present": bool(i2c.get("present", {}).get("bme280")),
        "ina3221_present": bool(i2c.get("present", {}).get("ina3221")),
        "vl53l0x_present": bool(i2c.get("present", {}).get("vl53l0x")),
        "bno085_present": bool(i2c.get("present", {}).get("bno085")),
        # Serial/UART
        "serial_port_present": bool(serial.get("candidates")),
        "serial_open_ok": bool(serial.get("opened")),
        # GPS (LC29H)
        "gps_present": gps.get("available", False),
        "gps_device": gps.get("device"),
        "gps_nmea_received": gps.get("nmea_received", False),
        "gps_fix_quality": gps.get("fix_quality"),
        # Ultrasonic (HC-SR04 array)
        "ultrasonic_gpio_accessible": ultrasonic.get("gpio_accessible", False),
        "ultrasonic_sensors_ok": ultrasonic.get("available", False),
        # Cameras
        "stereo_camera_present": stereo.get("available", False),
        "stereo_camera_device": stereo.get("device"),
        "picamera_present": picamera.get("available", False),
        "picamera_model": picamera.get("model"),
        # Permissions
        "groups": groups,
        "needs_i2c_group": ("i2c" not in groups),
        "needs_dialout_group": ("dialout" not in groups),
        "needs_video_group": ("video" not in groups),
    }

    # Overall health: at minimum need either GPS or I2C sensors working
    overall_ok = (
        (summary["i2c_bus_present"] and (
            summary["bme280_present"] or
            summary["ina3221_present"] or
            summary["vl53l0x_present"] or
            summary["bno085_present"]
        )) or
        summary["gps_present"] or
        summary["serial_open_ok"]
    )

    return {
        "i2c": i2c,
        "serial": serial,
        "gps": gps,
        "ultrasonic": ultrasonic,
        "stereo_camera": stereo,
        "picamera": picamera,
        "summary": {
            **summary,
            "overall_ok": overall_ok,
        },
    }
