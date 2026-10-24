"""LawnBerry ultrasonic MCU (four JSN-SR04T) - CircuitPython entry point.

Copy ``boot.py``, ``code.py`` and every ``usonic_*.py`` to the CIRCUITPY drive, set the MCU pin
names in ``usonic_config.py`` (TBD, owner) and power-cycle. See README.md in this directory.
"""

import usonic_main

usonic_main.run()
