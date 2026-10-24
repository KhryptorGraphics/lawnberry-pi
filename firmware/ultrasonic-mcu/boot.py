"""CircuitPython boot configuration for the ultrasonic MCU.

Enables a second USB serial port (``usb_cdc.data``) that carries ONLY the framed protocol. The
REPL console stays on the first port. Frames contain arbitrary bytes, including 0x03 and 0x04,
which the console would interpret as Ctrl-C / Ctrl-D and use to interrupt or reset the firmware,
so the host must never talk to the console port. ``usb_cdc.enable`` takes effect after a power
cycle.
"""

import usb_cdc

usb_cdc.enable(console=True, data=True)

try:
    import supervisor

    # Makes the data port recognisable under /dev/serial/by-id (product string below).
    supervisor.set_usb_identification("LawnBerry", "LawnBerry Ultrasonic MCU")
except Exception:
    pass
