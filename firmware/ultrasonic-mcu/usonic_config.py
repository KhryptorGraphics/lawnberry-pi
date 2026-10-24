"""Owner-editable constants for the ultrasonic MCU firmware.

Everything marked TBD is NOT decided by the repository. No Raspberry Pi header pin is involved:
the MCU talks to the Pi over USB serial only. Nothing here has been run on hardware.
"""

# --- MCU GPIO assignment: TBD (owner) --------------------------------------------------------
# One TRIG and one ECHO board-pin name per sensor, in sensor-index order, as attributes of
# CircuitPython's ``board`` module (for example "GP2"). The repository assigns NO pins and does not
# know how the MCU will be wired. While either is None the firmware fires nothing and reports 0
# sensors, which the host treats as a link fault (all channels UNKNOWN).
#
# Example shape only (NOT an assignment, NOT verified):
#   TRIG_PINS = ("GP2", "GP4", "GP6", "GP8")
#   ECHO_PINS = ("GP3", "GP5", "GP7", "GP9")
#
# JSN-SR04T ECHO is a 5 V-class output on many boards; whether the RP2040's 3.3 V inputs (not 5 V
# tolerant) need a level shifter or divider depends on the fitted module revision and supply:
# TBD, audit M-40 (measure ECHO high level at the real supply with the MCU disconnected).
TRIG_PINS = None
ECHO_PINS = None

# --- Firing -----------------------------------------------------------------------------------
# Order in which the sensors are triggered (time-division: never two at once). TBD with the
# physical mounting: consecutive entries should not be neighbours on the deck if that can be
# avoided. The default is simply the index order.
FIRING_ORDER = (0, 1, 2, 3)

# Defaults used until the host sends CONFIG, and again after every reset. TBD, audit M-40.
DEFAULT_ECHO_WINDOW_MS = 30  # time after TRIG during which an echo is accepted
DEFAULT_GAP_MS = 30  # quiet time between the end of one measurement and the next trigger
DEFAULT_QUIET_AFTER = 8  # consecutive silent measurements before a sensor is reported QUIET

TRIGGER_PULSE_US = 10  # TRIG high time (HC-SR04 style); TBD for the fitted JSN-SR04T revision
INFO_PERIOD_MS = 1000  # heartbeat: INFO frame at least this often
PARTIAL_FRAME_TIMEOUT_MS = 50  # abandon a half-received host frame after this long
WATCHDOG_TIMEOUT_S = 4  # hardware watchdog (reset); the main loop feeds it every iteration

FW_VERSION = (1, 0)
