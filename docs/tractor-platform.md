# Ride-on Lawn Tractor Platform (Craftsman-class conversion)

The mower is a converted Craftsman-class ride-on lawn tractor — an
Ackermann-steered, gas-engine vehicle — not a differential-drive robot. It is
operated through seven discrete actuators rather than two wheel motors.

## Actuators

| Actuator | Type | Drive | Range / semantics |
|----------|------|-------|-------------------|
| steering | positional | PCA9685 I2C PWM (ch 0) | −1 (full left) .. +1 (full right) |
| throttle | positional | PCA9685 I2C PWM (ch 1) | 0 (engine idle) .. 1 (full RPM) |
| gas pedal (ground speed) | positional | PCA9685 I2C PWM (ch 2) | 0 (stop) .. 1 (full) |
| clutch / brake | positional | PCA9685 I2C PWM (ch 3) | 0 (released/driving) .. 1 (pressed = declutch+brake) |
| gear (F/N/R) | discrete | PCA9685 I2C PWM (ch 4) | forward / neutral / reverse |
| starter | momentary relay | GPIO | engine crank pulse |
| blade (PTO) | latching relay | GPIO | engage / disengage |

Positional actuators are PCA9685 I2C PWM channels (0-indexed, `config/tractor.yaml`'s
`pca9685:` block sets the I2C address/bus/frequency — moved off the PCA9685's 0x40
power-on default, which collides with the INA3221 power monitor). The blade-PTO
relay's GPIO was moved off pin 6 (collides with the ToF Left Interrupt) to GPIO 26.
Both the I2C address and the GPIO choice still need to be run through
`scripts/check_hardware_pin_conflicts.py` once it lands. Calibrate each channel's
`us_min/us_center/us_max` on the bench.

A PCA9685 board holds its last commanded pulse if the I2C bus or host is lost —
it does not fail safe on its own. `PCA9685Driver` propagates any write failure
that occurs after hardware was confirmed present (as opposed to a tolerated
no-op when no board is wired up at all) so a real transport fault is visible to
callers, but a physical bus-fault failsafe (e.g. a watchdog driving the board's
`OE` pin) is a separate, not-yet-built hardware-phase requirement.

## Safety interlocks (standard lawn-tractor, ANSI/OPEI-style)

- **Engine start** only when *authorized + NEUTRAL + clutch/brake pressed +
  blade off*.
- **Blade/PTO** engages only with the *engine running* and *not in reverse*;
  selecting reverse **auto-disengages the blade** (Reverse Operation System).
- **Emergency stop** disengages the blade, shifts to neutral, presses the
  clutch/brake, and idles the throttle/pedal — **the engine keeps running**
  (configurable); authorization is revoked and further commands are blocked
  until cleared.

## Configuration

`config/tractor.yaml` maps actuators to PWM channels / GPIO relays and sets the
interlock flags. Set `enabled: true` on the converted tractor. When enabled, the
autonomous AI loop drives the tractor via `ActionPrediction.to_tractor_command()`
(steering, ground-speed pedal in FORWARD, blade PTO) instead of differential
motors.

## API & UI

- REST: `/api/v2/tractor/{state,steering,throttle,speed,clutch,gear,blade,
  starter,stop-engine,command,authorize,revoke,emergency-stop,clear-emergency}`.
- UI: the **Tractor** dashboard view (`/tractor`).

## Code

- `models/tractor_control.py`, `drivers/actuators/tractor_actuators.py`,
  `services/tractor_service.py`, `api/routers/tractor.py`.
