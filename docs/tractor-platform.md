# Ride-on Lawn Tractor Platform (Craftsman-class conversion)

The mower is a converted Craftsman-class ride-on lawn tractor — an
Ackermann-steered, gas-engine vehicle — not a differential-drive robot. It is
operated through seven discrete actuators rather than two wheel motors.

## Actuators

| Actuator | Type | Drive | Range / semantics |
|----------|------|-------|-------------------|
| steering | positional | RoboHAT RC-PWM | −1 (full left) .. +1 (full right) |
| throttle | positional | RoboHAT RC-PWM | 0 (engine idle) .. 1 (full RPM) |
| gas pedal (ground speed) | positional | RoboHAT RC-PWM | 0 (stop) .. 1 (full) |
| clutch / brake | positional | RoboHAT RC-PWM | 0 (released/driving) .. 1 (pressed = declutch+brake) |
| gear (F/N/R) | discrete | RoboHAT RC-PWM | forward / neutral / reverse |
| starter | momentary relay | GPIO | engine crank pulse |
| blade (PTO) | latching relay | GPIO | engage / disengage |

Positional actuators are RC-PWM channels; the RoboHAT RP2040 firmware must accept
the per-channel form configured by `pwm_line_format` (default `pwm,<ch>,<us>`).
Calibrate each channel's `us_min/us_center/us_max` on the bench.

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
