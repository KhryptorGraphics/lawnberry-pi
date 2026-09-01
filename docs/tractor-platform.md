# Ride-on Zero-Turn Mower Platform (Toro TimeCutter conversion)

The tractor platform is a converted 50" Toro TimeCutter zero-turn mower
(ZTR) — a gas-engine vehicle with twin-lever hydrostatic drive — not a
differential-drive robot and not an Ackermann-steered tractor. There is no
steering wheel, gas pedal, clutch, or gear selector: each of the two drive
levers independently commands its own side's ground speed and direction, and
the mower's own hydrostatic transaxles turn that into motion (push a lever
forward to drive that side forward, pull it back to reverse it, center it to
stop that side). The engine and both hydrostatic transaxles are kept stock;
two high-torque servos (Wingxine ASMC-04B) physically push/pull the existing
lap-bar levers. Only the PWM signal from each servo runs through the same
PCA9685 I2C PWM transport already used elsewhere on this platform — servo
power itself is a dedicated 24V rail, run at 24V rather than 12V specifically
for E-stop physical-settle margin under Principle VI's 500ms requirement
(see `docs/hardware-overview.md`'s Power section for the full power tree,
the wiring constraint, and the rationale). It is operated through five
discrete actuators.

## Actuators

| Actuator | Type | Drive | Range / semantics |
|----------|------|-------|-------------------|
| left_lever | positional | PCA9685 I2C PWM (ch 0) | −1 (full reverse) .. 0 (neutral) .. +1 (full forward) |
| throttle | positional | PCA9685 I2C PWM (ch 1) | 0 (engine idle) .. 1 (full RPM) |
| right_lever | positional | PCA9685 I2C PWM (ch 2) | −1 (full reverse) .. 0 (neutral) .. +1 (full forward) |
| starter | momentary relay | GPIO | engine crank pulse |
| blade_pto | latching relay | GPIO | engage / disengage |

Direction is intrinsic to each lever's own position — there is no separate
steering, gas-pedal, clutch, or gear actuator. **Reverse is defined as BOTH
levers negative.** One lever negative while the other is at or above neutral
is a routine zero-radius pivot turn made while otherwise mowing forward, not
reverse — this distinction matters directly for the Reverse Operation System
interlock below.

Positional actuators are PCA9685 I2C PWM channels (0-indexed, `config/tractor.yaml`'s
`pca9685:` block sets the I2C address/bus/frequency — moved off the PCA9685's 0x40
power-on default, which collides with the INA3221 power monitor, to 0x41). The
blade-PTO relay's GPIO was moved off pin 6 (collides with the ToF Left Interrupt)
to GPIO 26. Both are verified conflict-free by `scripts/check_hardware_pin_conflicts.py`
(wired into `.github/workflows/hardware-guard.yml`). Calibrate each channel's
`us_min/us_center/us_max` on the bench.

A PCA9685 board holds its last commanded pulse if the I2C bus or host is lost —
it does not fail safe on its own. `PCA9685Driver` propagates any write failure
that occurs after hardware was confirmed present (as opposed to a tolerated
no-op when no board is wired up at all) so a real transport fault is visible to
callers, but a physical bus-fault failsafe (e.g. a watchdog driving the board's
`OE` pin) is a separate, not-yet-built hardware-phase requirement — see
`docs/tractor-acceptance-criteria.md` item 1.

## Safety interlocks (standard lawn-tractor, ANSI/OPEI-style)

- **Engine start** only when *authorized + both levers neutral + blade off*.
  A hydrostatic ZTR has no clutch pedal, so there is no separate
  clutch-pressed condition to check.
- **Blade/PTO** engages only with the *engine running* and *not in reverse*;
  entering reverse (both levers negative — see the actuator table above)
  **auto-disengages the blade** (Reverse Operation System). A one-lever pivot
  turn is not reverse and does not, by itself, disengage the blade. Note the
  mower's own OEM mechanical Reverse Operation System may also cut the blade
  in reverse independently of this software interlock; which one actually
  fires (or both) must be confirmed on the bench, not assumed — see
  `docs/tractor-acceptance-criteria.md`.
- **Emergency stop** has three effects: disengages the blade/PTO, returns
  both drive levers to neutral, and idles the throttle — **the engine keeps
  running** (configurable); authorization is revoked and further commands are
  blocked until cleared. Unlike the superseded Craftsman-class design, there
  is no clutch/brake actuator for `emergency_stop()` to press — levers-to-neutral
  gives only the hydrostatic transaxles' own self-braking, not a positive
  parking brake (see "Physical integration decisions" below).

## Physical integration decisions

Sourced from the Toro operator's manual (**TimeCutter Max 50 in, form
3465-589**) plus items still open at bench time. The manual settled several
questions that were previously guesses — those are marked **confirmed**.

1. **Lever kinematics — two axes (confirmed).** The motion-control levers
   move fore/aft (Fast – Slow – NEUTRAL – Reverse) *and* swing **outboard to
   PARK**. Per the manual, PARK is both the parking brake and a condition of
   the engine-crank interlock. The servo linkage in `hardware/mounts/`
   deliberately actuates **fore/aft only** and is designed to stay clear of
   the outboard swing (ball joints at both rod ends plus a quick-release
   clevis pin at the lever). Verify the full outboard sweep by hand, with
   the linkage fitted, before running the engine.
2. **A positive parking brake exists, but this linkage cannot reach it
   (confirmed).** Contrary to an earlier revision of this document, the
   machine is **not** limited to passive hydrostatic self-braking: moving
   the levers outboard to PARK engages a real parking brake. It is simply
   not reachable by a fore/aft-only actuator, so **engaging the parking
   brake remains a manual action**. `emergency_stop()`'s levers-to-neutral
   therefore yields hydrostatic braking only, and the documented slope
   limit still applies to autonomous operation — but the machine itself has
   a proper brake, and a future outboard actuator could reach it.
3. **Autonomous engine start is physically impossible with this linkage
   (confirmed).** The manual's interlock requires the blade switch OFF *and*
   the levers **in PARK (outboard)** before the engine will crank. Software's
   `require_levers_neutral_to_start` checks levers **centred** — a different
   position. These are not equivalent and must not be conflated. The starter
   relay can energise the solenoid, but the OEM interlock will refuse to
   crank unless a human has first placed the levers in PARK. **Operating
   sequence: human starts the engine with levers in PARK, moves them to
   centre, then hands over to autonomous control.**
4. **Seat / operator-presence switch (confirmed behaviour, decision open).**
   The manual specifies the engine shuts off **within 1 second** whenever the
   levers are out of PARK and the operator rises from the seat. Empty-seat
   autonomy therefore cannot work until this circuit is deliberately
   addressed. Whichever way it is resolved — satisfy vs. bypass — MUST be
   documented here once decided, and MUST be explicitly paired with the
   Constitution's operator-attestation gate (Principle VI) as the
   compensating control: bypassing a physical presence switch is only
   acceptable alongside a logged, enforced software attestation that stands
   in for it. **Status: undecided — open item.** For supervised bring-up the
   simplest and safest answer is to ride along, leaving the interlock intact.
5. **Manual choke (confirmed).** The machine has a separate, manual choke
   control. No actuator drives it; **cold starting is a human action in every
   phase** of this build.
6. **Lever self-centring — UNVERIFIED, and it gates the failsafe design.**
   The bus-fault failsafe assumes that cutting servo power lets the levers'
   own return springs pull them to neutral (the Constitution's "spring-return
   actuator hardware" option). The manual does not state whether these levers
   self-centre. **If they do not, cutting servo power leaves the levers where
   they sat — a runaway, not a failsafe** — and the failsafe must be
   redesigned. Check this by hand before building anything that depends on
   it. **Status: open, highest-priority bench check.**

### Machine data from the manual

| Value | Figure |
|---|---|
| Weight | 292 kg (644 lb) |
| Maximum rated slope | 15° |
| Cutting width | 127 cm (50 in) |
| Throttle | continuous-variable lever, SLOW–FAST |
| Bypass levers | both sides of the engine; disengage hydro drive for pushing |

**Model caveat**: form 3465-589 covers the *TimeCutter Max 50 in*. The
deployment target has been described as "TimeCutter 2", which is not a
published Toro model name; the 50" line includes TimeCutter, TimeCutter MAX
and TimeCutter MyRIDE. Control architecture is consistent across the line, so
the confirmed items above hold, but dimensions may differ — check the serial
plate and pull the matching manual before fabricating.

## Configuration

`config/tractor.yaml` maps actuators to PWM channels / GPIO relays and sets the
interlock flags. Set `enabled: true` on the converted mower. When enabled, the
autonomous AI loop drives the tractor via `ActionPrediction.to_tractor_command()`
(mapped into left/right lever + throttle + blade PTO) instead of differential
motors. The AI model's own action-prediction output space still describes a
single differential-style steering axis (see `docs/ai-architecture.md`); owning
the steering-to-twin-lever mixing inside `to_tractor_command()` is an open item
for the actuation/navigation workstream, not yet resolved here.

## API & UI

- REST: `/api/v2/tractor/*` — state, per-actuator setpoints (left lever, right
  lever, throttle), blade, starter, authorize/revoke, emergency-stop/clear.
  Exact route names are owned by the actuation workstream's implementation,
  not fixed by this document.
- UI: the **Tractor** dashboard view (`/tractor`).

## Code

- `models/tractor_control.py`, `drivers/actuators/tractor_actuators.py`,
  `services/tractor_service.py`, `api/routers/tractor.py`.
