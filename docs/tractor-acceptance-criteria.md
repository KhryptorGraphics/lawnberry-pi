# Tractor Platform — Safety Acceptance Criteria

This is the sign-off checklist for the ride-on tractor's actuation and navigation
work (see the plan's Workstreams 1–2) before any tractor deployment is represented
as constitutionally compliant (Constitution Principle V/VI, amended 2026-07-08,
v3.0.0). It is written now, ahead of that code landing, as the contract it must
satisfy — not produced retroactively as paperwork.

**None of these items are satisfied by the existing 13 `tests/unit/test_tractor.py`
interlock tests passing.** Those tests already pass today and will keep passing
regardless of whether any command ever reaches real hardware — they check
`state.field == value` in-process, not a physical outcome. A passing unit test
suite is necessary evidence, never sufficient evidence, for anything below.

## 1. Bus-fault failsafe (top priority — do this first)

PCA9685 boards hold their last commanded PWM value on I2C/host loss; they do not
fail safe on their own. Once PCA9685 replaces the currently-broken RC-PWM
transport, this becomes the single largest new risk the tractor effort
introduces: a bus fault mid-mission could freeze throttle/gas-pedal at whatever
was last commanded, with the engine running.

- [ ] A failsafe mechanism is designed and documented (e.g., a hardware watchdog
      driving the PCA9685 `OE` pin low on timeout, or spring-return actuator
      hardware for throttle/gas-pedal/steering).
- [ ] The failsafe is **physically demonstrated on the bench**: with the engine
      running and a non-idle throttle/gas-pedal command active, the I2C bus or
      the PCA9685's host connection is physically interrupted, and the actuators
      are observed (not inferred) to reach a safe state without further software
      involvement.
- [ ] The demonstrated time-to-safe-state is recorded and compared against the
      Constitution's bus-fault-failsafe requirement (Principle VI).

## 2. `emergency_stop()` — all 5 effects, mechanically observed

`TractorControlService.emergency_stop()` is specified to produce five effects:
blade off, gas-pedal to 0, gear to neutral, clutch pressed, throttle to idle.

- [ ] With the engine running and the tractor mid-command (non-neutral gear,
      non-zero throttle/gas-pedal, blade engaged where applicable), trigger
      `emergency_stop()` and visually/mechanically confirm, on the physical
      machine, each of:
  - [ ] Blade/PTO physically disengages.
  - [ ] Gas pedal / ground-speed actuator physically returns to 0.
  - [ ] Gear selector physically reaches neutral.
  - [ ] Clutch/brake actuator physically reaches the pressed position.
  - [ ] Throttle actuator physically reaches idle.
- [ ] Each effect's timing is measured against the Constitution's tiered
      latency requirement (Principle VI): the GPIO relay (blade PTO) within
      100ms of signal; each positional actuator's safe-value *command* issued
      within 100ms of signal; each positional actuator's *physical* settle
      within 500ms of signal.
- [ ] Confirm authorization is revoked after the stop and that a subsequent
      drive command is rejected until `clear_emergency()` is explicitly called
      by an operator.

## 3. Fault-injection test: I2C link broken mid-e-stop

- [ ] With an `emergency_stop()` in progress, break the I2C link (physically or
      via a simulated bus fault) partway through the sequence of PWM-bearing
      calls, and confirm the remaining safing steps still complete — per the
      plan, each of `emergency_stop()`'s PWM-bearing calls is wrapped in its own
      try/except so one bad I2C write cannot abort the rest of the safing
      sequence, and the GPIO blade-PTO relay cutoff (pure GPIO, unaffected by
      I2C) still fires regardless.
- [ ] Confirm the fault itself is surfaced (logged, and where applicable
      escalated), not silently swallowed — a mid-mission I2C write failure must
      reach the caller so it can trigger emergency stop; this is a deliberate
      behavior change from `_send_pwm`'s current swallow-all-exceptions state
      under the RoboHAT-serial transport, which would make this fault-injection
      test meaningless if left in place under PCA9685.

## 4. Per-actuator-class latency measurement

- [ ] Relay actuators (blade PTO, starter): measured de-energize time on
      real hardware, engine running, against the <100ms requirement.
- [ ] Positional actuators, command issuance: measured time from E-stop signal
      to the safe-value command being issued on the bus, against the <100ms
      requirement.
- [ ] Positional actuators, physical settle: measured time from E-stop signal
      to the actuator physically reaching its safe position (steering
      centered, throttle idle, gas pedal 0, clutch pressed, gear neutral),
      against the <500ms requirement. This requires physical instrumentation
      (e.g., a potentiometer/limit switch/high-speed video on each actuator),
      not a log timestamp of the commanded value.

## 5. Navigation controller — interlocked-path-only verification

- [ ] Code review (or an automated check) confirms the navigation controller
      (`navigation_service.py`) drives the tractor exclusively through
      `tractor_service.py`'s existing public, interlocked methods
      (`apply()`, `start_engine()`, `set_ground_speed()`, `emergency_stop()`,
      etc.) and never constructs or writes to a PCA9685 channel, GPIO relay, or
      other raw actuator interface directly.
- [ ] Confirm the soft-stop / hard-stop distinction holds: routine stops
      (waypoint reached, mission paused) go through a non-revoking soft-stop
      path (e.g., `set_ground_speed(0)`), and only an actual command failure or
      genuine emergency escalates to full `emergency_stop()` (which revokes
      authorization and blocks further commands until manually cleared).

## 6. Operator-attestation gate (Constitution Principle VI)

- [ ] Confirm an explicit, logged operator attestation ("seat/area clear") is
      required and enforced before blade/PTO engagement.
- [ ] Confirm the same attestation gate is required and enforced before the
      start of any autonomous motion.
- [ ] Confirm this attestation is a genuine gate (command rejected without it),
      not merely a UI checkbox with no backend enforcement.

## 7. Two pre-existing gaps this platform must not ship without closing

These are not new requirements — Constitution Principle VI already required
them for "all motor control operations" before the tractor existed. They are
listed here because they are currently unmet by `tractor_service.py` (see
Constitution v3.0.0 Sync Impact Report, Follow-up TODOs) and must not be
forgotten under the tractor-specific items above:

- [ ] IMU tilt-cutoff (<200ms from threshold breach to blade stop) is wired to
      the tractor's blade/PTO control path.
- [ ] Motor-watchdog heartbeat, with automatic emergency stop on timeout, is
      wired to the tractor's actuator control loop.

## Sign-off

None of the above may be checked off from simulation or unit-test evidence
alone. Each item requires a real, instrumented, on-hardware observation, logged
with date, operator, and measured values (not just pass/fail) for the latency
and fault-injection items.

| Item | Date | Verified By | Measured Result | Notes |
|------|------|-------------|------------------|-------|
| 1. Bus-fault failsafe | | | | |
| 2. E-stop 5 effects | | | | |
| 3. I2C fault injection | | | | |
| 4. Per-class latency | | | | |
| 5. Interlocked-path-only | | | | |
| 6. Operator-attestation gate | | | | |
| 7. Tilt-cutoff + watchdog wiring | | | | |
