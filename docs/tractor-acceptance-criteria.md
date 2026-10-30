# Tractor Platform — Safety Acceptance Criteria

This is the sign-off checklist for the ride-on zero-turn mower's actuation and
navigation work before any tractor-platform deployment is represented as
constitutionally compliant (Constitution Principle V/VI, amended 2026-09-01,
v4.0.0). It is written now, ahead of that code landing, as the contract it
must satisfy — not produced retroactively as paperwork.

**None of these items are satisfied by the existing unit-test interlock
suite passing.** Those tests already pass today and will keep passing
regardless of whether any command ever reaches real hardware — they check
`state.field == value` in-process, not a physical outcome. A passing unit test
suite is necessary evidence, never sufficient evidence, for anything below.

## 1. Bus-fault failsafe (top priority — do this first)

PCA9685 boards hold their last commanded PWM value on I2C/host loss; they do
not fail safe on their own. This is transport-level, not platform-level: it
was true when PCA9685 replaced the tractor's original RC-PWM transport and
remains equally true after the Craftsman-to-zero-turn actuator swap. It is
the single largest risk the tractor effort introduces: a bus fault
mid-mission could freeze the drive levers or throttle at whatever was last
commanded, with the engine running.

- [ ] A failsafe mechanism is designed and documented (e.g., a hardware watchdog
      driving the PCA9685 `OE` pin low on timeout, or spring-return actuator
      hardware for throttle/left-lever/right-lever).
- [ ] The failsafe is **physically demonstrated on the bench**: with the engine
      running and a non-idle throttle/lever command active, the I2C bus or
      the PCA9685's host connection is physically interrupted, and the actuators
      are observed (not inferred) to reach a safe state without further software
      involvement.
- [ ] The demonstrated time-to-safe-state is recorded and compared against the
      Constitution's bus-fault-failsafe requirement (Principle VI).

## 2. `emergency_stop()` — 3 effects, mechanically observed

`TractorControlService.emergency_stop()` is specified to produce three effects:
blade/PTO disengages, both drive levers return to neutral, throttle returns to
idle. (This is down from the superseded Craftsman-class design's 5 effects —
there is no clutch/brake or gear selector on a hydrostatic ZTR for
`emergency_stop()` to actuate.)

- [ ] With the engine running and the tractor mid-command (non-neutral levers,
      non-idle throttle, blade engaged where applicable), trigger
      `emergency_stop()` and visually/mechanically confirm, on the physical
      machine, each of:
  - [ ] Blade/PTO physically disengages.
  - [ ] Left lever physically reaches neutral.
  - [ ] Right lever physically reaches neutral.
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
      calls (left lever, right lever, throttle), and confirm the remaining
      safing steps still complete — per the plan, each of `emergency_stop()`'s
      PWM-bearing calls is wrapped in its own try/except so one bad I2C write
      cannot abort the rest of the safing sequence, and the GPIO blade-PTO
      relay cutoff (pure GPIO, unaffected by I2C) still fires regardless.
- [ ] Confirm the fault itself is surfaced (logged, and where applicable
      escalated), not silently swallowed — a mid-mission I2C write failure must
      reach the caller so it can trigger emergency stop. This is a deliberate,
      permanent design requirement: do not let a future refactor of the PWM
      send path reintroduce the swallow-all-exceptions behavior the original
      RoboHAT-serial `_send_pwm` transport had, which would make this
      fault-injection test meaningless if it crept back in.

## 4. Per-actuator-class latency measurement

- [ ] Relay actuators (blade PTO, starter): measured de-energize time on
      real hardware, engine running, against the <100ms requirement.
- [ ] Positional actuators, command issuance — measured **separately per
      actuator**: `left_lever`, `right_lever`, and `throttle` each get their
      own measured time from E-stop signal to the safe-value command being
      issued on the bus, against the <100ms requirement. `left_lever` and
      `right_lever` are two independently-actuated servos and MUST be
      measured as two separate rows, not one combined "levers" line.
- [ ] Positional actuators, physical settle — measured **separately per
      actuator**:
  - [ ] `left_lever`: time from E-stop signal to physically reaching neutral,
        against the <500ms requirement.
  - [ ] `right_lever`: time from E-stop signal to physically reaching neutral,
        against the <500ms requirement.
  - [ ] `throttle`: time from E-stop signal to physically reaching idle,
        against the <500ms requirement.

  This requires physical instrumentation (e.g., a potentiometer/limit
  switch/high-speed video on each actuator), not a log timestamp of the
  commanded value.
- [ ] For `left_lever` and `right_lever` specifically, the measured
      physical-settle time is recorded **at the actual servo supply voltage
      used** (24V via boost converter, per `docs/hardware-overview.md`'s
      Power section) — this measurement is the entire justification for
      running the lever servos at 24V instead of 12V and must not be
      skipped. If bench measurement on the real linkage and real lap-bar
      load shows the <500ms requirement is comfortably met at 12V instead,
      that is a legitimate simplification to revisit — but only once
      measured on hardware, never assumed from a datasheet speed rating.

## 5. Navigation controller — interlocked-path-only verification

- [ ] Code review (or an automated check) confirms the navigation controller
      (`navigation_service.py`) drives the tractor exclusively through
      `tractor_service.py`'s existing public, interlocked methods (e.g.
      `apply()`, `start_engine()`, `emergency_stop()`, and whatever
      per-lever/throttle setter methods the actuator rework lands — the exact
      method surface is owned by the actuation workstream, not fixed here)
      and never constructs or writes to a PCA9685 channel, GPIO relay, or
      other raw actuator interface directly.
- [ ] Confirm the soft-stop / hard-stop distinction holds: routine stops
      (waypoint reached, mission paused) go through a non-revoking soft-stop
      path (e.g., both levers commanded to neutral without revoking
      authorization), and only an actual command failure or genuine
      emergency escalates to full `emergency_stop()` (which revokes
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
Constitution Sync Impact Report, Follow-up TODOs, carried forward unchanged
from v3.0.0 through v4.0.0) and must not be forgotten under the
tractor-specific items above:

- [ ] IMU tilt-cutoff (<200ms from threshold breach to blade stop) is wired to
      the tractor's blade/PTO control path.
- [ ] Motor-watchdog heartbeat, with automatic emergency stop on timeout, is
      wired to the tractor's actuator control loop.

## 8. OEM seat / operator-presence (OPC) — current condition

**Owner report, 2026-10-30:** The seat-presence input on this particular mower
always indicates that an operator is seated. Treat this as **current reported
hardware condition**, not as an available or functioning safety interlock.
Do not count it among the redundant protections in a minimal-cost redesign,
and do not treat this report as evidence that unattended operation is safe.

The manufacturer's normal operator-presence protective function is therefore
unavailable as described. The design must preserve the remaining factory
start/PTO/PARK interlocks and add independently verifiable stopping safeguards
appropriate to an unoccupied machine. No instructions to defeat protective
circuits are implied or authorized by this status record.

- [ ] Validate and document the actual electrical condition on the mower;
      distinguish user report from instrumented test.
- [ ] Do not count the seat signal as a heartbeat, operator-presence sensor,
      emergency-stop channel, or proof that the area is clear.
- [ ] Document and review the loss of operator-presence protection with the
      physical fail-stop architecture; no unmanned deployment before
      independent engine/PTO/propulsion stops are physically proven.
- [ ] Keep the separate pre-mission area-clear/operator-authorization gate;
      it does not replace an independent physical safety stop.

## 9. Reverse definition: pivot turn vs. true reverse

Reverse is defined as **both** drive levers negative; one lever negative with
the other at or above neutral is a zero-radius pivot turn made while
otherwise mowing forward, not reverse (`docs/tractor-platform.md`).

- [ ] On the physical machine, confirm the blade/PTO does **NOT** disengage
      during a normal one-lever pivot turn (one lever negative, the other
      neutral or positive, blade previously engaged).
- [ ] On the physical machine, confirm the blade/PTO **DOES** disengage when
      true reverse is entered (both levers negative).
- [ ] Confirm on the bench, not assumed, whether the mower's own OEM
      mechanical Reverse Operation System also cuts the blade in reverse
      independently of this software interlock — if so, document both
      mechanisms and which one is authoritative; do not rely on the software
      interlock alone being the only thing preventing blade-in-reverse if the
      OEM mechanism is also present and could mask a software regression.

## 10. Lever self-centring — gates the §1 failsafe design

The bus-fault failsafe in §1 assumes that removing servo power lets the
motion-control levers' own return springs pull them to neutral (the
Constitution's "spring-return actuator hardware" option). **The operator's
manual does not state whether these levers self-centre.** If they do not,
removing servo power leaves the levers wherever they were commanded — a
runaway, not a failsafe — and §1's design must be replaced.

- [ ] By hand, with the engine off, displace each motion-control lever fore
      and aft and release it. Record whether it returns to neutral unaided,
      and with how much residual offset.
- [ ] Repeat with the servo linkage fitted but unpowered, confirming the
      linkage itself does not prevent return (ball-joint friction, pushrod
      binding, servo cogging torque).
- [ ] If the levers do **not** self-centre, record that §1's failsafe design
      is invalid as written and must be redesigned before any powered
      autonomous test.

## 11. Manual-start operating sequence (fore/aft-only linkage)

Confirmed from the manual: the engine cranks only with the blade switch OFF
**and the levers in PARK (outboard)**. The servo linkage actuates fore/aft
only and cannot reach PARK, so autonomous engine start is physically
impossible in this configuration.

- [ ] Confirm the documented operating sequence is what operators are actually
      trained to do: human starts the engine with levers in PARK, moves levers
      to centre, then hands over to autonomous control.
- [ ] Confirm the software's `require_levers_neutral_to_start` interlock
      (levers **centred**) is understood as distinct from the OEM crank
      interlock (levers **outboard in PARK**) — these check different
      positions and neither substitutes for the other.
- [ ] Confirm the starter relay's presence in the wiring does not lead anyone
      to believe unattended autonomous start is available.

## 12. Dual-controller complete safe stop (Pi 5 OR Libre Alta failure)

The owner requires a **complete, latched stop if either controller fails**, not
merely separate local shutdown or a routine pause. See
[`2026-10-30-dual-controller-safe-stop-requirement.md`](2026-10-30-dual-controller-safe-stop-requirement.md).
This overrides the older engine-keeps-running emergency-stop preference for
**controller failures**. Control messages may travel over Wi-Fi HaLow, but a
physical safe stop cannot depend solely on either host or the radio link.

- [ ] Loss of either controller individually, both together, or intercontroller
      heartbeat drives PTO authorization OFF, disables starting, shuts down the
      engine and demonstrably halts propulsion through independent protection.
- [ ] Confirm physical lever neutral/braking under power-loss conditions; cutting
      servo power **alone does not prove** a safe mechanical stopping outcome.
- [ ] Fault-inject controller crashes, reboots, unidirectional radio loss,
      stale messages, stuck PWM outputs and loss of both supplies.
- [ ] Record measured response times, blade coast-down to zero RPM, engine
      shutdown and propulsion stop time.
- [ ] Confirm the stop remains latched until deliberate operator inspection
      and rearm; connectivity recovery never restarts mowing.
- [ ] No autonomous engine-on deployment until these tests pass on real hardware.

## Sign-off

None of the above may be checked off from simulation or unit-test evidence
alone. Each item requires a real, instrumented, on-hardware observation, logged
with date, operator, and measured values (not just pass/fail) for the latency
and fault-injection items.

| Item | Date | Verified By | Measured Result | Notes |
|------|------|-------------|------------------|-------|
| 1. Bus-fault failsafe | | | | |
| 2. E-stop 3 effects | | | | |
| 3. I2C fault injection | | | | |
| 4. Per-class latency (left_lever / right_lever / throttle / relays) | | | | |
| 5. Interlocked-path-only | | | | |
| 6. Operator-attestation gate | | | | |
| 7. Tilt-cutoff + watchdog wiring | | | | |
| 8. Seat/OPC-switch decision | | | | |
| 9. Reverse definition: pivot vs. true reverse | | | | |
| 10. Lever self-centring (gates item 1) | | | | |
| 11. Manual-start operating sequence | | | | |
| 12. Dual-controller complete safe stop | | | | |
