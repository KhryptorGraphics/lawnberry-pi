# Claude Code — Pololu #2806 bidirectional steering failover (2026-11-07)

Status: design only; not wired or bench-verified.

## Inventory
Two PCA9685 16-channel PWM boards (one per computer); two 150 kg steering servos; Raspberry Pi 5 primary and Libre Alta backup. One assembled Pololu 4-Channel RC Servo Multiplexer #2806 is documented as purchased; owner now reports **two** #2806 boards. Confirm second unit before assigning it. **One** #2806 has enough channels for both steering servos; the other may remain spare. No new multiplexer or independent hardware watchdog is required for this proposed steering signal selection. Simultaneous controller failure: **accepted risk**, not a safety guarantee.

## Proposed wiring and selection
Pi PCA9685 left PWM -> #2806 M1; Alta PCA9685 left PWM -> S1; OUT1 -> left servo signal.
Pi PCA9685 right PWM -> M2; Alta PCA9685 right PWM -> S2; OUT2 -> right servo signal.
Alta PCA9685 spare PWM channel -> #2806 SEL. SEL requires **valid RC pulses**, not a static GPIO level. Verify manufacturer pulse widths, threshold, polarity, voltage levels, power and grounds: https://www.pololu.com/docs/0J60/all .
Set FAILMODE jumper OFF **only after testing**: invalid/missing SEL should default to master/Pi. FAILMODE ON drives outputs low on invalid SEL, which is **not** servo neutral. Never short two PCA9685 outputs together. Servo motor power is separate from signal logic.

## Bidirectional transitions
- Both healthy: Pi master PWM selected; Alta backup PWM ready, not driving selected output.
- Pi failure, Alta healthy: Alta heartbeat detection requests slave selection using valid SEL pulse, then controls both servo signals. No dependence on Pi to switch.
- Alta failure, Pi healthy: SEL becomes invalid; tested FAILMODE-OFF behavior defaults to Pi master; Pi handles peer fault independently.
- Heartbeat network partition: fault latch rather than competing master decisions.
- Recovery: no uncontrolled oscillation or automatic blade/motion rearm.

**Important existing policy:** The software-only failover spec currently requires **fault-latched PTO OFF, neutral/idle, and no automatic mowing continuation** after either host fails. Source-selection failover does not itself authorize continued mowing. Preserve this behavior until the owner explicitly changes it. Engine start/stop remains OEM manual. PTO control is separate from the Pololu mux.

## Claude Code work and acceptance criteria
Implement explicit selector-source states, pulse configuration, dual steering PWM channels on both PCA9685s, heartbeat-based selection, telemetry and fault latching. Initialize Alta backup steering PWM before selecting it. Calibrate actual mechanical neutral; do not assume a generic 1500 us pulse is neutral.

With blades disabled, mower immobilized, and servo linkages disconnected, bench-test Pi process hang/power loss, Alta process hang/power loss, SEL absent/invalid/stuck, mux power loss, heartbeat loss, boot order, recovery, and repeated transitions. Measure actual output pulses and check for unwanted motion. Then validate electrical isolation and steering behavior before any mower operation. Never claim operational failover before successful testing; preserve OEM interlocks.
