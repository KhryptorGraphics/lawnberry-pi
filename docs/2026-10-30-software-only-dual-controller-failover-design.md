# Lawnberry Pi — software-only dual-controller failover design

**Decision date:** 2026-10-30  
**Target:** Toro TimeCutter MAX Havoc MyRIDE 60-inch, owner-reported model **77603**.  
**Document type:** Software functional specification; **proposal only**, not code, installed wiring, or a verified mower stop capability.

## 1. Scope and boundaries

Implement the owner's **software-based primary/backup control strategy** using the **Raspberry Pi 5** and **Libre Alta**, with no **additional external hardware watchdog modules** and no **added physical emergency-stop switch** in this proposed configuration. Continue using already owned **two PCA9685 servo-control modules**, **two HiLetgo 5 V single-channel relay modules**, and other purchased parts where electrically appropriate; do **not** presume either relay can perform more than one independent switching function.

This specification defines **what the software must attempt and how it coordinates**. It cannot replace the electrical paths, servo signal ownership, or mechanical stop/neutral capability needed for those commands to work. Existing OEM safety protections must not be bypassed. It is **not a declaration that software-only redundancy creates a fail-safe autonomous mower**.

**Single-computer failure is the selected software fault model**. Simultaneous Pi + Alta failure, shared power failure, relay contacts welded closed, servo/PWM command stuck, I²C bus physically held, severed wiring, or jammed steering/throttle linkages are not made safe merely by duplication of software.

## 2. Roles

| Component | Software responsibility while healthy | Responsibility if its peer fails |
| --- | --- | --- |
| **Raspberry Pi 5 — primary** | Own normal navigation and requested motion/steering, throttle, starter, blade/PTO control *through verified available local interfaces*. Publish desired mode and actual state. | If **Alta fails**, immediately stop accepting new motion/start/PTO commands, request steering **NEUTRAL**, throttle **IDLE**, **PTO OFF**, and **ENGINE OFF**, then latch the fault. |
| **Libre Alta — hot standby** | Run the same heartbeat parser, fault detector, stop state machine, and local stop-output service in parallel. Keep its output authority available without requiring Pi messages during a failure. | If **Pi fails**, immediately stop accepting new commands, request steering **NEUTRAL only if Alta has an independently usable steering output path**, throttle **IDLE** if controllable, **PTO OFF**, and **ENGINE OFF** through its **own verified interfaces**, then latch the fault. |
| **PCA9685 on Pi** | Pi-local PWM for two steering servos as currently proposed. | May hold its last PWM if the Pi/I²C fails. **Alta cannot assume it can update Pi-controlled PWM remotely.** |
| **PCA9685 on Alta** | Alta-local PWM for MG995 throttle as currently proposed. | It is **not automatically an alternate controller for Pi steering PWM** merely because it has spare channels. |
| **Relay/control outputs** | Must expose correctly rated and independent off-command paths for engine and PTO. | A surviving computer cannot issue an electrically effective OFF command through a failed host's GPIO or I²C path. |

**Normal authority:** One software master for motion/output commands: Pi 5. Alta is a **shutdown backup**, not a second competing active driver. Avoid both computers commanding the same servo or relay in parallel.

**Output access dependency (open):** To meet the owner's *full steering-neutral request after Pi failure*, the Alta needs a **real independently operable steering signal path** with defined single-writer ownership/isolation. A common network protocol does not establish such a path; connecting two PWM outputs or I²C masters together without suitable arbitration/isolation is not a valid software solution. If this path is not physically available, the correct documented outcome is **Alta requests engine and PTO OFF but cannot guarantee or command steering neutral**.

## 2C. Independent blade/PTO shutdown — explicit owner requirement

**Owner clarification (2026-10-30):** In addition to backup steering, the **Libre Alta must independently be capable of switching the mower's blades OFF** when the Raspberry Pi 5 fails. Conversely, if Alta fails, **Pi 5 must independently command blades OFF**. The Pololu #2806 does **not** participate in PTO control; it is for steering servo PWM only.

| Failure | Surviving computer | Mandatory PTO behavior |
| --- | --- | --- |
| Pi 5 failure | Libre Alta | Issue **PTO OFF** through an **Alta-accessible electrical path**, without a Pi GPIO, Pi I²C, Pi process, or network acknowledgement. |
| Alta failure | Raspberry Pi 5 | Issue **PTO OFF** through a **Pi-accessible electrical path**, without an Alta GPIO, Alta I²C, or network acknowledgement. |
| Pi–Alta heartbeat link lost | Each working computer | Revoke local permission to run blades and request **PTO OFF**, with no assumption of a healthy peer. |

**Software contract:** Implement a local `request_pto_off()` adapter on **each** computer and call it during any latched fault **without waiting for steering return or engine shutdown**. The adapter's existence in code is not evidence that its relay or clutch circuit actually switches OFF.

**Unresolved physical dependency:** The two existing **HiLetGo 5 V, 30 A single-channel relay modules** do not automatically provide independently controllable steering power, PTO, engine kill, and starter circuits. One module is planned for steering-power control; whether the other plus the owned automotive relay harness can implement **both hosts' independent PTO-OFF authority** (as well as distinct engine-control functions) must be determined from the model 77603 serial-matched wiring and actual relay contacts. **Do not tie Pi and Alta GPIO outputs together**, assume a software-only GPIO command interrupts clutch current, or bypass the factory blade safety circuit.

**Verification gate:** With PTO electrically isolated in a safe test arrangement, demonstrate (a) Pi removed and Alta commands clutch permission OFF, (b) Alta removed and Pi commands clutch permission OFF, (c) heartbeat loss results in OFF requests, (d) no automatic re-engagement after restart. A PTO-OFF command disengages an electric clutch; **blade coast-down takes additional time** and must be measured separately.

## 2A. Owner-proposed Alta connection to steering HiLetGo relay — clarified

**Owner clarification (2026-10-30):** The Libre Alta is intended to be **wired into the same existing HiLetGo relay interface that switches the steering-servo power**, so Alta can act if the Pi 5 fails. This is a **proposed wiring design**, not an already verified installed connection.

**What the shared relay does:** The HiLetGo 5 V, 30 A, **one-channel relay module** is an electrically operated **on/off contact**. A software command from Alta could operate a properly isolated, independently controllable **steering power-enable/power-cut circuit**. The relay **does not generate the per-servo PWM signals** needed to set both steering levers to their neutral positions. With power removed, positional/geared servos may keep the bars where they were; this must not be represented as commanded neutral.

**Separate control paths must be distinguished:**
- **Power authority:** If Alta can independently make the existing steering relay release, it can request *servo power OFF*, subject to actual supply topology and the relay's contact state. Two SBC GPIO outputs **must not be shorted directly together** into a single trigger input; design and verify stop-dominant electrical input isolation/combining.
- **Position authority:** To command **both bars to NEUTRAL** on Pi failure, Alta must also control the PWM signals feeding the two steering servos. Options to evaluate (not presumed present) are **an electrically isolated selectable PWM source** using spare channels on Alta's already owned PCA9685, or a **properly arbitrated and isolated I²C path** to the steering PCA9685. Neither is accomplished by connecting Alta only to the HiLetGo relay. No extra servo PWM controller may be necessary, but **signal-selection/isolation wiring or hardware is still needed**.
- **Engine and blades:** The steering-power relay does not independently switch the ignition RUN/kill and blade/PTO circuits. Their separate OFF-command paths still need verification.

**Proposed software behavior with the owner's relay connection:** On Pi failure, Alta first revokes starting/motion/PTO authorization and issues its independently connected **engine OFF** and **blade PTO OFF** requests. Alta may request steering **NEUTRAL only if a live and independently selectable PWM path is verified**; otherwise it can only request steering-servo **power OFF**, reporting `STEERING_POWER_CUT_NO_NEUTRAL_CONTROL`. Do not claim stopped motion based on relay activation alone.

Manufacturer PCA9685 reference: https://www.nxp.com/products/power-drivers/lighting-driver-and-controller-ics/led-drivers/16-channel-12-bit-pwm-fm-plus-ic-bus-led-driver:PCA9685 . HiLetGo model identified in owned-hardware document as **B0CHFJSNP6**; confirm actual purchased relay coil marking and contact wiring before designing the interface.

## 2B. Owner-purchased hardware: one Pololu 4-channel RC servo signal multiplexer, assembled #2806

**Purchase update — owner-confirmed 2026-10-30:** **ONE assembled Pololu 4-Channel RC Servo Multiplexer, model #2806, has been purchased.** This is the correct quantity to select PWM for both steering servos; only two of its four mux channels are needed. The partial soldering kit **#2807 was not selected**. **Do not buy another multiplexer.** Delivery, installation, and hardware testing remain unconfirmed.

- **Purchased item:** 1 × [Pololu assembled 4-Channel RC Servo Multiplexer #2806](https://www.pololu.com/product/2806).
- **Purchase status:** Bought by owner; not yet physically verified installed. Earlier shopping comparisons are historical and are **not** open purchase recommendations.
- **Manufacturer setup/pinout guide:** https://www.pololu.com/docs/0J60/all

### Proposed functional signal topology (NOT a final wiring diagram)

```text
Pi 5 PCA9685 steering LEFT PWM  ──> M1  ┐
Alta PCA9685 spare LEFT PWM     ──> S1  ├── OUT1 signal ──> left steering servo PWM input
Pi 5 PCA9685 steering RIGHT PWM ──> M2  ┐
Alta PCA9685 spare RIGHT PWM    ──> S2  ├── OUT2 signal ──> right steering servo PWM input

Alta PCA9685 spare RC PWM selector ──> SEL
  normal: selects Pi/master; Pi failed + Alta responsive: selects Alta/slave.
  Pi healthy + Alta failed: invalid SEL defaults to Pi/master only if
  the manufacturer's FAILMODE configuration is left in its master-default state.
```

**Important:** the Pololu **SEL** input expects **valid hobby RC pulses**, not simply a static GPIO level: **10–330 Hz, 0.5–2.5 ms pulse width**, default slave selection above roughly **1700 μs**, with a configurable threshold/direction. A spare Alta PCA9685 channel could generate SEL at 50 Hz while Alta's local steering backup channels produce neutral. Test the exact state and logic, rather than assuming a stuck or invalid selector produces a safe steering position. Manufacturer states that when SEL is invalid, the **FAILMODE** jumper selects either **master source** (jumper OFF) or **outputs forced low** (jumper ON); low outputs do not guarantee servo neutrality. The board **does not have a safe-start feature**.

**Steering-servo power:** The owned 150 kg servo class can require high current and a supply separate from controller logic. **Do not pass their unverified high-current power supply through the multiplexer board's VM/servo-header power traces.** Route selected **PWM signals only** to servo signal pins, with ground reference and a separately validated low-current board-power arrangement that does not backfeed the other SBC or the servo power supply; verify board logic and input levels with the actual installed devices. The manufacturer's ordinary RC wiring diagrams assume shared servo power through VM and therefore **cannot be copied directly** to this high-current mower wiring.

**This product is a candidate for a bench-tested signal selector only, not an automotive safety-rated failover device.** It switches signals but cannot guarantee neutral if the Alta also fails, if servo power is lost, if a drive linkage sticks, or if its selector fails. An I²C/PCA9685 process freeze may leave a previously commanded pulse active; both modes must be fault-injected and actual lever positions measured before an engine-on trial.

## 3. Selectable peer transport

Support **Mode A: Wi-Fi HaLow**, or **Mode B: a direct Ethernet cable connecting Pi 5 and Alta**. In Mode B, **all** Pi–Alta peer packets (heartbeats, state, fault messages, coordination, acknowledgments and telemetry) use Ethernet; HaLow can independently serve base station, Thor and the remote UI. The software must explicitly configure which interface/IP is the active peer transport; **do not silently fail back to HaLow** on Ethernet loss without a separately specified policy.

Use the same application-level monitoring and stop logic on either transport. Peer monitoring traffic is tiny relative to either link's capacity.

## 4. Heartbeats and fault detection

- Each computer independently sends a heartbeat **every 200 ms (5 Hz)** and monitors the other. Typical healthy **application payload target: ~24 bytes**, potentially larger if authentication/anti-replay requires it. At 24 bytes, application heartbeat payload = **120 bytes/s = 960 bits/s per direction**, before headers and retries.
- Include protocol version, sender, boot/session identifier, increasing sequence number, current state/authorization and fault flags. Authenticate the transport/messages and reject invalid, stale, duplicated or replayed data. Measure freshness using **local monotonic receive time**, not wall-clock synchronization.
- **Three missing heartbeat opportunities (~600 ms)** is the initial configurable failure detection candidate, **not an assured deadline**. The software fault detector must also respond to an explicit fault packet, service-health failure or incompatible protocol state. Avoid premature false triggers by bench-measuring jitter and scheduler stalls.
- Send normal dashboard/telemetry updates separately, approximately **1 Hz**. Loss of web UI or Thor does not prevent the surviving local computer from issuing shutdown commands.

## 5. Software states

`BOOT_DISARMED` → `PEER_VALIDATED` → `RUN_AUTHORIZED` → `FAULT_LATCHED`.

- **BOOT_DISARMED:** No automatic engine start, PTO engagement, throttle advance or motion commands. Confirm local hardware services, peer health and authorized operator request before arming.
- **PEER_VALIDATED:** Bidirectional fresh heartbeat established, outputs known and neutral/idle/OFF requests accepted; no motion yet.
- **RUN_AUTHORIZED:** Pi 5 is the sole normal command authority. Alta monitors but does not send contradictory active drive commands.
- **FAULT_LATCHED:** New starter, PTO-ON, motion and throttle-up commands rejected. The appropriate host issues its own available stop commands; fault remains latched after peer reconnect/reboot until a deliberate human re-arm following inspections and functional checks.

If a link is partitioned with both computers alive, **both enter FAULT_LATCHED** instead of electing competing masters. A recovered connection never automatically reauthorizes mowing.

## 6. Emergency shutdown request — software logic

**If Libre Alta fails: Pi 5 acts.**  
**If Raspberry Pi 5 fails: Libre Alta acts.**

When a peer is declared failed, the survivor locally performs the following **without waiting for the peer, Thor, web UI or network command acknowledgment**:

1. Atomically set `run_authorized = false`, `starter_allowed = false`, `pto_allowed = false`, `movement_allowed = false`; latch the fault and record the cause.
2. Immediately request **PTO OFF** and **ENGINE OFF** via its *own electrically accessible, independently operable* control interfaces.
3. Concurrently request **both steering levers to NEUTRAL** and **throttle to IDLE** **only on interfaces the survivor actually controls**. Do not delay PTO/engine OFF while waiting for servo repositioning. If the survivor lacks steering authority, report **NEUTRAL_NOT_COMMANDABLE**, not success.
4. Verify whatever physical feedback actually exists (engine RPM, PTO/clutch state, steering position/neutral sensors). A transmitted GPIO/I²C command is **not proof** of clutch release, blade cessation, or neutral.
5. Continue publishing `FAULT_LATCHED` and actionable diagnostics. No automatic restart or blade re-engagement even if the other computer boots or radio/Ethernet recovers.

**Do not** treat cutting 5 V servo power, disabling PCA9685 PWM, or killing the engine as equivalent to mechanically returning both hydrostatic lap bars to neutral. The parked/braked position requires outboard movement that the documented fore/aft steering linkage does not perform.

## 7. Simplified software pseudocode

```python
# Conceptual control flow; not deployable mower control code.
HEARTBEAT_PERIOD_S = 0.200
MISSING_LIMIT = 3
PEER_TIMEOUT_S = HEARTBEAT_PERIOD_S * MISSING_LIMIT

def on_cycle(now):
    send_authenticated_heartbeat(local_state, sequence_number)
    if local_health_failed() or peer_last_valid is None        or now - peer_last_valid >= PEER_TIMEOUT_S:
        enter_latched_stop(reason="PEER_OR_LOCAL_HEALTH_LOST")

def enter_latched_stop(reason):
    revoke_all_run_start_motion_and_pto_permissions()
    record_fault(reason)
    request_pto_off_via_my_local_stop_output()
    request_engine_off_via_my_local_stop_output()
    if my_steering_controller_is_independently_accessible():
        request_both_steering_levers_neutral()
    else:
        report("NEUTRAL_NOT_COMMANDABLE")
    if my_throttle_controller_is_independently_accessible():
        request_throttle_idle()
    publish_latched_fault_without_waiting_for_peer()
    require_deliberate_local_rearm()
```

Implementation must use independent send/receive scheduling, nonblocking shutdown requests, a durable state latch, defined timeout/fault precedence, and authenticated control messages. The pseudocode does not address operating-system service crashes when its own process cannot execute.

## 8. Software packages / minimum tests

**Modules:**
- `peer_transport`: HaLow/Ethernet selection, binding, message authentication, sequences.
- `peer_health`: 200 ms transmit and receive age, duplicate/stale rejection, timeout events.
- `control_arbiter`: single Pi normal-authority policy; no split-brain actuator commands.
- `stop_coordinator`: common latched-fault procedure in both hosts, with explicit capability flags for local engine, PTO, steering and throttle interfaces.
- `device_adapters`: *separate* Pi steering-PCA9685 adapter, Alta throttle-PCA9685 adapter, and per-host verified engine/PTO stop interfaces. Do not silently assign absent backup GPIO.
- `status_api`: UI status, fault reason, last peer time, output command acknowledgment vs. observed physical feedback.

**Software tests (simulation only):** healthy at 5 Hz; Alta dies; Pi dies; one-direction packet loss; full disconnect; replayed or malformed heartbeat; peer restart and boot-ID change; software service freeze; repeated STOP; arbitration conflicting master; capability missing; reboot persistence. Test Ethernet and HaLow configurations separately. Assert no automatic rearm and that fault outputs are requested by the *surviving* computer.

**Bench verification still required before mower operation:** a disconnected controller must not prevent the survivor from requesting OFF through the actual electrical interfaces; tests must distinguish a command from an observed shutdown. Real neutral behavior, engine-kill polarity, PTO release, blade coast-down, starter interlocks and hardware power ratings remain unresolved.

## 9. Existing inventory and additional purchase implications

- **Already purchased/owned:** Pi 5, Alta, two PCA9685 boards, two HiLetgo single-channel relays, 150 kg steering servos, MG995 throttle servo, existing servo and SBC converters, automotive relay harness, **one assembled Pololu #2806 four-channel servo multiplexer (owner confirmed purchased 2026-10-30; delivery not verified)**.
- **No additional external hardware watchdog or added E-stop switch** is included in this *user-selected* software proposal.
- **Not yet proven sufficient:** independent engine and PTO switch contacts for both computers, correct steering-source selection wiring and operation through purchased Pololu #2806 under Pi failure, regulated MG995 supply, 5 V relay coil supply, and OEM-compatible starter/ignition control. **Software cannot create an extra relay pole or a missing signal wire.** Do not finalize a purchase list until actual model/serial pinouts, power ratings and interface counts are verified.

**Authoritative cross-references:** [Owner-selected primary/backup roles](2026-10-30-primary-backup-software-control.md), [hardware allocation](2026-10-30-integrated-hardware-and-control-decisions.md), [Ethernet/HaLow alternative](2026-10-30-intercontroller-ethernet-alternative.md).  
**Status:** Documentation only — not implemented, integrated, or verified as safe for unattended mowing.
