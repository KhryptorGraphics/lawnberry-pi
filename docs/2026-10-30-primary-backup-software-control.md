# Pi 5 primary / Libre Alta backup — software monitoring and shutdown authority

**Decision recorded:** 2026-10-30  
**Status:** Owner-selected functional architecture, **not implemented or physically verified**. This is a control-design decision, **not certification of a safety system**.  
**Mower:** Toro 77603 (serial-specific wiring remains unverified).

## Primary / backup authority — symmetric single-computer fault response

The **Raspberry Pi 5 is the primary control computer**. The **Libre Alta is a running backup controller** with its own software process and a physically independent means to request that the engine and blade PTO be turned off. Both must be able to detect peer unavailability and command their *own* accessible stop interfaces. **Neither may need the failed controller to relay the shutdown command.**

| Condition | Detecting / acting computer | Required software action |
| --- | --- | --- |
| Both healthy | Pi 5 primary; Alta standby | Pi controls normal mower operation. Alta monitors peer status and is prepared to issue shutdown, rather than take over mowing automatically. |
| **Libre Alta stops responding, crashes or loses power** | **Pi 5** | Pi detects missing Alta heartbeats, withdraws run authorization, commands **PTO/blades OFF** and **engine OFF**, blocks starter and motion requests, and reports the fault. |
| **Raspberry Pi 5 stops responding, crashes or loses power** | **Libre Alta** | Alta detects missing Pi heartbeats, withdraws run authorization, commands **PTO/blades OFF** and **engine OFF** through outputs Alta controls directly, blocks restart, and reports the fault. |
| Intercontroller link fails with both hosts alive | Both locally | No assumption that the peer is healthy: each locally revokes run authorization and commands the available shutdown interfaces. Do not wait for a STOP message from the other host. |
| Both hosts fail, shared electrical failure, actuator/relay sticks, or mechanical linkage jams | No functioning software path is guaranteed | **Not covered by software redundancy alone.** Real-world stop outcomes are not guaranteed. Do not report this case as handled, negligible, or validated. |

## Communications

- Intercontroller traffic is selectable: **Mode A — Wi-Fi HaLow**, or **Mode B — direct Ethernet carrying *all* Pi–Alta communications**. See [parallel interconnect design](2026-10-30-intercontroller-ethernet-alternative.md). HaLow radios may remain in use for off-mower connectivity when Ethernet is selected.
- **Bidirectional heartbeat every 200 ms** (5 Hz each), with **three missed intervals (~600 ms)** as a first design target, *not* a measured safety stopping time. Loss detection uses monotonic receive timestamps and rejects old, repeated, unauthenticated or out-of-order packets.
- Software fault monitoring and the shutdown command path must operate locally; the web interface and NVIDIA Thor are not necessary to issue a local request to shut down.

## Output arbitration and electrical interface — unresolved

- The **Pi and Alta must each have an independent, electrically appropriate way to revoke engine RUN and blade PTO permission**. Two GPIO pins must **never** be wired together as if they were interchangeable outputs.
- Shared **fail-to-stop / stop-dominant** relay logic must be defined from the **actual mower schematic**. An appropriate interlock may need series permissions, separate rated contacts, or an isolation stage; **do not assume the two existing single-channel HiLetgo relays are sufficient** for separate PTO and engine control, backup authority, and starter lockout.
- The preexisting PCA9685s do not drive starter/PTO circuits and can hold their last PWM outputs if Linux or I²C stops. Shutting down the engine and disengaging the PTO **does not prove steering drive levers return to neutral or the blades instantly stop spinning**. Confirm actual mechanical outcomes rather than crediting a software command as a complete stop.
- The current owner request omits **additional external watchdog modules** and an **added physical emergency-stop switch** from this proposed architecture. This does not imply those components are unnecessary for safe operation or that OEM protective circuits may be bypassed. Preserve the manufacturer's existing machine protection functions.
- Do not make an unattended operation claim until serial-matched electrical interfaces, single-fault command availability, drive neutrality, engine stop, clutch release and blade coast-down have been verified using appropriate isolated test setups.

## Software implementation targets

1. One common, versioned **peer-health protocol** shared by Pi and Alta, transport independent (HaLow or Ethernet).
2. A local state machine on each host: `DISARMED` → `PEER_HEALTHY` → `RUN_ALLOWED` → fault `STOP_REQUESTED` / latched `REARM_REQUIRED`. An automatically restored heartbeat never resumes engine/PTO.
3. Pi normal operation; Alta backup monitors peer and has **local outputs** needed to request OFF even if Pi has crashed and its I²C/PCA9685 board remains frozen.
4. Explicit relay/authorization truth table and a wiring diagram based on measured, serial-matched hardware; validate default GPIO startup, power-off and floating-input behavior.
5. Unit and bench-in-the-loop tests for each single-host power failure, freeze, restart, stale packets, unilateral network loss and failed relay feedback. Record command response plus **observed physical** engine/PTO/drive effects; hardware-actuation timing is not derivable from 600 ms heartbeat timeout alone.

**Boundary:** This document records the owner-preferred redundant *software supervisory* architecture, replacing superseded prescriptive safety-spec files on this feature branch. It does not itself establish an independent safety-rated emergency stop.
