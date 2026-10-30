# Toro model 77603 PTO / blade switch research and integration
Date: 2026-10-30
Status: Research and design handoff, **not a verified wiring installation procedure**.

## Identified platform and references
- User-reported model from ID label: **Toro TimeCutter Max Havoc MyRIDE 60-inch, model 77603**. Exact serial number remains unverified.
- Official Toro operator publication **3473-450C**, with model-specific electrical diagram in Chapter 8, page 8-1, as identified in the project research.
- Official manual source: https://www.toro.com/getpub/273204
- Do not check the entire third-party copyrighted manual into this public repository. Access the manufacturer's official hosted publication instead; check the link remains current.

## 2026-10-30 manufacturer illustration cross-check — CORRECTION (HIGH CONFIDENCE)
The manufacturer's actual Toro 3473-450C operator manual, p. 3-2 (PDF physical page 17), **labels the round pull knob above the ignition as CHOKE control, NOT PTO.** The rectangular switch near the lower-left of the console is labeled **Blade-Control (PTO)**. The diagram on p. 3-3 explicitly shows the blade-control as a rocker/rectangular control. **Prior conversation descriptions identifying the round knob as PTO were wrong and must not be used for mechanical actuator or electrical wiring work.**
- Manual control-panel illustration: https://www.toro.com/getpub/273204#page=17 (print p. 3-2).
- Blade-control illustration: https://www.toro.com/getpub/273204#page=18 (print p. 3-3).
- Electrical schematic: https://www.toro.com/getpub/273204#page=87 (print p. 8-1, drawing G454156).
- Original user's photo shows a round control immediately above the key consistent with the choke, plus rectangular panel switches. The rectangular lower-left switch is the PTO **by visual layout comparison**, but its underside pins are not visible. A yellow upper-left switch appears to have a light/brightness marking and is not established as PTO.
- Toro parts catalog for model 77603 (serial 418961000 and up), Form 3476-743A, electrical assembly printed pp. 26-27, lists ref. 7 = **139-6368, Switch-Rocker, On-None-On**, ref. 4 = 110-6765 (2 switches), and main wiring harness ref. 1 = 145-6382. Parts catalog: https://www.toro.com/getpub/265916#page=27 . Note that part selection remains serial-dependent; do **not** assign pin numbers to the mower using this catalog alone.
- Toro operator manual p. 4-4 states startup is interlocked so PTO must be disengaged and motion control levers parked; it also describes an engine-off interlock involving seat occupancy and the motion-control levers. Do not bypass any such safety function.

**Pinout verification status: still NOT physically confirmed.** The printed diagram identifies the functional circuit but does not document connector orientation from the photo. Get the full serial number and a clear straight-on picture of the rear of the rectangular PTO rocker *with its molded pin labels and plug/wire colors visible*, taken with key removed, engine off, and battery negative disconnected by a qualified technician. Continuity/function checks must use a serial-matched service diagram. Do not transfer terminal numbers from older Toro TITAN/SS/MX service manuals to this 77603.

## What is known
- The blade engagement/disengagement function is electric PTO clutch control, not mechanical movement of the mower blades by a servo.
- The PTO operator control is associated with the engine-start / safety interlock logic.
- The accessible control-panel photograph helps locate the user-facing PTO knob/switch but **does not reveal its connector pinout, switch contact map, clutch coil polarity, or the exact serial-range wiring**.
- Electronic removal of PTO-clutch power can eliminate the latency of a mechanical servo moving a physical button. **Power removal is not equal to instantaneous blade stop:** account for mechanical spin-down.
- Prior proposed single normally-open relay **in series with the clutch feed, downstream of retained OEM interlocks**, only grants/removes clutch power; it **does not independently engage** the PTO when the operator's existing switch is off. Treat that as a provisional concept, not approved wiring.

## Minimal concept for feasibility comparison (unverified)
```
Pi/SBC blade-allow request
    -> isolated, correctly rated interface / normally-open switching device
    -> series interruption of OEM-authorized PTO-clutch supply
    -> OEM electric clutch
OEM PTO switch, safeguards and engine-kill chain remain independent.
```
- On loss of controller power or signal, the added blade-permission contact defaults **open**.
- Keep OEM seat, brake, reverse, start, PTO, engine and other interlocks operative. Do not bridge, defeat or emulate safety contacts to force the blades on.
- Use hardware E-stop/engine-kill path independent of Linux, plus a watchdog where appropriate.
- Relay/module must be sized for **measured clutch steady-state current and inductive release transients**; proper suppression and isolation are required. No unverified GPIO-direct or guessed terminal-to-terminal wiring.
- Blade control from SBC must not be treated as the sole protective layer for detecting people, pets, or obstacles.

## Dual-controller owner requirement (2026-10-30)
The current Pi 5 primary / Libre Alta backup architecture requires **both** hosts independently to request **PTO OFF** after the other fails. A proposed **two-contact series, energize-to-allow PTO permission logic** can satisfy this at the functional level **without bypassing factory interlocks**. It does not itself provide independent PTO ON when the OEM PTO rocker is OFF. With one of the two owned one-channel HiLetGo modules reserved for steering, **only one such PTO permission contact is accounted for**; the existing automotive relay harness may or may not include an installed second relay. **Do not mark the dual contact electrical interface as complete** until actual relay bodies, ratings and mower serial-specific wiring are checked. The Pololu #2806 is for steering PWM only. See [dual-controller PTO electrical feasibility, relay inventory and tests](2026-10-30-dual-controller-pto-circuit-feasibility.md). No separate new watchdog or physical E-stop is part of the owner-selected *software-supervisory* scope, but that choice does not prove safe unattended operation.

## Decisions to verify against serial-specific Toro service information
1. Confirm serial number and obtain the **exact** electrical/parts wiring revision for the machine.
2. Identify switch connector, pin numbering, switch contact states, and whether it is momentary, maintained, multiplexed or part of a controller module.
3. Verify whether the final control should (a) only permit cutting when manually selected, or (b) request autonomous engagement. These are electrically different architectures. **Do not design an interlock bypass to achieve option (b).**
4. Independently verify clutch voltage, current, inductive transient limits, factory fuse and harness topology under supervision of a qualified technician.
5. Bench-test using a dummy load with the engine off before an authorized supervised live integration test.
6. Validate loss of Pi/SBC power, boot/reboot, hung process, broken wire, relay failure, disconnection, emergency stop, interlock changes and blade spin-down.
7. Record switch photos, wiring connector photos and an as-built trace in docs/media manifest before assigning any numbered pins.

## Media inventory
- One original control-panel photo was made available to this session; binary upload to GitHub is **not yet verified**. Do not mark it uploaded or infer the four additional user-mentioned photos are present.
- On eventual media import, put original photos under `docs/media/2026-10-30/`, preserve original bytes and names, and add a manifest mapping image to subject/location and related design requirement.
- Links to nonexistent media files must not be added as if committed.

## Deliverable boundary
This document provides research status and a safe, simple conceptual option. It is **not** a pin-by-pin wiring diagram or an authorization to build an untested blade controller.