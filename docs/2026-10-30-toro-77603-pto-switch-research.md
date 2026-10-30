# Toro model 77603 PTO / blade switch research and integration
Date: 2026-10-30
Status: Research and design handoff, **not a verified wiring installation procedure**.

## Identified platform and references
- User-reported model from ID label: **Toro TimeCutter Max Havoc MyRIDE 60-inch, model 77603**. Exact serial number remains unverified.
- Official Toro operator publication **3473-450C**, with model-specific electrical diagram in Chapter 8, page 8-1, as identified in the project research.
- Official manual source: https://www.toro.com/getpub/273204
- Do not check the entire third-party copyrighted manual into this public repository. Access the manufacturer's official hosted publication instead; check the link remains current.

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