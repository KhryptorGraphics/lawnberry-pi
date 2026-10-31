# Claude Code — executable hardware CAD and dual-controller PTO design prompt
**Project:** Lawnberry Pi, 2026-10-30. **Status:** Paste-ready implementation task for Claude Code, not work already executed by an agent.

Copy the text below into Claude Code after opening the local repository. It is written to generate actual, editable geometry and a reviewable, provisional electrical design.

---

You are Claude Code, acting as lead mechanical CAD, mechatronics and electrical-systems engineer on **Lawnberry Pi**. **Start implementing hardware model deliverables, not merely writing a plan.** Work in the existing repository:

- Repository: https://github.com/KhryptorGraphics/lawnberry-pi
- Source requirements branch: `main` (latest owner-approved integration scope)
- First inspect `git status`, fetch, switch to current `main` and create a **new feature branch based on it**, preserving working tree changes, existing geometry and documentation. Do not commit onto the default branch or delete previously authored files.
- If unable to run CAD tooling, record the missing dependencies honestly and still implement valid parametric source and tests where possible. Never claim STLs/renders/STEP exports exist unless generated and inspected.

## Read and reconcile requirements BEFORE coding

Read in order:
1. `docs/2026-10-30-user-confirmed-owned-hardware.md`
2. `docs/2026-10-30-integrated-hardware-and-control-decisions.md`
3. `docs/2026-10-30-software-only-dual-controller-failover-design.md`
4. `docs/2026-10-30-primary-backup-software-control.md`
5. `docs/2026-10-30-throttle-servo-printable-variants-addendum.md`
6. `docs/2026-10-30-throttle-right-steering-offset.md`
7. `docs/2026-10-30-throttle-servo-3d-revision.md`
8. `docs/2026-10-30-claude-code-throttle-cad-revised-prompt.md`
9. `docs/2026-10-30-toro-77603-pto-switch-research.md`
10. `docs/2026-10-30-dual-controller-pto-circuit-feasibility.md`
10a. `docs/2026-10-30-followup-hardware-research-findings.md` (**mandatory new manufacturer-verified corrections: Kawasaki ground-to-kill, 1-643275 OEM relay not spare, Pololu selector fallback, Seeed SPI-only Mini PCIe, Quectel RTK option, four JSN-SR04T sensors and Raspberry Pi physical pin 17 correction**)
11. `hardware/mounts/README.md`, `INSTALL.md`, `assemblies.scad`, `enclosure_common.scad`, other relevant `*.scad`, and `build_printables.py`.

These documents contain corrections and supersede legacy model **77502** assumptions. **Actual mower is Toro TimeCutter MAX Havoc MyRIDE 60-inch, model 77603**; serial unknown. The existing 77502 CAD is useful as a baseline **only**, not proven to fit the 77603. Do not reinstate deleted obsolete acceptance/safety documents.
**Highest-precedence owner update, 2026-10-31:** manual key start and manual key stop ONLY; blade PTO-OFF remains dual-computer independent; no new park-swing hardware, starter GPIO/relay, engine kill interface, or remote ignition dashboard controls. Treat earlier engine stop/start instructions in referenced historical documents as superseded.

## Owned hardware — don't order duplicates

- Raspberry Pi 5 (primary) and Libre Alta 4 GB (hot standby), each with own power converter; independent bidirectional heartbeats at 5 Hz (200 ms) over selectable direct Ethernet or Wi-Fi HaLow.
- Two PCA9685 16-channel PWM boards: Pi board normally controls two 150 kg-class steering servos; Alta board controls MG995 throttle and can generate backup steering PWM with spare channels.
- **ONE purchased, assembled Pololu 4-channel RC Servo Multiplexer, SKU 2806**: use channels 1 and 2 to select between Pi and Alta steering PWM for both servos. Do NOT buy a second Pololu, and do not confuse it with engine/PTO control. Its SEL is an **RC pulse input**, not a static digital GPIO input. Follow manufacturer guide: https://www.pololu.com/docs/0J60/all . Engineer an explicit default/boot/host-failure source-selection truth table and separate validated logic supply. Do not route high-current 150 kg servo power through mux board VM traces.
- Two purchased HiLetGo 5 V, 30 A **single-channel** relay boards, provisionally one intended for steering-servo power; one purchased automotive relay harness with unverified relay/sockets/contacts. Two steering-servo power converters, two SBC USB-C converters, MG995 throttle servo and X horn also owned.
- Do not connect two SBC GPIO outputs together or assume two single-channel relays suffice for independent PTO permissions and steering-power needs.

## Workstream A — produce genuine parametric 3D CAD

1. Inventory existing source/exports under `hardware/mounts/`; identify reusable right-side steering-servo mount and attachment datums. **Preserve left/right steering actuation, original 150 kg servos, pushrod/ball joint assumptions and their mounting interfaces**; this is an **additive throttle revision**, not a complete redesign.
2. Add an independent, elevated, removable **MG995 throttle mount** to the right-side mounting system. User MG995 provisional measurements: body **1.50 in high, 0.50 in wide, 1.50 in long**; shaft projects about **0.50 in**; mounting-hole centers **2.00 in** front-to-back, at **1.125 in** above body base. Mark shaft X/Y, holes, horn arm radius and actual lug dimensions **TO MEASURE**, do not invent exact fit.
3. Five complete editable throttle linkage alternatives, matching the detailed design documents: **A** direct existing X-horn arm; **B** 3D-printed horn reinforcement adapter; **C** continuously adjustable effective horn radius; **D** positively retained printable ball/socket with replaceable insert; **E** double-shear captured printable clevis with captive printed pin/bushing. Each includes full mounting and linkage assembly.
4. All **new** linkage, joints, pushrod connectors, adjustment and retaining hardware must be **3D-printable**; no new bought metal bolts, screws, nuts, ball links, pushrods, clevis pins or threaded inserts. Retain the existing purchased MG995, X horn, and stock mower throttle control. If all-printed parts cannot meet realistic durability/retention needs, explicitly report the limitation and flag as unvalidated rather than secretly adding metal or asserting safety.
5. Build a **continuously sliding**, **micro-fine length-adjustable**, completely printable pushrod lock, with captive anti-separation and adjustable overlap. No indexed-hole adjustments or metal clamp bolt. Printability within Anycubic Kobra 2 Max **420 × 420 × 500 mm** build volume (nozzle max 260°C, bed max 95°C); split long rod segments, add printable positive capture, never rely on glue-only retention. Evaluate PETG/ASA compatibility and environmental degradation; label assumptions.
6. Throttle interface is an inverted-U/open-bottom printed fitting. **Initial INSIDE opening is 12.70 mm front/back × 31.75 mm left/right** (not exterior). Rightward throttle lever tilt is real but angle unknown: parameterize `throttle_lean_right_deg` (and yaw/pitch if necessary); provide opening test gauge. The throttle lever is **2.5–3.5 in RIGHT** of the right steering bar at idle and **3.5–4.5 in RIGHT** fully forward. Existing 40/44-inch endpoint distance estimates have ambiguous datums; **do not construe them as a verified 4-inch stroke**.
7. Generate editable parametric OpenSCAD CAD **and** if available valid STEP exports; generate printable STL parts, five assemblies, five exploded/annotated views and small fit gauges, only when tools exist. Establish build scripts, deterministic exports and geometry checks; flag actual mower-fit measurements outstanding.

## Workstream B — electronics and backup steering packaging

8. Create an electronics placement/enclosure model for **one Pi 5, one Libre Alta, two PCA9685 modules, one purchased Pololu #2806 and owned relays/converters**, with cable clearances, vibration mounting, weather sealing assumptions, fusing/cable routing and access. Existing case geometry may be adapted, but don't claim OEM machine dimensions verified. Do not add watchdog/E-stop components contrary to current owner-selected scope.
9. Produce a **non-deployable schematic/block diagram** showing: Pi steering PWM → Pololu M1/M2; Alta backup steering PWM → Pololu S1/S2; Pololu OUT1/OUT2 → steering servo **signal** leads; independent high-current power to each servo; Alta MG995 PWM separate. Document signal source switching, RC-pulse SEL requirements, failure modes and truth table including Pi freeze/PWM held, Alta crash, broken communications, and mux loss of supply. A software stop request does not prove mechanical neutral.

## Workstream C — verify independent blade PTO-OFF interface; no engine start/stop automation

10. Review Toro model **77603** official operator manual https://www.toro.com/getpub/273204 and parts portal, but select the exact wiring revision using serial number **when available**. Correct console identification: rectangular rocker is PTO; round knob above key is CHOKE. Do not invent rocker pins, wire colors, polarity, contact ratings, or direct installation instructions.
11. Requirement: if Pi 5 crashes **while PTO is engaged**, Libre Alta must independently command blade PTO OFF, and vice versa. **The Pololu mux has nothing to do with blades.** Prototype a **conditional**, OEM-interlock-preserving PTO power-permission concept: a **normally-open, energize-to-allow contact owned by Pi** in series with an **independent normally-open, energize-to-allow contact owned by Alta** in the *factory-authorized* clutch supply path. Either computer withdrawing its own permission must open the series path. Represent this at the **functional block** level only; do not finalize connections until serial-specific OEM diagram, clutch current/inductive transients, existing relay harness contacts and supplies are verified. OEM PTO switch OFF must always keep blades OFF. A series permission system **does not autonomously engage** PTO when OEM switch is OFF.
12. Under current proposed allocation, a HiLetGo relay dedicated to steering leaves at most **one** HiLetGo relay for PTO. Independently controlled **two PTO permission contacts** are necessary for this candidate architecture; determine whether the **owned automotive relay harness includes an appropriate installed relay/contact**. If not, explicitly list missing contact/relay and driver as **conditional purchase**, not an already-owned feature. Neither GPIO pins nor a PCA9685 output replace switching contacts. Verify common power failure and contact-sticking limitations. Note manufacturer's relay marketing '30 A' does **not** by itself establish DC inductive clutch compatibility.
13. The owner has **removed all electronic starter and engine-kill integration**. Leave original Toro key switch, manual PARK lever motion and OEM safety interlocks in place. Do not generate starter/engine-stop relay layouts, GPIO allocations, software endpoints, UI buttons, purchases, or automation requirements. Keep factory engine shutdown functionality unchanged.
14. Implement **offline/simulation** `request_pto_off()` for each host with peer timeout and latched-fault tests. Test healthy, either-host failure, heartbeat split, reboot, stuck contact and power loss. PTO OFF must be requested immediately, concurrently with independently attainable neutral/idle. No software engine-stop claim.
15. Write a wiring verification matrix: `KNOWN FROM SOURCE`, `HYPOTHESIS`, `TO MEASURE`, `BLOCKED`; label connection and current measurements with the required photo/serial proof. Add dual-controller PTO logic drawing with generic contacts, fuse, OEM interlock retained, not terminal assignments. Simulate with an isolated low-voltage dummy load and state that actual clutch release and blade coast-down need separate supervised physical validation.

## Deliverables required on your new feature branch

- Real modified/new `.scad` (or other editable CAD) files for each throttle variant, base mount and selectable electronics packaging; reproducible regeneration script; actual STL exports and images only if rendered successfully.
- Functional **two-independent-PTO-contact** block diagram/truth table, relay inventory sufficiency assessment, unknowns/measurement register, and bench test plan; no guessed Toro 77603 switch pinout.
- Zero duplicate purchases of assembled Pololu #2806 or other confirmed existing items.
- Tests and at least one actual run of any available rendering/geometry scripts, with outputs and honest failures documented.
- Commit the resulting files to your feature branch, report commit SHAs and changed paths. End with **Completed / Blocked on measurement / Requires bench testing / Conditional purchases**. If some deliverables are blocked, complete independent CAD/model work now; don't stop for broad questions.

START by surveying the repository and implementing the throttle CAD and two-contact PTO block model. Keep moving through unblocked tasks and produce files, not just recommendations.
