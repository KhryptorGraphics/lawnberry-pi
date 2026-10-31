# Toro 77603 — simplest lockable lateral-swivel joints for connected steering pushrods

**Owner direction: 2026-10-31.** **STATUS: specification only for later Claude Code CAD; no modified CAD/STL, structural verification, or installed mechanism is claimed.** The user expressly asked for the simplest practical redesign, not to generate the 3D model yet.

## Goal and retained scope

Keep **both existing steering pushrods connected at BOTH ends**, even while manually swinging the Toro 77603 lap bars *outward to PARK* and back *inward to DRIVE / fore-aft neutral*. The user wants the added lateral movement **lockable at any chosen angle**, particularly inward/centered, while preserving normal forward/reverse steering actuation. Keep the existing two high-torque steering servos, rear mounts, adjustable pushrods, original four-bolt lap-bar clamps, and Pololu #2806 selection. Change only the *end joints/adapters* that require more articulation. Do NOT add powered PARK-swing actuation, starter wiring, engine-kill wiring, or autonomous engine start. Operator still manually swings bars to PARK and uses OEM key to start/stop. Retain OEM PARK and PTO interlocks and independently controlled blade PTO-OFF paths.

**Important baseline:** Existing `hardware/mounts/INSTALL.md` and `README.md` have a simple flat rod eye in a clevis on an M6 pin at **each** end; those aligned pin axes permit mostly planar fore/aft motion, **not independent outboard PARK swing**. Their rod-length geometry (936.4 mm and 4.12-degree skew) is an **unverified estimate for Toro 77502**, not a measurement for the user's **77603**.

## Preferred MINIMUM-parts concept: two-axis articulated ends, independent lateral locks

There are **four identical-function joints total**: steering-bar end and servo end for each of left and right pushrods. At EACH joint:

1. **Keep the existing M6 captured fore/aft hinge** (or equivalent reoriented hinge) as the permanently FREE steering-axis pivot. It must still swing throughout forward–neutral–reverse even when the lateral lock is engaged.
2. Add exactly **one second, orthogonal, captive lateral swivel axis** (a short fork/yoke on a pivot pin, preferably reusing existing clevis/rod-eye interfaces). This is the axis that follows manual inward/outward lap-bar PARK movement.
3. Put **one adjustable split-collar pinch screw/clamp** on the lateral-swivel axis, operable by hand or common tool. Loosen = lateral axis freely swings but remains mechanically captured. Tighten = friction-clamps that **lateral axis only** at *any* angular setting (not just pre-drilled PARK/DRIVE index positions). If geometry or load makes friction locking inadequate, report that clearly; do NOT claim this is a rated positive interlock or parking brake.
4. Use **captured pins with retained nuts/keepers**. No joints or rod ends may pop off on steering, vibration, lateral swing, servo overtravel, or loss of power. Prefer to reuse the existing M6 steel pin/fastener and existing printed features where physically possible. A load-bearing all-printed friction screw or snap-fit alone is not acceptable without tests.
5. **DO NOT lock the fore/aft steering pivot.** Locking a normal ball joint in all directions would block steering; do not substitute a single fully clamped spherical joint for the two-axis design.

This is a **functional kinematics prescription**; the precise axis orientations, pin spacing, and clevis bracket revisions must come from measurements and CAD motion studies, not guessed values. Favor the smallest modifications to rod-end eyes and clevis sleeves, and avoid introducing a third axis or telescoping stroke unless the measurements prove it necessary.

## How the mechanism is intended to be used

- **Manual startup:** blade PTO OFF; steering servos idle/unpowered as appropriate; release *both* lateral-axis clamps; manually swing both steering bars OUT to OEM PARK **without disconnecting either pushrod**; start engine manually using the original key and factory interlocks.
- **Ready to drive:** manually swing bars IN to center/fore-aft neutral; with confirmed alignment, tighten the lateral-axis clamps to hold the chosen inboard lateral orientation. Normal steering must still be possible via the *free fore/aft hinges*.
- **Lock-any-angle request:** the split clamp is continuously adjustable and may clamp the lateral hinge at another chosen angle when engine/blade are safely off. **Do not treat clamp friction alone as a mechanism that positively holds the entire lap bar against all lateral loads or replaces the OEM PARK latch/brake.**
- **Manual shutdown:** stop motion, blades OFF, return to a hand-controlled safe state; release lateral clamps to move bars OUT to PARK if needed, then stop engine with OEM key. Do not claim engine stop is automated.
- **Failover:** Pi 5 / Libre Alta retain their steering PWM selection and independent blade PTO-OFF request logic. Swivel locks must not impede real emergency steering-to-neutral motion. A software request is not proof of lever position or physical blade stop.

## Non-negotiable feasibility checks for Claude Code, before calling this a working redesign

1. Model *measured* left/right 77603 lever pivot loci and clamp-point trajectory through **OUT-PARK ↔ IN-neutral**, plus the full fore/aft drive sweep, at both lateral locks released and (in the intended drive pose) clamped.
2. With both rod ends **continuously attached**, verify that each fixed-length rod can follow the measured three-dimensional endpoint path. **Two free pivots cannot fix an incompatible endpoint separation.** If measured endpoint spacing changes too much, stop and document the exact extra translation required; only then propose a *minimal positively lockable axial sliding feature* or different clamp-point/servo location. Never leave an unlocked telescopic rod in powered DRIVE.
3. Sweep for interference with servos, moving crossplates, clamp bodies, seat/MyRIDE movement, console, chassis, PARK safety switches, throttle attachment, and operator. No lateral or axial preloading through full travel.
4. Confirm the **lateral-axis locks do not arrest fore/aft steering** at neutral, forward or reverse, and the rods cannot unintentionally force levers out of PARK or prevent OEM PARK from engaging.
5. Evaluate steering-force/buckling, fastener retention, vibration, clamp loosening, pin bearing wear, weather and print creep. A freely adjustable friction clamp is **not automatically a certified lateral lock**; require a secondary retention/locking approach if bench results show slip. The factory PARK brake remains authoritative, not the printed joint.
6. Hand-cycle on a disabled mower first, then use isolated low-energy bench actuation. Actual powered-drive integration is **blocked** pending measurements, load validation, failsafe tests, independent PTO-off circuit verification and OEM safety-function inspection.

## Claude Code delivery (next session; do not fabricate today)

- Extend existing editable OpenSCAD rod-end/adapter modules and the assembled left/right linkages, **not** unrelated steering geometry.
- Provide only ONE preferred minimalist joint design, mirrored/adjusted for all four positions; avoid five concepts or a wholesale steering rebuild.
- Produce dimensioned assembly views showing **two independent axes and clamp** in unlocked PARK, inward neutral + lateral locked, forward, reverse; measured parametric ranges; separate STL parts, print/assembly notes and locked/unlocked collision checks.
- Show one side in a motion diagram and a lock/unlock sequence; list the minimal extra part/fastener count, known inventory reuse and all measurements missing.
- Mark joints and entire PARK motion as **UNVERIFIED UNTIL BENCH-TESTED**.

## Known conflict / higher-priority project decision

This is a *later* owner correction to older instructions saying there is no new PARK-swing hardware. **There is still no powered PARK actuator:** the operator moves bars manually, and only the two linkage ends per rod gain articulation and lockability. The existing Toro 77502 rod dimensions must **not** be copied as measured Toro 77603 geometry.
