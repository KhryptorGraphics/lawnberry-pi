# Claude Code — revised executable prompt for Lawnberry Pi throttle CAD
Updated October 30, 2026. This prompt is to be given directly to Claude Code operating in the mower repository. The goal is **actual editable CAD and printable STL deliverables**, not documentation alone.

```text
ROLE: You are a senior mechanical engineer, robotics/mechatronics engineer and parametric CAD implementation specialist.

REPOSITORY: https://github.com/KhryptorGraphics/lawnberry-pi
DESIGN-REQUIREMENTS BRANCH: docs/2026-10-30-throttle-servo-linkage
REQUIRED DOCUMENTS:
- docs/2026-10-30-throttle-servo-printable-variants-addendum.md
- docs/2026-10-30-throttle-right-steering-offset.md

MISSION
Retrieve the two current documents from the specified branch and inspect existing native mower CAD. Create an implementation branch based on that documentation branch. Preserve all existing steering servos, steering bar/pushrod geometry and other mower components. This is a throttle-servo mounting and linkage modification, NOT a full mechanical redesign. Deliver five independently viewable, editable, printable throttle linkage variants integrated into the existing model. Generate CAD/STEP/STL, drawings, exploded views, renders, instructions, bill of printed components, checks and an engineering comparison. Never claim completed CAD unless actual geometry has been generated and files verified.

1. MEASURED MG995 SERVO ENVELOPE (user supplied, inches; convert to mm internally):
- Body upright with shaft vertical and toward far front/back end: height 1.50 in, left/right width 0.50 in, front/back length 1.50 in.
- Output shaft protrudes approximately 0.50 in above housing. Provide further clearance for attached horn and pushrod.
- Front/back mounting-hole center spacing 2.00 in; hole centerline 1.125 in above body bottom.
- Existing X-shaped four-arm servo horn remains. Shaft XY location, mounting hole diameters, lug thickness, exact horn radius and fastening details are unknown: parameterize and verify before printing.
- Vertically extend and structurally reinforce existing right steering servo bracket to carry an INDEPENDENT throttle MG995, while retaining original right steering servo and linkage, servicing access, clearance and steering function.

2. OFFSET THROTTLE CONTROL:
- Throttle is to the RIGHT of the right-hand steering bar.
- Fully back / idle position: 2.5–3.5 inches to the right of that bar.
- Fully forward / full speed: 3.5–4.5 inches to the right.
- These are positional ranges, not proof of exactly 1 inch lateral travel. Parameterize reference datums and kinematics. Check collisions and changing linkage angles through complete throttle AND steering travel, with adequate articulation and minimum sideways servo shaft loads.
- Earlier 40-inch and 44-inch throttle measurements, and proposed bands 38–41 and 43–46 inches, lack confirmed reference points and do NOT prove a four-inch pushrod stroke; do not design to an invented stroke.

3. ENTIRELY 3D-PRINTED NEW LINKAGE — NO PURCHASED JOINTS OR LOCKING HARDWARE:
- Retain the servo, existing X-horn and mower throttle lever; print all NEW bracket, horn adapters, pushrod segments, pivot/clevis/ball components, bushing inserts, end stops, retainers, telescoping fine adjusters and locks.
- Do not require a purchased ball link, metal clevis, metal pushrod, bolt, threaded insert, screw, nut, pin, bearing, or metal fitting to operate the new linkage. Existing stock servo attachment is retained as needed.
- Use captured joints and replaceable wear parts, reinforce weak regions, evaluate print layer orientation, heat, UV, water, vibration, fatigue and creep. All-printed is a design requirement, NOT proof that it will be safe or durable.

4. NEW MANDATORY LOCK RULE — NO BOLT:
- Pushrod length MUST be CONTINUOUSLY SLIDING and MICRO-FINE adjustable. Never use indexed holes, notched steps, or discrete positions to choose length.
- The pushrod must lock at ANY selected length with a mechanism whose entire new hardware is 3D PRINTED. Do not use a metal bolt, nut or screw to squeeze a clamp.
- Develop and compare at least three printable continuous-adjustment/locking concepts: captive sliding wedge/cam with second retainer; printed split collet tightened by a large printable threaded ring; printed lead screw/sleeve with anti-backdrive retainer, or equivalent.
- Include visible lock engagement, captive mechanical end-stops so parts cannot separate on clamp slip, repeatable fine adjustment, minimum safe overlap, stiffness and access for readjustment. Do not rely on friction alone to prevent catastrophic disconnection. If none meets validation requirements, document the failure and do not release the design as safe.

5. INVERTED-U / UPSIDE-DOWN-BOX THROTTLE END:
- At the far throttle end, create a fully printable inverted-U/open-bottom box fitting engaging the actual mower throttle lever.
- Its required CLEAR INTERNAL SIDE-TO-SIDE WIDTH and CLEAR INTERNAL FRONT-TO-BACK DEPTH are NOT MEASURED. The user specifically needs both numbers to know if it will fit. Do not invent them or claim it already fits.
- Expose independently editable parameters: actual throttle engagement-point outside left/right width, outside front/back thickness or depth, chosen clearance on each side, interior contact height, wall thickness and installation path.
- Interior width = measured exterior throttle width + 2 × left/right running clearance; interior depth = measured exterior throttle depth + 2 × front/back clearance.
- Provide a small printable fit-check sample/template before final print, and ask for just the two critical outside dimensions when they cannot be obtained from an existing scaled CAD or trustworthy measurements.
- Model the throttle-lever cap/engagement geometry and sweep to prevent binding, excessive side loading or open-throttle jamming. Do not confuse steering-bar offset with the throttle lever's size.

6. FIVE DISTINCT COMPLETE CAD OPTIONS; DO NOT SELECT ONE:
A. Direct connection to one existing X-horn arm with printable captive joint.
B. Custom printable X-horn reinforcement adapter plus captured joint.
C. Continuously adjustable servo-horn effective radius, with printable continuous locking and no indexing holes.
D. Mechanically captured printable ball/socket with a positively retained, replaceable wear insert; reject if weak.
E. Double-shear printable clevis with captive printable pin and replaceable bushing.
All variants include the same servo mount, right offset, print-only micro-adjustable telescoping pushrod and dimensionally configurable throttle-end inverted U. Provide an individual complete assembly, exploded rendering, part STLs, parametric CAD and evaluation.

7. PRINTER AND ASSEMBLY:
- Build volume: 420 × 420 × 500 mm (X × Y × Z). Reserve usable margins for brim, adhesion, printer motion and orientation.
- Split any component that will not fit the usable oriented print envelope, including long pushrods or high brackets where necessary. Use keyed overlaps, alignment, positive mechanical capture and 3D-printable joint retainers; never rely on structural glue alone.
- Provide dimensions, build-plate plan, distinct labeled STL for each printed segment, part orientation, fit/collision checks and assembly order.
- Consider PETG, ASA and compatible engineering filaments; verify printer nozzle/bed/enclosure before prescribing a material. State recommendations as hypotheses until tested.

8. FAILURE AND SAFETY VALIDATION:
- Calculate range of motion, torque demands, loads, buckling, fatigue concerns, backlash, print-layer stresses, joint retention, lock slip, creep, temperature limits, clearance and service access to the extent actual geometry and load data permit.
- Retain independent emergency engine stop and independent return-to-idle/fail-safe throttle action. No single broken printed link/lock should hold throttle open. Require physical bench tests for loss of power, disengagement, heat, repetitive cycling, vibration and manual override before operating mower. Do not claim simulation replaces physical tests.
- Surface unresolved measurements, untested strength and unachieved safety requirements explicitly; parameterize missing details rather than fabricating values.

9. ACTUAL IMPLEMENTATION AND DELIVERY:
- Read both updated addenda and inspect existing source first. Create a dedicated implementation branch. Never overwrite original CAD or unrelated work.
- Use available FreeCAD, CadQuery, OpenSCAD or equivalent parametric CAD. Build FIVE real models and exports, not placeholder STLs or generic diagrams. Verify re-import, printable part size with margin, assemblies and range-of-motion checks.
- Deliver five native editable CAD assemblies, STEP exports where supported, separate individual STL files, the fit-check gauge for the inverted U, labeled previews and exploded diagrams, dimensioned drawings, printable-only BOM, assembly and printing guides, comparison matrix, validation/failure report and unresolved measurement list.
- Update repository documentation and commit CAD scripts/models/exports and reports. Only push changes if allowed by repository/project authorization. Report branch, exact commit SHA, paths, test outcomes, all five renders and remaining blockers.
- Do NOT pick a preferred configuration without user review. Do NOT substitute a proposal/document for actual geometry.
START IMPLEMENTATION.
```
