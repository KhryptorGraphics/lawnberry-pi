# Throttle servo linkage — additive 3D-print revision (2026-10-30)

## Scope
Preserve the existing steering-servo mount, steering pushrod and attachment geometry. Extend the existing right-side steering-servo mounting structure vertically to provide a second, separately serviceable servo mounting position for the throttle. Do not redesign the entire mechanical assembly.

## Field dimensions (verify before fabrication)
- Throttle lever at LOW: measured servo-to-throttle separation **3 ft 4 in = 40 in = 1016 mm**.
- Throttle lever fully FORWARD: separation **3 ft 8 in = 44 in = 1117.6 mm**.
- Requested adjustment at one end: **3 ft 2 in–3 ft 5 in = 38–41 in (965.2–1041.4 mm)**.
- Requested adjustment at other end: **3 ft 7 in–3 ft 10 in = 43–46 in (1092.2–1168.4 mm)**.
- These are endpoint-distance targets, NOT necessarily a single telescoping rod length or 4 inches of actuator stroke. Confirm measurement datum, mounting position, lever arc, linkage path, and interpretation of the two adjustable ranges before final CAD dimensions.

## Coding-agent CAD instructions
1. Locate existing parametric CAD/STL/STEP source for the right-side steering servo bracket and pushrod; preserve source and interfaces, and make a derivative revision rather than replacing the assembly.
2. Add a vertical bracket extension with an independent mounting pattern for the second servo. Parameterize vertical offset, hole spacing, servo body clearance, wall thickness, ribs, fasteners and adjustment slots. Do not invent dimensions for the actual servo; measure/model the installed unit.
3. Model a separate throttle pushrod, with mechanically secure, lockable length adjustment and articulation at both ends as needed. Account for full throttle lever travel and collision-free operation.
4. At the throttle end, model an **inverted U-shaped / upside-down box capture** that fits over the top of the throttle lever. Parameterize internal width, depth, lever thickness, engagement clearance, retention and non-marring contact surfaces. Ensure the lever cannot unintentionally escape or bind; retain manual override.
5. Include low/high throttle poses and swept-volume checks in CAD; output an assembly, printable component models, BOM, dimensioned drawings and an adjustment/installation guide. Mark unmeasured dimensions TBD rather than guessing.
6. Verify linkage kinematics: 40-to-44-inch separation change does not prove a 4-inch straight-line servo stroke. Calculate servo horn radius and rotation, lever force, mechanical advantage, torque margins, end stops and the servo's usable travel before approving the design.
7. Preserve electronics/control mapping documented in the repo: PCA9685 address 0x41; channel 1 throttle (channels 0/2 steering). Do not change hardware configuration without an explicit review.
8. Use a separate working branch/PR; simulation/dry-run only. Do not drive physical servos or deploy to mower without human inspection and approval. Fail safe to a verified low-throttle condition, provide manual override and independent emergency stop. A servo is not a safety-rated stop.

## Acceptance criteria
- Existing steering mount and linkage unchanged except the additive vertical extension.
- Second servo securely accommodated without obstructing steering.
- Throttle pushrod and inverted U capture modeled as separate adjustable components.
- Both measured positions and both requested adjustment intervals represented and checked.
- No interference through full movement; safe manual operation maintained.
- Source CAD, print files, drawings, BOM and validation notes included in PR; all unknown measurements called out explicitly.

## Agent handoff
Read `docs/2026-10-24-physical-integration-plan.md`, `docs/2026-10-24-development-roadmap.md`, `docs/2026-10-24-compatibility-gap-register.md`, and `docs/OMP-THOR-HANDOFF-2026-10-24.md` on the audit branch; inspect current CAD and `config/tractor.yaml` before changing anything. Implement this as an additive, reviewable CAD revision. Do not assume this document means models have already been generated.
