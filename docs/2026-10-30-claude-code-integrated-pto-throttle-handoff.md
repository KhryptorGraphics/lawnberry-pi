# Claude Code integrated handoff — Toro model 77603 throttle CAD and PTO control
Date: 2026-10-30
Target repository: https://github.com/KhryptorGraphics/lawnberry-pi
Read-requirements branch: `docs/2026-10-30-throttle-servo-linkage`

## Prompt to execute in Claude Code

You are the lead mechanical CAD, mechatronics, embedded safety and systems-integration engineer for Lawnberry Pi. First review the current repository state and these authoritative user-requirement documents, **in this order**:

1. `docs/2026-10-30-throttle-servo-printable-variants-addendum.md`
2. `docs/2026-10-30-throttle-right-steering-offset.md`
3. `docs/2026-10-30-throttle-servo-3d-revision.md`
4. `docs/2026-10-30-claude-code-throttle-cad-revised-prompt.md`
5. `docs/2026-10-30-toro-77603-pto-switch-research.md`

Do not discard or supersede known user measurements without stating the conflict. Work on a feature branch created from the design-requirements branch; preserve existing code, documentation and CAD.

### Part A: Mechanical throttle modification — do not redesign steering
- Project mower: user-identified Toro TimeCutter Max Havoc MyRIDE 60-inch **model 77603**. Serial remains to be confirmed.
- Retain existing right steering mechanism, servo, linkage, and original geometries. Integrate an **additional MG995 throttle servo** onto/above the right-side mount rather than reworking the mower drivetrain.
- Follow the user-provided MG995 body/mount measurements in the existing docs, rather than guessing generic specifications.
- The throttle is right of the right steering bar, with offset ranges given in the existing spec; do not translate those offsets into an assumed lever stroke.
- Implement five fully distinct editable/parametric printable linkage variants specified in the addendum: A direct X-horn arm; B printed X-horn adapter; C continuously adjustable-radius horn; D positively retained printed ball joint; E captured printed double-shear clevis with replaceable insert.
- Every **new** linkage element, adjustment mechanism and lock must be printable, without purchased new metal joint hardware. Preserve the already-owned servo and horn.
- Length adjustment must be continuously sliding and micro-fine, with a completely printed positive lock; no metal clamp bolt. Provide retention against accidental separation if the clamp slips.
- Throttle-end fitting is an inverted-U/open-bottom shape. The initial **inside** dimensions are 12.70 mm front-to-back by 31.75 mm left-to-right; use tunable parameters and a test-fit gauge.
- The end fitting must align with the throttle lever's rightward tilt, **angle unknown until measured**. Include editable right-lean, yaw and pitch where appropriate.
- Design for outdoor engine-bay heat, UV, moisture, vibration, creep and wear; evaluate appropriate filaments against actual printer capability. Do not claim an all-printed assembly is safe for long-term live use until verified by physical tests.
- Export genuine CAD sources, STEP and STL files, dimensioned drawings, 5 assembled and exploded renders, bill of printed parts, tests and unresolved-measurements checklist. Do not invent models or say deliverables exist unless generated and inspected.

### Part B: Electric blade/PTO control — simplest verified feasibility path
- Review the model 77603 research document and original official Toro manual: https://www.toro.com/getpub/273204 ; publication 3473-450C; relevant electrical diagram Chapter 8 page 8-1 (confirm before citation).
- Distinguish two requirements: (1) SBC can **disable** the clutch when factory PTO has been selected, versus (2) SBC can independently request ON and OFF. A lone normally-open series interruption relay meets only (1), not necessarily (2).
- Prefer the smallest reliable electrical-control interface **that leaves factory interlocks and engine shutdown intact**. Do not make a servo press the PTO button merely for speed when electrical removal of power may be feasible.
- Do not infer wire colors, pin numbers, contact states, PTO switch type, coil voltage or current. Obtain serial-specific diagrams and physical connector measurements first. Any interface is provisional until those checks are complete.
- Blade de-energization does not instantly stop blade rotation. Define measurable blade stop-time tests.
- Pi/SBC power loss or watchdog failure must remove blade authorization. Keep a separate hardware emergency-stop/engine-kill system, independent of software and sensor inference.
- Avoid constructing a bypass around OEM safety interlocks. Draft only a conditional schematic with clearly named unknown measurements; no claims of final installation-ready wiring before verification.
- Model controller states (OFF, ARMING, PERMITTED, FAULT, E-STOP) and bench tests for failed relay, brownout, reboot, broken wire, false sensor trigger and shutdown.
- For the user's fast object-detection blade-off objective, separate sensor/inference latency, GPIO command latency, relay/clutch release latency and physical blade spin-down; report validated or expected values separately, and mark unknowns.

### Part C: Platform variants and project continuity
- Maintain interchangeability requirements for Raspberry Pi 5, Orange Pi 4 Pro and Libre Alta where the current architecture supports them. Use a small hardware abstraction rather than assuming Pi-only GPIO pin numbering for all boards.
- Preserve AI/inference, RTK GPS, dual HaLow networking and NVIDIA Thor data handoff design notes already in the repository. This handoff does not authorize redesign of those subsystems.
- Use existing application safety/telemetry APIs as applicable. Safety-critical disable must not rely on a network round-trip or a cloud server.

### Part D: Photographs and provenance
- User asked to archive **all photos uploaded on 2026-10-30** so Claude Code can reference them.
- The conversation session confirmed only **one** accessible control-panel image; **other four images were not confirmed accessible**. Do not claim all five are uploaded.
- Inspect `docs/media/2026-10-30/` and a media manifest if present. If photos are missing, list precisely which originals are needed, without inventing imagery or labels.
- When originals become available, commit the binary images as images (not transcript text), retain useful filenames and a Markdown manifest with subject, date, repository-relative link, and which design measurement they inform.
- Distinguish observed visual features from any connector/pinout assumptions.

### Deliverables and reporting
1. Working mechanical CAD source files and exports for all five variants, or a transparent list of what could not be completed.
2. Updated PTO research and conditional schematic/finite-state-machine docs.
3. An explicit unknowns/measurements and validation/test checklist.
4. GitHub commits/PR links with changed files and status; never claim uploads occurred until their commit SHAs are confirmed.
5. A final handoff: completed / unverified / blocked; specify safe next step and any photographs needed.

Start with a repository file inventory and compliance matrix against these requirements, then implement. Prefer non-destructive commits and reviewable changes.