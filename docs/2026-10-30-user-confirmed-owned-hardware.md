# Toro 77603 — user-confirmed purchased hardware (2026-10-30)

**Status:** Directly confirmed by owner in voice conversation on 2026-10-30. This is the authoritative purchase-status correction for the Toro 77603 starter/PTO shopping review. Brand/SKU, voltage, current, dimensions and exact relay contact/channel count are **not confirmed** unless stated. Do not convert ownership statements into unverified vendor model assignments.

## Confirmed purchased / on hand
| Count | Item | Notes |
|---:|---|---|
| 2 | **125 kg-class steering-bar servo motors** | The owner explicitly says **125 kg**, not the 150 kg units in the older `hardware.md` list. Exact SKU and whether 125 kg is a torque marketing figure must be inspected; do not buy replacements. |
| 1 | **Controller/relay board** | Owner says "a single controller relay board." Board channel count, output types and electrical ratings not yet established. Do not assume it is the generic PCA9685 or a specific HiLetgo relay board. |
| 2 | **Power converters for the steering bars** | Already purchased; electrical ratings and whether one supplies each servo require reading device labels. Do not reorder steering converters. |
| 1 | **Automotive relay harness** | Already purchased. Quantity of relay sockets, installed relays, wiring and pin ratings unverified. Do not order duplicate harnesses without matching to the actual parts. |
| 2 | **Battery-to-USB-C power converters** | Already purchased: one to run the **Libre Alta**, one for **Raspberry Pi 5**. Do not reorder a separate Pi 5 buck converter merely because an older BOM mentions it. |

## Other hardware known from previous project discussion
- Raspberry Pi 5 and Libre Alta are owned.
- A separate MG995 throttle servo and horn were described as already owned in the 2026-10-30 throttle design handoff; this is distinct from the two steering servos.
- OEM Toro starter solenoid and PTO clutch are to be reused, not replaced as part of this upgrade.

## Outstanding assessment for new starter / blade ON-OFF shopping list
1. Identify **existing relay board** from its label/photos, especially channel count, coil voltage, contact arrangement and DC/inductive rating.
2. Check whether the **automotive harness includes relay bodies** and how many sockets/contacts. A purchased harness does not establish that relays are included.
3. Verify the Toro **model 77603**, serial-specific switch-contact topology, ignition kill/run/start and PTO interlock circuits before prescribing any new relay. See `2026-10-30-toro-77603-starter-ignition-technical-research.md` and `2026-10-30-toro-77603-pto-switch-research.md`.
4. Retain factory interlocks, physical E-stop, fused circuits and independent loss-of-control protection. Never emulate safety contacts or switch the solenoid's high-current starter studs.
5. After comparison, produce an **incremental** shopping list only; do not rebuy the steering servos, steering converters, battery-to-USB-C converters, relay-controller board or automotive harness without explicitly showing why an additional part is needed.

**Legacy-document warning:** `hardware.md` on the older platform lists 150 kg servos and several components under "Need to buy." That list is not a current receipt and uses an older Toro 77502 build. Treat this file's direct owner confirmation as overriding those earlier purchase-status assumptions, without treating the old SKU choices as model-matched to the user's parts.
