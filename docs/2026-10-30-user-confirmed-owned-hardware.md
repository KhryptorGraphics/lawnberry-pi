# Toro 77603 — user-confirmed purchased hardware (2026-10-30)

**Status:** Directly confirmed by owner in voice conversation on 2026-10-30. This is the authoritative purchase-status correction for the Toro 77603 starter/PTO shopping review. Brand/SKU, voltage, current, dimensions and exact relay contact/channel count are **not confirmed** unless stated. Do not convert ownership statements into unverified vendor model assignments.

## Confirmed purchased / on hand
| Count | Item | Notes |
|---:|---|---|
| 2 | **150 kg-class steering-bar servo motors** | **Owner's latest explicit correction: 150 kg, NOT 125 kg**. Exact SKU and whether the stated rating is a marketing torque figure must be inspected; do not buy replacements. The older `hardware.md` lists a 150 kg servo class but does not establish the purchased vendor/SKU. |
| 2 | **PCA9685 16-channel PWM servo-control modules** | Owner provided the two-piece PCA9685 product title; 16 PWM channels on each board, suitable for controlling steering and throttle servo signals. **Not relay modules.** |
| 2 | **HiLetgo 5 V 30 A single-channel optocoupler relay modules** | Owner identified the two-piece HiLetgo relay pack. These switch contacts and need separately sourced, regulated 5 V power; verify contact and coil ratings for Toro circuits. |
| 2 | **Power converters for the steering bars** | Already purchased; electrical ratings and whether one supplies each servo require reading device labels. Do not reorder steering converters. |
| 1 | **Automotive relay harness** | Already purchased. Quantity of relay sockets, installed relays, wiring and pin ratings unverified. Do not order duplicate harnesses without matching to the actual parts. |
| 2 | **Battery-to-USB-C power converters** | Already purchased: one to run the **Libre Alta**, one for **Raspberry Pi 5**. Do not reorder a separate Pi 5 buck converter merely because an older BOM mentions it. |

## Other hardware known from previous project discussion
- Raspberry Pi 5 and Libre Alta are owned.
- A separate MG995 throttle servo and horn were described as already owned in the 2026-10-30 throttle design handoff; this is distinct from the two steering servos.
- OEM Toro starter solenoid and PTO clutch are to be reused, not replaced as part of this upgrade.

## Outstanding assessment for new starter / blade ON-OFF shopping list
1. Verify electrical ratings of the purchased HiLetgo relay modules and confirm whether additional switching contacts or independent watchdog outputs are required. The two PCA9685 boards cannot substitute for engine/PTO relays.
2. Check whether the **automotive harness includes relay bodies** and how many sockets/contacts. A purchased harness does not establish that relays are included.
3. Verify the Toro **model 77603**, serial-specific switch-contact topology, ignition kill/run/start and PTO interlock circuits before prescribing any new relay. See `2026-10-30-toro-77603-starter-ignition-technical-research.md` and `2026-10-30-toro-77603-pto-switch-research.md`.
4. Retain factory interlocks, physical E-stop, fused circuits and independent loss-of-control protection. Never emulate safety contacts or switch the solenoid's high-current starter studs.
5. After comparison, produce an **incremental** shopping list only; do not rebuy the steering servos, steering converters, battery-to-USB-C converters, PCA9685 boards, HiLetgo relay modules or automotive harness without explicitly showing why an additional part is needed.

**Legacy-document warning:** `hardware.md` lists 150 kg-class servos, consistent with the owner's corrected rating, but that document is for an earlier Toro 77502 build and is **not a purchase receipt or proof of exact part number**. Treat the owner's 150 kg purchase confirmation as authoritative without assuming the older listed brand/SKU is an exact match.


## 2026-10-30 architecture clarification
See [integrated hardware, control and safe-stop decisions](2026-10-30-integrated-hardware-and-control-decisions.md). A PCA9685 supplies PWM signals, **not 5 V servo motor power, ignition switching or PTO switching**. One module per SBC is sufficient for the three existing servo channels; the separately identified HiLetgo modules are relay-control hardware.
