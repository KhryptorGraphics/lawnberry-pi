# Toro 77603 — Integrated owner decisions, hardware allocation, power and control
Date: 2026-10-30
Status: Consolidated **requirements and purchase-status notes**, not an installed or validated electrical design.
Target: Toro TimeCutter MAX Havoc MyRIDE 60-inch **model 77603**; serial-specific wiring remains unverified.

## Architecture — two local controllers, not one central PWM bus
- **Raspberry Pi 5:** autonomous steering/navigation, two 150 kg-class steering servos, AI/vision. A **PCA9685** board drives servo **control signals** locally; steering servos receive high-current power from their separately purchased steering converters, not the PCA9685.
- **Libre Alta:** independent engine and blade control (virtual ignition RUN/OFF, momentary starter, blade/PTO), plus **MG995 throttle servo** from a second local PCA9685 board. The Alta's GPIO/relay controller interfaces with separate appropriately rated switching components for ignition and PTO. A PCA9685 cannot directly switch starter, ignition or electric clutch power.
- **Two PCA9685 16-channel boards (32 PWM channels total)** are more than enough for the **three planned servos** (two steering, one MG995). With one board per computer, each host has up to 16 local PWM channels, not a shared 32-channel inter-host bank; the two boards do not remove the need for relay drivers and independent safety hardware. Both controllers use I2C and 3.3 V-compatible logic; address and electrical compatibility on each host require bench checks. The repository documents an INA3221 at 0x40 on the Pi, so set the Pi's PCA9685 to a nonconflicting address such as 0x41.
- Both Pi 5 and Alta have 5 V on header physical pins 2/4, but their safe **spare current** depends on each installed USB-C power converter and other loads. PWM boards' logic can use compatible 3.3 V rail, **but servo power must be separate**. Do not assume the host's 5 V pins can supply relay coils without supply-budget measurement.
- HaLow: Each SBC owns a Wi-Fi HaLow card. Exchange **small state/health messages**, timestamps, authorization and heartbeat, not continuous servo PWM. A modest communication rate suffices; **bandwidth is not the safety problem**. Stale or missed heartbeat triggers SAFE_STOP. A web dashboard can aggregate telemetry through network communication; local safety functions remain independent of web UI and radio connectivity.

## Owner-confirmed purchased items (do not rebuy)
- **Two 150 kg-class steering servos**. Exact brand/SKU, rated voltage/current and physical mounts require inspection. The 125 kg reference was corrected as inaccurate.
- **Two power converters for the steering bars**, assumed individually dedicated pending label verification.
- **Two battery-to-USB-C converters**, one dedicated to Pi 5, one to Alta; confirm Pi 5 requires robust approximately 5.1 V/5 A support and each converter's actual supported 5 V current.
- **One automotive relay harness** (does not necessarily include relay bodies).
- **One two-piece pack of PCA9685 16-channel, 12-bit IIC PWM servo-controller modules**. Product identified by owner as "2Pcs PCA9685 16 Channel 12-bit PWM Servo Motor Controller Module IIC Interface Module, Compatible with Raspberry Pi, Compatible with Arduino, Robotics DIY." These are **not relay boards**.
- **HiLetgo 2pcs 5V 30A one-channel opto-isolated relay module pack**, as identified by owner (reference catalog: https://www.amazon.com/dp/B0CHFJSNP6). Two independent one-channel boards; these are switching interfaces, **not DC-DC power converters**. Output/inductive ratings, coil draw and whether 3.3 V GPIO reliably triggers must be bench-verified.
- Separate MG995 throttle servo and horn were previously reported owned in the throttle handoff.

**Historical BOM caution:** `hardware.md` includes earlier model 77502 design, old power assumptions, and "Need to buy" items that overlap purchased hardware. Treat it as sourcing reference, NOT an order sheet for model 77603.

## Power-distribution decisions and unresolved supply gaps
- Mower power source is a **12 V nominal starting battery** with real charging and cranking transients. Each new branch requires appropriate fusing, grounding, load protection and vibration/moisture handling.
- Pi 5 and Libre Alta are each supplied by their **existing dedicated battery-to-USB-C converter**. Verify USB-C output current at the required voltage.
- Steering servos are powered **from the two owned steering converters**, *only after checking servos' input range vs converter voltage*. Never apply legacy 24 V instructions to nominal 12 V servos without explicit data.
- PCA9685 logic VCC may be 3.3 V from its respective host; SDA/SCL are signal connections. Its PWM outputs do **not** deliver motor current. Do not route high-current steering supply through PCA9685 V+ traces or servo-header bus when not rated.
- MG995 throttle servo generally needs a regulated supply around 5–6 V, sized for startup/stall current; it **may share an existing converter only if voltage, capacity, common-ground and surge budget are proven compatible**. A 12 V steering rail cannot directly supply it.
- 5 V HiLetgo relay modules need a **regulated 5 V supply**, not raw 12 V from the mower and not 3.3 V GPIO power. Reuse an SBC 5 V rail only with demonstrated spare current, startup behavior and safe isolation; otherwise add a small protected 12-to-5 V converter. Exact supply is **undecided**.
- Switching contacts do not create supply voltage: they must interface with the existing validated factory circuits. Preserve factory safety interlocks.

## Virtual operator controls — requirements, not verified wiring
- GUI contains **virtual ignition OFF/RUN** toggle. RUN persists as a software request until switched OFF or safety revoked; OFF requests reliable OEM-compatible engine shutdown.
- GUI has **momentary press-and-hold START**: start request must stop immediately upon finger/mouse release, pointer cancel, navigation loss, communication loss, emergency stop, or engine-running detection. No software-latched starter command.
- User explicitly requested 5 seconds between attempts; **manufacturer manual specifies max 5 seconds crank and 15 seconds rest after unsuccessful start**. Record the 5-second preference but enforce the manufacturer protection unless validated, approved redesign establishes safe alternative; display actual enforced timer.
- Start request follows the **factory interlock path**, including PTO OFF and bars in PARK. Existing fore/aft-only servo linkage does not place bars in PARK, so no unattended autonomous start is available with that arrangement as-is. Do not bypass seat/PARK/PTO interlocks.
- GUI has **BLADE ON/OFF** but the factory PTO is the **rectangular rocker (Toro 139-6368 in applicable serial range)**; the round knob above ignition is **CHOKE**, not PTO. A single relay in series with clutch feed provides only permission-to-engage if the OEM switch is already ON; it cannot by itself independently command ON/OFF.
- OEM components identified provisionally from Toro serial >=418961000 catalog: ignition switch 137-4100, starter solenoid 117-1197, auxiliary relay 1-643275 (role unknown), PTO rocker 139-6368. No specific physical pins or full engine starter-motor suffix confirmed. Do not bridge high-current solenoid studs.
- Minimal-cost design should reuse OEM solenoid and clutch, use owned electronics first, and buy only electrically verified missing switching and power items.

## REQUIRED controller-failure rule (priority over routine stop)
If **either Pi 5 OR Alta** loses power, crashes/freezes, disconnects from HaLow, stops transmitting valid heartbeat, or otherwise loses safe control, **the entire mower must enter a latched SAFE_STOP**, including:
1. Immediate PTO power authorization off and actual blade spin-down measured.
2. Starter permission removed, and **engine shut off** through an independently operating, OEM-compatible physical stop path.
3. Drive motion halted with **physically verified neutral/braking** of both lap bars (simply cutting power to a geared servo may leave its last position; the lever self-centering and PARK mechanisms have not been proven).
4. All authorization revoked; deliberate operator rearm required. No automatic restart after connectivity or power returns.

The safety shutdown **must function even if both SBCs are dead or the wireless link fails**. A network heartbeat alone is not an independent hardware stop. Design hardware watchdog/independent safeguard first, then test it by killing power to each SBC alone, both at once, inducing stuck PWM and split/failed HaLow, including actual mechanical stop measurements. See `2026-10-30-dual-controller-safe-stop-requirement.md` and `tractor-acceptance-criteria.md` section 12.

## Throttle mechanical scope preserved
MG995 on right-side steering structure operates throttle via **five printable linkage variants** from the existing addendum; steering linkage is not to be redesigned. A printable design and fitting remain prototypes until measured and tested under heat/vibration. Keep original geometry and previously recorded measurements.

## Purchase-ready status / verification gates
**No final order-ready shopping list yet.** Confirm: exact servo/steering converter nameplates, existing relay harness pin/socket/relay details, 5 V relay trigger behavior and power draw, MG995 supply budget, mower serial number, OEM key/PTO rear connector contacts and original wiring, safe physical lever return and engine kill. The original console JPEG and other images are not verified uploaded to GitHub; see the 2026-10-30 photo manifest. No unverified pin-to-pin instructions, interlock bypass or claims of autonomous safety.

## Source documentation already on this branch
- `2026-10-30-user-confirmed-owned-hardware.md`
- `2026-10-30-toro-77603-starter-ignition-technical-research.md`
- `2026-10-30-toro-77603-pto-switch-research.md`
- `2026-10-30-claude-code-integrated-pto-throttle-handoff.md`
- `2026-10-30-dual-controller-safe-stop-requirement.md`
- `tractor-acceptance-criteria.md`
- Official Toro electrical manual: https://www.toro.com/getpub/273204
