# LawnBerry Pi — Field Integration Observations and Recommendations

**Updated:** 2026-10-24  
**Status:** Living engineering review; reported hardware and symptoms must be verified on-site.  
**Scope:** Full-size Toro zero-turn conversion, Pi 5 rover, Orange Pi 4 Pro HaLow/RTK base station, and autonomous-operation safety.

## Update protocol
For each future review, add dated entries recording: (1) operator-reported observations, (2) repository-verified facts, (3) hypotheses, (4) diagnostic evidence, (5) proposed solution, (6) tests and acceptance criteria, (7) remaining questions. Never treat an untested recommendation as deployed or validated. Updates occur during subsequent authorized development sessions or commits, not automatically in the background.

## 2026-10-24 — Current physical installation (operator reported)
- Mower: full-size, adapted zero-turn; Raspberry Pi 5 (8 GB, 512 GB storage), Ubuntu 24.04, Hailo-8L NPU, Heltec Wi-Fi HaLow V2 HAT, Quectel LC29H-DA RTK rover HAT.
- RTK rover's antenna cable was chewed through; replace **antenna/cable**, not receiver, if connector, bias voltage, active/passive type, and L1/L5 band coverage match. Exact antenna SKU/connector not independently verified.
- Ground station: Orange Pi 4 Pro (4 GB), Ubuntu 24.04, already operating the HaLow access point; spare Pi Zero W available but not required for initial RTK base plan.
- Intended new RTK base: Quectel LC29H-BS plus properly mounted outdoor multiband GNSS antenna; connect to Orange Pi by supported USB-UART bridge or verified voltage-level UART, not an unverified direct HAT stack. Verify compatible RTCM output and LC29H-DA firmware/input support before purchasing.
- Proposed correction flow: fixed GNSS base -> RTCM -> Orange Pi local correction relay/server -> existing HaLow network -> Raspberry Pi 5 -> GNSS rover corrections input -> LawnBerry positioning. Baseline survey and persistent coordinates are required; receiver update-rate and RTK FIX behavior need field validation.
- Property: approximately 8–9 acres, roughly rectangular, HaLow base station at midpoint of one long boundary; mostly line of sight, a few trees; rover antenna planned above camera tower. Exact side lengths not measured. ~600–700 ft maximum link distance previously discussed is an *illustrative estimate* dependent on property aspect ratio; measure actual max distance.
- Antennas: operator believes HaLow antennas are approximately 3 dBi. Verify model, gain, 902–928 MHz match, connector, cable loss, polarization, and regulatory approval before replacement. Begin with existing antennas and optimize mounting/line of sight. Consider higher gain only if site measurements justify it.
- HaLow channel plan: start with 8 MHz if enabled and permitted by actual module/firmware, compare 4 MHz in shadowed locations. Log RSSI/SNR, MCS, retries, goodput, latency, and packet loss at far corners and behind trees. Prefer reliable RTK/telemetry to maximum nominal PHY throughput.

## Significant HaLow fault — confirmed operator observation vs diagnostic hypotheses
**Reported reproduction:** On Raspberry Pi 5, 20 dBm transmit-power configuration operates; when configured to 28 dBm the *HaLow link alone drops* but the Pi remains running. A software reboot does **not** restore the link; removing and restoring hardware power **does**.

**Not established:** Root cause. Do not characterize antenna size as cause, or conclude firmware/power amplifier failure without logs and measurements.

**Hypotheses (priority for investigation):**
1. Firmware, driver, or radio chip enters a latched fault that persists across warm host reboot.
2. Peripheral power/reset rails are not fully cycled during a Pi reboot.
3. Supply droop/current surge or RF power-amplifier protection at the higher requested setting.
4. Regulatory, driver, supported power-vs-MCS, or calibration mismatch.
5. Connector damage, antenna mismatch/VSWR or thermal effects.

**Diagnostic plan (stationary bench only; do not attempt at speed or with blades active):**
- Leave transmit power at confirmed stable **20 dBm** for all mowing/network tests. Avoid repeated 28 dBm trials until instrumentation and recovery are available.
- Record exact Heltec HAT part number, radio module, firmware, kernel, out-of-tree HaLow driver revision, interface name, and power source/current budget. Confirm whether HAT is SDIO, SPI, or USB on this assembly.
- Preserve logs before/after failure: `sudo journalctl -k -b --no-pager`; `dmesg -T`; `ip -d link`; `iw dev`; `iw phy`; `nmcli dev status`. Identify interface-reset/firmware/SDIO errors and whether the network device remains enumerated.
- Compare 20 dBm and manufacturer-supported intermediate settings **only in a controlled radio bench test** after verifying legal power and certified antenna limits; instrument voltage at the module during TX and inspect temperatures.
- Determine whether the board exposes a documented hardware reset or switched power path, and whether a controlled reset recovers the radio. Do not assign an arbitrary GPIO to reset/power without the HAT schematic.
- Document the repeatable failure mechanism and reversible fix only after physical test. If no verified recovery, require a manual power-cycle maintenance procedure and conservative power setting.

## Repository-verified integration details (as read on 2026-10-24)
- `config/tractor.yaml` defines PCA9685 I2C bus 1 at **0x41**, 50 Hz; output channels **0 left drive lever**, **1 throttle**, **2 right drive lever**; starter relay **BCM GPIO 5**, blade PTO relay **BCM GPIO 26**. These are **logical/software allocations**, not proof of installed wiring.
- `docs/tractor-platform.md` indicates fore-aft lever actuation cannot engage the actual outboard PARK brake position. It also states current autonomous starting is blocked by OEM PARK interlock and lever self-centering is **unverified**.
- PCA9685 may hold its last servo PWM if I2C/host fails. Independently engineered safe-state mechanics and hardware watchdog are therefore **required** before unsupervised tests.
- OEM seat/operator-presence interlock remains a design decision. Do not bypass protective interlocks as a shortcut; perform supervised testing with OEM protections intact until a properly reviewed, independent safety solution is implemented.

## Recommended work order / development gaps
1. **Physical inventory and pin verification:** exact HaLow HAT revision, interfaces, power pins, installed servo controllers, rover GNSS wiring, antenna connector details, 12 V servo rail, fuses, grounding, E-stop harness. Create pin-by-pin cable schedule with measured voltages and board references.
2. **Independent safety:** mechanical neutral-return test (engine off), positive braking strategy, maintained physical emergency stop, fail-safe PTO/engine controls, command timeout independent of Linux and radio, and sensor interlock tests. No autonomous operation until confirmed.
3. **HaLow stability:** capture evidence of high-power latch-up; keep 20 dBm default; test full-property packet loss and link recovery at 8 MHz and 4 MHz.
4. **RTK purchase/bring-up:** verify LC29H-BS board SKU and RTCM compatibility, replace rover antenna, mount permanent base antenna, establish survey coordinates, test RTK FIX under trees, define stale-RTCM/no-FIX stop response.
5. **Navigation integration:** calibrate steering/hydrostatic actuators, inspect wheel/IMU odometry, validate geofencing and obstacle stop independently of GNSS, separate safety controller from perception stack.
6. **Acceptance record:** each interface gets source pin, destination pin, cable, voltage, fuse, connector, driver, test command, measured result, and open issue.

## Acceptance criteria (not yet completed)
- Stationary HaLow coverage map includes far corners and tree obstructions with stable correction transport and no uncommanded peripheral lockups at the selected settings.
- After simulated loss of HaLow, RTK fix, Pi control service, I2C bus, and main power, safety hardware independently reaches a verified safe state.
- No claim of centimeter-level *mower trajectory* based solely on receiver precision; verify absolute base position, fix availability, vehicle dynamics, and independent boundary containment.

## Change log
- **2026-10-24:** Initial log assembled from user-reported field conditions and repository configuration; no hardware diagnostics or modifications executed.
