# House RTK Base Station — Implementation Specification
Status: hardware ordered; implementation pending arrival. Target: LawnBerry Pi mower.

## Hardware and topology
- Base: Raspberry Pi Zero 2 W (planned), Waveshare LC29H(BS) GNSS HAT (claimed SKU 25280; verify physical label and vendor documentation on arrival), stationary outdoor active L1/L5 GNSS antenna.
- Rover: Raspberry Pi 5 with LC29H(DA) GNSS HAT and matching L1/L5 antenna.
- Network: Pi Zero 2 W connects to home LAN via Wi-Fi; existing HaLow access point routes correction traffic to mower. Do not assume Pi Zero has a HaLow radio.
- Flow: satellites -> fixed antenna -> LC29H(BS) -> Pi Zero UART -> RTCM transport -> home LAN/HaLow -> Pi 5 -> LC29H(DA).
- Hardware connector, antenna voltage/bias, UART pin mapping, voltage levels, serial baud, RTCM message set, and vendor-specific survey-in commands MUST be verified from exact shipped hardware/manual before wiring/configuring. Never assume rover and base are interchangeable.

## Functional requirements
1. Headless Raspberry Pi OS Lite service managed by systemd, starts on boot and recovers from serial/network interruption.
2. Detect and configure the GNSS serial device without taking over the OS serial console; avoid conflicting GPIO services.
3. Obtain base position via supported survey-in, or enter externally surveyed known antenna coordinates. Record datum/reference frame, coordinate epoch, antenna reference point and measured antenna height; do not claim survey-in alone guarantees centimeter absolute accuracy.
4. Stream supported RTCM 3 correction messages from base to one or more rover clients using a documented, configurable TCP or NTRIP interface. Prefer established maintained software; avoid bespoke RTCM parsing when not needed.
5. Pi 5 client reconnects automatically, validates freshness, forwards raw corrections to LC29H(DA), and reports rover RTK status (no fix/float/fixed), age of corrections, satellites, and link health.
6. Configuration in environment/config file; no hard-coded LAN IPs, passwords, coordinates or device paths. Restrict caster access to LAN/VPN; authenticated NTRIP if exposed; no public default exposure.
7. Logging and health endpoints: serial state, correction throughput, last RTCM timestamp, connection count, base survey state, GNSS status, restart count, and clock timestamps. Include a simple dashboard integration hook.
8. Network disruptions must never cause mower to continue autonomous motion on stale positioning. Signal degraded/unavailable RTK to navigation; safety supervisor must enforce a validated safe state, including controlled stop where appropriate. Do not treat RTK alone as collision avoidance.

## Antenna and site
Mount rigidly with clear sky, away from reflective metal and tree canopy; weatherproof connections and strain relief. Do not move after calibration. Document antenna location, cable length/loss, lightning/surge protection plan, and power supply. Use short low-loss RF cable or position receiver closer to antenna.

## Claude Code implementation tasks
- Inspect existing repository GNSS/NTRIP/RTK components before editing; reuse interfaces and avoid breaking current CORS NTRIP client.
- Add `docs/rtk-base-station.md` (this spec), `config/rtk-base.example.env`, a base station deployment/service, and rover integration adapter only after identifying existing code layout.
- Separate base station and rover configs. Provide an offline mock RTCM source and simulated disconnect tests before connecting hardware.
- Tests: serial reconnect, startup without hardware, RTCM framing/transport integrity, stale correction timeout, authentication, LAN-to-HaLow routing, rover FIX/FLOAT transitions, and graceful shutdown.
- Provide setup instructions for Pi Zero 2 W, antenna survey/known-point workflow, systemd installation, logs, network ports and firewall.
- Do not enable unattended mower movement or blade operation during RTK bring-up.

## Acceptance criteria
- Base reboots unattended and resumes valid RTCM transmission.
- Rover receives corrections over house LAN + HaLow and reports correction age and RTK status.
- Drop network or base serial: client reconnects, reports degraded state immediately, and navigation inhibits autonomous operation as specified by safety controller.
- Verify static open-sky RTK FIX with measured horizontal error against a surveyed reference; record results and conditions. Do not promise a fixed centimeter error without field validation.

## Hardware-arrival checklist
Confirm LC29H(BS) printed marking and firmware, antenna L1/L5 support and connector, Pi Zero GPIO pinout/serial voltage, RTCM output capabilities, and actual HaLow network routing. Photograph labels and record firmware versions before implementing hardware-specific commands.
