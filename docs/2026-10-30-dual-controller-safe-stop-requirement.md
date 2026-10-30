# SAF-ALT-001 — Dual-controller failure requires a complete safe stop

**Date:** 2026-10-30
**Status:** Mandatory design requirement; not implemented or hardware-validated.
**Scope:** Toro model 77603, Raspberry Pi 5 steering/navigation controller and Libre Alta engine/PTO/throttle controller over a **selectable peer link: Wi-Fi HaLow (Mode A) or direct wired Ethernet (Mode B)**.

## Requirement

If **either** the Raspberry Pi 5 **or** Libre Alta fails, loses power, freezes, reboots unexpectedly, becomes unable to control its assigned outputs, or loses the required intercontroller safety heartbeat, the mower **MUST immediately enter a latched SAFE_STOP state**, regardless of which controller originated the fault.

SAFE_STOP means, as independently verified physical outcomes:

- **Blade PTO:** disengage clutch authorization promptly; recognize that the blades coast and measure the actual time to zero rotation.
- **Starter:** remove cranking authorization; do not start or restart automatically.
- **Engine:** command and achieve engine shutdown via a validated OEM-compatible independent kill/stop path.
- **Steering and propulsion:** prevent further commanded motion and ensure both motion-control levers reach a validated stop/neutral condition. Merely dropping power to geared servos is NOT evidence of neutral: they can remain in their last position. Require a physically proven return/stop mechanism before powered autonomy.
- **Throttle:** return to safe idle/stop setting where still meaningful; engine shutdown remains required for this failure class.
- **Authorization:** revoke all run, blade and movement permission; latch until a person checks the fault and intentionally rearms. A reconnect or reboot alone MUST NOT resume operation.

## Architecture constraints

1. Each controller may drive its own assigned actuators locally. The selected peer link (HaLow or Ethernet) carries lightweight state/health messages, not real-time servo PWM.
2. Both hosts send periodic bounded-age heartbeat/state messages and independently monitor peer health. Missing, delayed, stale or invalid heartbeats MUST be treated as loss of authorization. Design and measure a timeout budget appropriate to the physical machine, not just the network bandwidth.
3. **A network message alone is NOT an independent safety mechanism.** Use a safety mechanism that remains effective when either host is dead, both are dead, HaLow fails, or the process/I2C bus locks up. It must not require functioning Linux, network connectivity or web UI to maintain the safe state.
4. Preserve OEM interlocks and physical emergency-stop effectiveness. No shorting of starter high-current terminals or bypassing seat/parking/PTO/other OEM safety contacts.
5. This fault response is **stricter than the older generic `emergency_stop()` contract that can leave the engine running**. For either-controller failures use SAFE_STOP with mandatory engine shutdown, not an engine-running soft stop.
6. Do not represent the mower as capable of safe autonomous operation until physical braking/neutral return, engine kill, blade spin-down and both-controller failures have been demonstrated on the actual machine. The current fore/aft steering linkage does not actuate the PARK brake; that is an unresolved design gate.

## Ultra-low-bandwidth peer monitoring over HaLow or Ethernet — owner requirement 2026-10-30

- Monitor both computers continuously using a **minimal, fixed-size, bidirectional health/authorization packet**; no camera frames, servo PWM streaming, or verbose dashboard logs on this safety heartbeat channel.
- **Initial test target:** <=24 application-payload bytes per message at **5 Hz per computer**, which equals 120 B/s = **960 bit/s in each direction** before UDP/IP, Wi-Fi MAC, encryption, retransmissions and management overhead. Measure real on-air utilization separately: it will be higher and signal conditions can increase it.
- Include protocol version, device identity, rolling counter, safety authorization/state and health/error bits. Authenticate traffic and detect stale/duplicate/replayed messages; a newly connected device is not automatically authorized.
- Treat **3 missed consecutive 200 ms heartbeats (~600 ms)** as an **initial candidate**, not a safety-certified delay. Set the final interval and watchdog threshold from measured network jitter plus allowable stopping risk, and independently demonstrate end-to-end shutdown timings.
- Compute peer freshness from **local monotonic time of last valid receipt**; do not assume synchronized device clocks. Keep heartbeat handling independent of web dashboards and nonessential telemetry.
- Loss of the **selected peer link** (radio in Mode A, wired Ethernet in Mode B), either computer, or both computers must result in a latched full-mower stop. Neither a wirelessly delivered STOP packet nor a Pi/Alta software watchdog is the sole protective measure: require an independent physical fail-closed safety mechanism.
- In **Mode A (HaLow)**, verify AP/client peer reachability and link-loss responses. In **Mode B (Ethernet)**, route **all Pi–Alta interconnect traffic** over a direct cable, leaving HaLow for separate remote/base/Thor traffic; measure cable-unplug, interface-down, and asymmetric packet-loss behavior. An external HaLow outage alone need not constitute loss of the Pi–Alta peer link in Ethernet mode. Do not automatically fail over between transports unless a separate design and test plan explicitly authorizes it.
- See [the two transport configurations and address/interface isolation examples](2026-10-30-intercontroller-ethernet-alternative.md).

## Acceptance tests (required; all currently unchecked)

- [ ] Kill power to the Pi 5 with the Alta operating: confirm starter disabled, PTO off, engine stops, and propulsion stops safely; record measured timings.
- [ ] Kill power to the Alta with the Pi 5 operating: confirm the same outcomes, including prevention of a held-last PWM steering command.
- [ ] Freeze each Linux host; simulate process crash, watchdog hang and repeated reboot for each.
- [ ] Disconnect the **selected peer link** in each direction (HaLow in Mode A, Ethernet in Mode B); inject stale/duplicated heartbeat messages; verify fail-closed behavior and no automatic resume.
- [ ] In Ethernet mode, verify peer traffic does not silently reroute over HaLow, and separately confirm that loss of the HaLow base-station uplink does not break healthy wired peer monitoring.
- [ ] Remove power to both hosts together; verify hardware still forces the safe state.
- [ ] Confirm E-stop and original mower interlocks continue to function independently.
- [ ] Confirm steering linkages' physical neutral/stop response during each test; power removal alone is not passing.
- [ ] Instrument actual clutch release and blade time to zero RPM, ignition-stop time, and loss-of-communication detection latency.
- [ ] Verify deliberate local rearm is required after every fault and reconnect.

**Implementation owner:** Claude Code / Lawnberry Pi safety and actuator workstreams. First create simulated failure tests, then implement and independently bench-test all paths before an engine-on evaluation.
