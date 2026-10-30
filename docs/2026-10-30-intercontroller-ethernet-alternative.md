# Pi 5 ↔ Libre Alta intercontroller network — parallel Ethernet configuration

**Date:** 2026-10-30  
**Status:** Candidate configuration requested by owner; neither wiring nor software is implemented or validated.  
**Purpose:** Preserve the original Wi-Fi HaLow peer link design while offering an alternative in which **all traffic between the Raspberry Pi 5 and Libre Alta is transported over wired Ethernet**.

## Selectable configurations

| Mode | Intercontroller data path | Wi-Fi HaLow use | Notes |
| --- | --- | --- | --- |
| **A — HaLow peer interconnect (existing)** | Pi 5 ↔ HaLow network ↔ Alta | Peer heartbeats, state, coordination; telemetry/base uplinks where available | Minimal bandwidth, but behavior depends on wireless topology/link conditions. |
| **B — Ethernet-only intercontroller link (new alternative)** | Pi 5 Ethernet ↔ direct cable ↔ Alta Ethernet | Remains available for links to the base station, remote UI, Thor, and other off-mower endpoints; **no Pi–Alta control/status traffic over HaLow** | Local peer traffic does not depend on HaLow access point or radio quality. |

**Mode B covers all Pi–Alta traffic**: heartbeat/health packets, safety/fault state, authorization, coordination/commands, acknowledgement, diagnostics, and peer telemetry or log transfers if needed. Do not transmit the same control message over both networks by default. The choice of intercontroller transport must be explicit and testable; a hybrid automatic-failover mode is *not* part of this option unless specified and tested separately.

## Mode B — minimum-cost physical topology

```text
    RPi 5 Ethernet port  ← Cat5e/Cat6 patch cable →  Libre Alta Ethernet port
             |                                          |
       local GPIO/I2C                                local GPIO/I2C
             |                                          |
         steering                                  throttle / relays
             |                                          |
     HaLow card → base/remote network       HaLow card → base/remote network
```

- Use the two boards' Ethernet interfaces and **one ordinary Cat5e or Cat6 patch cable**; a switch or additional Ethernet adapter is not required when both ports are available and working. Verify link negotiation on the installed boards.
- Keep the separate HaLow cards installed for external communications if required. Their continued presence does not obligate Pi–Alta traffic to traverse Wi-Fi HaLow.
- Example *isolated* IPv4 addresses (change if they conflict with actual LAN/HaLow subnets): **Pi 5 = 10.73.56.1/30**, **Alta = 10.73.56.2/30**. Do not set a second default gateway on this private link. Bind the peer-control service to the Ethernet interface or peer addresses; prevent accidental routing of peer traffic over HaLow.
- Protect connectors and cable against vibration, moisture, strain, abrasion, and accidental unplugging.

## Traffic and monitoring — identical application protocol in either mode

- Both endpoints continue to send **one 200 ms heartbeat interval (5 Hz)** each. Initial payload target **≤24 application bytes** per heartbeat = **120 B/s or 960 bit/s per sender**, excluding UDP/IP and Ethernet or HaLow framing.
- Use the same versioned packet format, sender identity, rolling sequence, authenticated authorization/state, and fault flags on either transport. Evaluate age against the receiver's monotonic clock, and reject replayed/stale/invalid traffic.
- **Three missed 200 ms heartbeat opportunities (~600 ms)** is a *proposed detection threshold*, not a verified stopping time; include scheduler delay, network delay, relay actuation, engine shutdown, and actual blade/drive stopping in measurements. Bandwidth is not the limiting factor.
- Keep dashboard/status updates separable (e.g., 1 Hz) from priority peer health monitoring. If Ethernet is selected, do not make a HaLow outage by itself look like peer failure when Pi and Alta are still connected by Ethernet; external-control-loss policy is a separate requirement.
- A disconnected cable, failed Ethernet interface, peer application freeze, or lost peer heartbeats must not be interpreted as permission to continue autonomous operation. Do not silently change to HaLow without an explicitly specified and validated failover policy.

## Roles, system limitations, and tests

- Retain Pi 5 as primary coordinator with Alta authorized to detect Pi 5 failure and request shutdown; Pi detects Alta failure likewise. The **dual computer software design and electrical relay/actuator authority are distinct from transport choice**.
- Ethernet improves local link predictability; it does **not** by itself prove safe stopping. An unresponsive controller can leave last-command PWM active, and remote software cannot undo a stuck mechanical drive linkage. Safety and stop hardware remain unresolved validation matters in other project documents.
- Before choosing Mode B for an operating mower, test real 5 Hz peer packets both directions, under load; unplug the cable; kill power to each host separately; freeze each peer process; inject stale/duplicate messages, unidirectional packet loss, and network flooding; record fault-detection, commanded stop, and physically measured neutral/blade/engine stopping times.
- Implementation should expose `intercontroller_transport = halow | ethernet` (illustrative setting), with separate network addresses/interfaces and a common health protocol, so both modes can be selected without rewriting mower control logic.

**Selection:** Both Mode A and Mode B remain **proposed supported configurations**. No transport mode is declared deployed or certified by this document.
