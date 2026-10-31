# Toro 77603 — dual-computer PTO OFF feasibility, relay allocation and verification gate

**Research date:** 2026-10-30.  
**Scope:** Pi 5 primary / Libre Alta backup, using existing purchased hardware wherever electrically verified.  
**Status:** **Conditional functional design only; no mower-specific relay pin assignments or installed circuit verified.** The mower serial number, clutch current and OEM connector pinout remain unknown.

## The owner requirement

If the Pi 5 crashes while blades are engaged, the **Libre Alta must still electrically request PTO OFF** and engine OFF without consulting the failed Pi. If the Alta fails, Pi 5 must independently request PTO OFF and engine OFF. On heartbeat link loss, each working controller revokes permission. The purchased **Pololu #2806 is steering PWM signal selection only**, never blade control. Blade release does **not** mean zero RPM immediately: blade coast-down must be measured.

## Verified context versus unknown physical details

- **Toro product model:** The manufacturer's parts portal identifies 60-inch Toro TimeCutter MAX Havoc MyRIDE **77603** in multiple **serial-dependent catalog ranges**, including 417600000–418960999 and 418961000–419002711. **Exact owner serial unknown**. Official catalog sources: https://www.toro.com/en/parts/partdetails?id=57303 and https://www.toro.com/en/parts/partdetails?id=58041 .
- **Operator control:** Toro publication 3473-450C identifies a **rectangular blade-control/PTO switch**. The round pull knob above ignition is **CHOKE**, not the PTO. Operator manual: https://www.toro.com/getpub/273204 . Do not treat panel photo identification as an electrical connector pinout.
- **Electrical function:** PTO clutch energization is distinct from engine RUN/kill and starting. A **series permission cutout** can disengage an **already OEM-authorized PTO clutch**; it **cannot engage** PTO when the factory switch is OFF, and must not bypass any interlock.
- **Owned relay inventory:** Two HiLetGo 5 V single-channel boards, identified by owner as ASIN B0CHFJSNP6, plus **one automotive relay harness** of unknown contact/relay count. One HiLetGo is intended for steering-servo power. That leaves **one confirmed independent switched contact** from the second HiLetGo, not two. **The uninspected harness does not establish that another relay/contact exists.**
- **Relay ratings are unresolved:** Reseller material describes the HiLetGo module as 5 V control and 30 A nominal contacts, but some catalog metadata inconsistently calls the coil 12 V; rely on the **actual module markings and controlled electrical measurements**. A claimed '30 A' resistive rating does not validate the inductive DC clutch release/suppression and vibration environment. The two computer GPIO outputs require independently appropriate coil drivers/inputs and signal isolation.

**Additional manufacturer verification:** Toro's 77603 OEM electrical-assembly parts catalog lists **factory relay 1-643275**, but this is part of the factory circuit, **not a spare independently owned extra relay contact**. Do not count it as closing the existing hardware shortfall. [Toro part catalog 3476-743A](https://www.toro.com/getpub/265916#page=27). Kawasaki's FR691V-family ignition is **ground-to-kill**; this cannot be derived from the separate PTO supply circuit. See [continued manufacturer-source research](2026-10-30-followup-hardware-research-findings.md).

## Proposed independent-stop **functional concept**, not a wire/pin diagram

Use **two independently controlled, de-energize-to-open, normally-open clutch-permission contact paths in series with the OEM-authorized clutch feed**, preserving all factory interlocks. Define the Pi's contact **K_PI** and the Alta's contact **K_ALTA**. Each controller alone may WITHDRAW authorization to engage PTO; neither controller alone may override the other or factory OFF.

```text
Toro factory-authorized PTO/clutch circuit
  [factory fuse/switch/interlocks remain active]
       |
       +---[K_PI normally-open permission]---[K_ALTA normally-open permission]---+
                                                                                |
                                                                     OEM electric PTO clutch
       (generic functional blocks only: actual feed/return, polarity,
        protection, circuit placement, and relay current ratings UNVERIFIED)
```

Logical authorization: `CLUTCH_PERMISSION = OEM_PTO_ALLOWED AND PI_PTO_ALLOWED AND ALTA_PTO_ALLOWED`. A **FALSE from either host** must remove clutch supply, subject to verified relay behavior. This isn't a complete autonomous PTO ON/OFF interface; factory controls still decide whether an engaged PTO is permitted.

| OEM allows | Pi contact K_PI | Alta contact K_ALTA | Result at series permission circuit |
| --- | --- | --- | --- |
| OFF | either | either | OFF |
| ON | CLOSED (permission) | CLOSED (permission) | PTO permitted (not independently guaranteed energized) |
| ON | OPEN | CLOSED | OFF |
| ON | CLOSED | OPEN | OFF |
| ON | OPEN | OPEN | OFF |

### Failure behavior targeted

- **Pi process crashes, locks up, or Pi loses power while PTO ON:** Alta detects missing valid Pi heartbeat, locally revokes its own PTO permission and causes **K_ALTA to open**, even if Pi output or K_PI was left energized.
- **Alta crashes or loses power:** Pi detects loss and opens **K_PI**; if Alta relay itself loses coil power, its normally-open contact opens as well.
- **Peer link fails:** Both survivors locally revoke; either independent contact should block clutch current.
- **One contact sticks/welds CLOSED:** The other still must physically open on fault; verify fault and loads in tests. **Both stuck closed, an improperly powered relay, lost independent stop interface, or a miswired bypass are NOT handled by software.**
- **Either SBC restarts:** startup must not re-energize its permission until both are healthy, fault re-arm is deliberate, and OEM permissions are satisfied. Manufacturer interlocks remain active.

### Why existing relays cannot yet be approved as sufficient

| Required switching job | Number of independently controlled circuits or other means | Ownership / assessment |
| --- | --- | --- |
| Steering servo power enable (owner's proposed allocation) | 1 | One HiLetGo proposed; steering pulse selection remains handled by the purchased Pololu #2806. Cutting servo power does NOT guarantee neutral. |
| Pi-controlled independent PTO permission contact | 1 | Candidate: remaining HiLetGo if coil and contact ratings verified. |
| Alta-controlled independent PTO permission contact | 1 | **NOT confirmed in inventory**. May be supplied by an appropriately rated **installed relay** in the already purchased automotive harness; an empty socket or wire harness alone does not count. Otherwise a properly matched extra relay, driver, contact protection and wiring are conditionally required. |
| Engine OFF / ignition kill paths | Independent Pi and Alta paths needed | **Not allocated** here. Engine kill can be *ground-to-kill* rather than interrupt +12 V; use engine/OEM schematic. Not safely inferred from PTO circuit. |
| Starter momentary control | Separately controlled/interlocked circuit | **Not allocated** here. Do not bridge starter motor current terminals or bypass PARK/PTO interlocks. |

**Preliminary verdict:** With one HiLetGo reserved for steering, **the two confirmed HiLetGo modules alone do not provide both required independent PTO contacts**, let alone separate starter and engine-kill interfaces. If the existing automotive harness includes at least one suitable **actual relay + contact + compatible coil driver**, a second PTO contact *may* already be available; the harness has **not** been verified to include one. Do not order anything until photographed and counted. A relay contact dedicated to PTO is **not automatically reusable for engine kill or starter**.

## Implementation handoff (non-deployable until verified)

- In the Pi process, implement `pi_pto_permission` and independent `request_pto_off()` using only Pi-owned outputs.
- In Alta process, independently implement `alta_pto_permission` and `request_pto_off()` using only Alta-owned outputs.
- On peer failure, simultaneous steering-neutral command and immediate PTO OFF / engine OFF **requests**: do not sequence PTO release *after* steering has physically moved.
- Fault latch: peer recovery never silently re-enables clutch. Remote GUI and Thor are not in local stop-command path. Proposed heartbeat **200 ms** and timeout **3 missed = ~600 ms** are candidates only; they do not specify blade stopping time.
- Unit/bench simulation must cover: Pi dies while PTO permitted; Alta dies; peer network partition; GPIO stuck-high; relay stuck-closed (other contact opens); one output fails to open; relay coil power removed; boots/reset; factory PTO OFF; both run authorization and e-stop requests.
- Bench isolate a **low-voltage test load**, with actual mower disabled and qualified electrical oversight, before any clutch-power integration. Capture voltage/current across **both** actual contacts and clutch only with a qualified technician; test vibration, current transients, coil power and fusing. No live blade testing until OEM integrity and coast-down are inspected.

## Required proof before assigning real connector terminals

1. Read **mower serial number** and match the applicable Toro electrical diagram/service part catalog.
2. Get clear photos of **rear of rectangular PTO rocker** with molded terminal IDs and connected wire colors, and **existing automotive relay harness with any relay bodies and readable markings**.
3. Record the existing **HiLetGo board labels** and controlled measurements: trigger polarity, coil supply, NO/COM contact default, and safe 3.3 V input drive; record clutch supply and inductive-load rating.
4. Verify clutch power polarity, current (steady and transient), suppression, upstream/downstream OEM fuses/interlocks and connector wiring using the serial-matched factory schematic.
5. Separately map **engine kill / RUN** and **starter interlock**, never assume they share PTO polarity or contacts.
6. Verify the Pololu/Alta can command steering neutral when Pi stops, servo motor power remains available for movement, and bars actually reach neutral. Engine stopped alone does not prove mower has stopped moving.

### Source notes

- Toro 77603 parts portal by serial range: https://www.toro.com/en/parts/partdetails?id=57303 and https://www.toro.com/en/parts/partdetails?id=58041 .
- Toro operator manual 3473-450C: https://www.toro.com/getpub/273204 .
- Pololu 2806 manufacturer guide: https://www.pololu.com/docs/0J60/all .
- HiLetGo owner-provided pack reference: https://www.amazon.com/dp/B0CHFJSNP6 . Product ratings still need physical verification.
- [Project PTO switch research](2026-10-30-toro-77603-pto-switch-research.md); [dual-computer control requirements](2026-10-30-software-only-dual-controller-failover-design.md).

**Bottom line:** **Functionally feasible** as dual-contact, stop-dominant PTO **permission** logic; **hardware completeness and Toro 77603 pin assignment not confirmed.** Do not claim the Libre Alta can already turn off engaged blades until actual independently functioning PTO contacts have been installed and measured.
