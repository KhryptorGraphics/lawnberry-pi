# Toro 77603 — Starter motor, ignition, solenoid, and remote-start research

Date: 2026-10-30
Status: **Manufacturer part numbers verified for Toro serial >=418961000; engine-mounted starter OEM part and final harness pinout need individual engine and mower serial validation.**

## Manufacturer sources
1. Toro serial-specific catalog 3476-743A, model 77603, serial **418961000 and up**, Electrical Assembly printed p. 27 and Engine and Muffler printed p. 23: https://www.toro.com/getpub/265916
2. Toro operator manual 3473-450C, engine start p. 4-11 and wiring schematic ch. 8 p. 8-1, drawing G454156: https://www.toro.com/getpub/273204
3. Toro's model 77603 product specification identifies the 23 hp, 726 cc engine as **Kawasaki FR691V**: https://www.toro.com/en/product/77603
4. Official Kawasaki FR/FS/FX 651V/691V/730V service manual, starter system chapter 8, especially 8-5 and 8-6: https://www.kawasakipower.com/kengine/3-TechService/Home/Serviceman/99924-2093-06_FR_FS_FX651V_691V_730V_English_eBook_Rev.pdf
5. Kawasaki-spec example (NOT a confirmed match for this mower): FR691V-AS10 starter assembly listing 21163-0714 and 21163-0749: https://kawasaki.gardening-parts.com/kawa-gardening/FR-Petits%20Moteurs/FR691V/FR%20MOTORS/FR691V-AS10/STARTER-MOTOR/5010694/C_04/0/25310
6. Another engine variant FR691V-DS52 lists starter 21163-0743: https://kawasaki.gardening-parts.com/kawa-gardening/FR-Petits%20Moteurs/FR691V/FR-MOTORS/FR691V-DS52/STARTER-MOTOR/5013722/B_13/0/28022

## Toro-verified mower-side part numbers (serial >=418961000)
| Function | Toro OEM part | Catalog reference | Verification |
|---|---|---|---|
| Starter contactor / remote starter solenoid | **117-1197** | Electrical Assembly ref. 3 | Confirmed |
| Ignition switch / keyed switch | **137-4100** | ref. 5 | Confirmed |
| Auxiliary relay (role to be traced) | **1-643275** | ref. 8 | Confirmed part; function not assumed |
| Main wire harness | **145-6382** | ref. 1 | Confirmed |
| Factory safety switches (two) | **110-6765** | ref. 4 | Confirmed catalog quantity; specific in-circuit function requires tracing |
| Blade control/PTO rocker | **139-6368** | ref. 7 | Confirmed |
| Battery | **140-2076** | ref. 2 | Confirmed catalog number |
| Main harness fuse, 15A | **109355** | Harness ref. 7 | Confirmed |
| Main harness fuse, 25A | **109102** | Harness ref. 8 | Confirmed |
| Engine-mounted electric starter MOTOR | **Kawasaki OEM item 21163-xxxx, not specified in Toro catalog** | Engine parts source Kawasaki | Exact suffix **NOT confirmed** |

Kawasaki FR691V is identified in the Toro engine assembly as the FR691 family (p. 23, marked obtain parts from Kawasaki). A 2024 Toro unit may have a different Kawasaki engine specification than an FR691V-AS10 or DS52. **Do not order 21163-0714, 21163-0749, 21163-0743 or 21163-0791 as exact matches on model 77603 alone.** Need the Kawasaki engine ID/spec code and engine serial from its label.

## Technical breakdown and terminals (functional, not measured physical pin numbers)
- **Ignition/key switch 137-4100:** OFF, RUN, momentary START functions. START produces a start request into the mower's existing protected interlock path. Do not import the terminal layout B/S/M/G/A from an unrelated Toro model as verified for this part/connector.
- **Toro 117-1197 solenoid:** 12 V, four-terminal type per parts descriptions. Two **large power studs** are battery + feed and switched high-current starter output. Two **smaller control terminals** actuate the coil; aftermarket fitment describes it as *grounded through terminal / base not grounded*. Verify exact terminals and polarity on the actual original or OEM drawing; don't assume case ground.
- **Engine starter (Kawasaki):** high-current DC motor that spins the pinion/engine flywheel, normally supplied from the solenoid's switched stud with return through its mechanical engine/body ground. The OEM Kawasaki chapter 8 distinguishes the solenoid, starter motor, and start-switch lead.
- **Additional relay 1-643275:** present in Toro catalog, but do not label it as the starter relay until comparing the vehicle's serial-matched schematic.
- **Engine stop:** must be designed from the real FR691V ignition/kill and fuel-solenoid control wiring. Do not feed +12 V into a magneto kill/stop line. Keep physical key-stop and engine-kill controls independent of software.

## Baseline start / stop architecture — functional only, NOT INSTALLATION WIRING
```
  SBC (Pi 5 / Orange Pi 4 Pro / Libre Alta) intent: START / STOP
                 |
       isolated, hardware-limited
       starter request controller
                 |
         original key/interlock
        authorization path intact
                 |
            Toro 117-1197
        (12V coil, high-current
             contacts)
                 |
         Kawasaki starter motor
```
- User's objective: simplest remote start/stop under SBC control, but **all stock safety systems are retained** and verified. The design can request engine start only when the factory system itself allows it; no jumpers around seat, motion control, PTO, brake or other interlocks.
- A local physical enable/arm control, functional emergency stop, start authorization logic independent of network, and an engine-running signal are required before unattended automation is considered.
- **Hardware pulse limitation** independent of Linux software: Toro restricts starter engagement to **maximum 5 seconds**, **at least 15 seconds between failed attempts** (operator manual 3473-450C, p. 4-11). Require starter-off on lost Pi power, relay fault, reboot, broken wire, or missing engine-running feedback. Prefer a lower pulse limit as engineering testing shows sufficient.
- Do not reengage starter when engine is running. Provide cranking RPM feedback/verified run detection, restart lockout, maximum retry count, and explicit operator arming per start sequence.
- Engine stop first removes PTO authorization and engine run permission in an OEM-compatible manner, while maintaining emergency engine-stop functionality independent of the Pi. Verify actual fuel solenoid / magneto kill topology before selecting output type (e.g. normally closed vs normally open), polarity, isolation and protection.
- Do **not** wire the Pi GPIO directly to the 12 V solenoid, the starter high-current studs or ignition coils. An isolated automotive-rated driver with flyback/clamping and independent safety logic is needed.
- Design separate *conceptual*, *bench-validated*, and *installed-validated* states; do not claim deployed operation until full testing.

## What is still unknown
1. **Mower serial number** to select the correct electrical wiring revision / catalog range.
2. **Kawasaki FR691V-xxxx full engine model/spec and engine serial**, required for exact OEM 21163-xxxx starter motor.
3. Actual connector pin orientation, start request contacts and conductor colors at switch 137-4100, coil terminals at 117-1197 and relay 1-643275 as installed.
4. Engine RUN, kill, alternator charging and fuel-solenoid circuit details that can safely support a fail-safe remote stop, including which OEM protection functions are still present without a seated operator.
5. Rated coil current/inductive load, real cranking current, starter speed and post-crank de-energization requirements.

## Do not claim
- Do not say the exact Kawasaki starter has been identified by part number.
- Do not say a schematic pin equals a physical connector cavity without serial-specific trace and inspection.
- Do not bypass safety interlocks to achieve autonomous starting.
- Do not use external manuals from different Toro families for pin-by-pin instructions without cross-verification.


## Operator preference update — retry interval (2026-10-30)
- **Requested UI behavior:** Virtual ignition ON/RUN toggle; press-and-hold START requests cranking, and immediately releasing/canceling START must cease the request. OFF requests engine shutdown. No latched START action.
- **User's explicit requested retry delay:** **5 seconds**, **not 15 seconds**.
- **Manufacturer constraint that conflicts with this preference:** Toro's operator manual 3473-450C p. 4-11 specifically says **maximum 5 seconds of starter engagement per attempt**, then **wait 15 seconds after an unsuccessful start** before another attempt. Source: https://www.toro.com/getpub/273204 . The 5-second cranking cap and 15-second rest are *two different specifications*.
- **Implementation resolution:** Record the 5-second retry interval as the user's requested product setting, *not an approved hardware protection value*. Until starter thermal/electrical testing and qualified review justify a different value, enforce the manufacturer's **15-second minimum failed-start lockout**; never silently claim that five seconds complies with Toro. Show the actual enforced lockout in the UI. Request explicit engineering review before shortening it.
- Press-and-hold control must stop on pointerup, pointercancel, loss of client heartbeat, navigation away or disconnect. Independent maximum starter-on timer and interlock authorization are required.
