# Dual-controller electrical model — conditional, not installation wiring

**Date:** 2026-10-30 · **Target:** Toro TimeCutter MAX Havoc MyRIDE 60 in, model **77603**, serial
**unknown** · **Status:** functional model + evidence register. **Nothing here is verified
installation wiring, and no Toro connector pin, wire colour, fuse rating or contact rating is
assigned.**

Everything below is derived from the committed requirements on
`docs/2026-10-30-throttle-servo-linkage`, the Pololu #2806 manufacturer guide
(<https://www.pololu.com/docs/0J60/all>), and the Toro/Kawasaki documents cited in the evidence
register. The executable counterpart of this document is `hardware/control/dual_controller_sim.py`
with its scenarios in `hardware/control/scenarios.py`.

---

## 1. Roles and authority

| Host | Normal role | On peer failure |
| --- | --- | --- |
| Raspberry Pi 5 | Sole normal motion/start/throttle authority | Latch fault, revoke permissions, request PTO OFF + engine OFF on **its own** outputs |
| Libre Alta | Hot standby, local peripheral host, **shutdown backup** | Latch fault, revoke permissions, request PTO OFF + engine OFF on **its own** outputs, and take the steering mux to neutral-only |

The Alta is never a second autonomous driver. Both hosts may hold RUN permission while healthy,
but only the Pi issues non-neutral steering or throttle-advance commands.

## 2. Steering signal path (signal only — no servo power through any logic board)

```text
Pi 5 PCA9685 @0x41, 50 Hz          Libre Alta PCA9685 (own bus, address TO VERIFY)
  CH0 ──────────────────────────► M1        CH0 ──────────────────────────► S1   NEUTRAL ONLY
  CH2 ──────────────────────────► M2        CH2 ──────────────────────────► S2   NEUTRAL ONLY
  CH1  (unused in this model)               CH1 ──────────────────────────► MG995 throttle
                                            CH3 ──────────────────────────► SEL
                                     │
                             Pololu #2806 (one assembled unit)
                             FAILMODE jumper OFF, inversion default, EN not driven
                                     │
                          OUT1 ─► left 150 kg servo SIGNAL
                          OUT2 ─► right 150 kg servo SIGNAL
                                  (each servo powered from its own owned converter,
                                   fused and sized separately — never through VM or PCA V+)
```

- **The Alta alone writes SEL.** The Pi never connects to SEL, so no two GPIOs are ever tied.
- 50 Hz, 1000 µs → master (Pi). 2000 µs → slave (Alta). Both are inside the manufacturer's
  500–2500 µs validity window and more than 64 µs from the 1696 µs threshold, so hysteresis cannot
  mis-select.
- **Alta S1/S2 are neutral-only by rule**, in software and in the model
  (`Simulation.steering_channel_write` refuses a non-neutral Alta write and counts it). A frozen
  valid slave SEL is then benign in the only way this architecture can make it benign: the slave
  inputs never carry anything but neutral.
- FAILMODE **OFF** is mandatory. FAILMODE ON and `EN` driven high both force outputs **low**, which
  is *no pulse*, **not** neutral. The mux has **no safe-start**.

## 3. Failure truth table (modelled)

`R` = requested, `M` = modelled electrical state, `P` = physical outcome.

| # | Condition | SEL at mux | OUT source | Steering command actually delivered | P |
| --- | --- | --- | --- | --- | --- |
| T1 | Both healthy, Pi driving | 1000 µs valid | Pi M1/M2 | Pi's commanded angle | unknown |
| T2 | Pi lost, Alta alive | 2000 µs valid | **Alta S1/S2** | neutral pulse only | unknown |
| T3 | Alta lost, Pi alive | pulses stop → invalid ≈130 ms | Pi M1/M2 | Pi continues | unknown |
| T4 | Alta frozen, SEL frozen 1000 µs | valid, below threshold | Pi M1/M2 | Pi continues | unknown |
| T5 | Alta frozen, SEL frozen 2000 µs | valid, above threshold | **Alta S1/S2** | neutral (neutral-only rule) | unknown |
| T6 | Mux supply lost | — | outputs low | **no signal — NOT neutral** | unknown |
| T7 | FAILMODE mis-set ON | any invalid | outputs low | **no signal — NOT neutral** | unknown |
| T8 | Pi PCA frozen, Pi still heartbeating | 1000 µs | Pi M1/M2 | **held angle persists** | unknown |
| T9 | Both hosts lost | — | low or frozen | **unprotected** | unknown |

T8 is the honest gap of a software-only architecture without an external watchdog: heartbeat
supervision cannot see a wedged-but-alive host.

## 4. PTO permission

```text
OEM-authorized clutch feed  [OEM fuse + OEM PTO rocker 139-6368 + OEM interlocks untouched]
   └─[ K_PI  NO, energize-to-allow ]─[ K_ALTA  NO, energize-to-allow ]─► OEM electric clutch

CLUTCH_PERMISSION = OEM_PTO_ALLOWED AND K_PI_closed AND K_ALTA_closed
```

| OEM PTO | K_PI | K_ALTA | Modelled feed |
| --- | --- | --- | --- |
| OFF | any | any | **OFF** |
| ON | closed | closed | permitted |
| ON | **open** | closed | **OFF** |
| ON | closed | **open** | **OFF** |
| ON | open | open | **OFF** |

Either host opens **its own** contact from a local request; no peer acknowledgement is needed and a
dead peer's output state cannot keep the clutch fed. This cannot engage PTO when the OEM rocker is
OFF, and it does not bypass any factory interlock.

## 5. Engine stop (conditional candidate)

```text
OEM kill node ──┬──[ K_KILL_PI  NC, energize-to-run ]──┐
                └──[ K_KILL_ALTA NC, energize-to-run ]──┴── chassis ground
```

The two contacts are **parallel**, because *either* host must be able to stop the engine. Series
contacts would require both hosts to close, so a living host would defeat the other's stop.

- Coil de-energised (host death, GPIO low, wire cut, coil-rail loss) → NC closes → kill node
  grounded → engine stops. That is the fail-safe direction for a **ground-to-kill** magneto.
- Consequence to accept explicitly: engine RUN becomes a **2-of-2 permission**. An Alta reboot
  therefore stops the engine — consistent with "Pi commands engine OFF when Alta fails", but it is a
  real behaviour change from any single-controller design.
- **Never apply +12 V to a kill lead.** The contact form (NC), coil voltage and engine-family
  compatibility are **BLOCKED** on inspection; Kawasaki FR691V ground-to-kill is documented in the
  branch research, not verified on this mower.
- A **stuck-open** NC contact is the dangerous direction: the engine would not stop through that
  path. The second path mitigates a single failure; both stuck open is unprotected.
- Starter is a separate function and is **not modelled**: the OEM PARK interlock requires the bars
  swung outboard, which the fore/aft linkage cannot do, so no autonomous crank is implemented.

## 6. Contact accounting (no invented inventory)

| Function | Poles | Ownership |
| --- | --- | --- |
| Steering-servo power enable | 1 | owner allocation retained; host ownership unresolved |
| Pi PTO permission | 1 | HiLetGo candidate — contact/rating **unverified** |
| Alta PTO permission | 1 | **CONDITIONAL** — not in verified inventory |
| Pi engine stop | 1 | **CONDITIONAL** — contact form/interface BLOCKED |
| Alta engine stop | 1 | **CONDITIONAL** — contact form/interface BLOCKED |
| Starter | 1 | separate function; not allocated |

Only **two** HiLetGo single-channel modules are confirmed, and one is intended for steering power.
The automotive harness is **not** verified to contain relay bodies or drivers, and Toro factory relay
1-643275 is part of the OEM circuit, **not** a spare. Software cannot create a pole.

## 7. Interface status

| Interface | Status |
| --- | --- |
| `pi_pto_permission` | owned candidate (HiLetGo, contact/rating unverified) |
| `alta_pto_permission` | CONDITIONAL |
| `pi_engine_stop`, `alta_engine_stop` | CONDITIONAL (contact form and engine interface BLOCKED) |
| `steering_power_relay` | owned allocation, host ownership unresolved |
| `starter` | not modelled |
| `pololu_2806` | owned; bench validation pending |

## 8. Evidence register

### KNOWN FROM SOURCE
- Toro operator manual 3473-450C: the **round knob above the ignition is the CHOKE**; the
  **rectangular lower-left rocker** is blade/PTO. Not a connector pinout.
- Pololu #2806 guide: one assembled unit; SEL needs a valid RC pulse (10–330 Hz, 0.5–2.5 ms);
  threshold 1696 µs ±64 µs; three good pulses to qualify; ≈130 ms no-activity invalidation;
  FAILMODE OFF → master, ON → outputs low; **no safe-start**; VM 2.5–16 V; OUT high ≈
  min(5 V, VM); SEL/M/OUT power share VM, S power shares separate VS.
- Pololu dimension drawing: board 0.97 in (24.638 mm) square, holes Ø0.086 in (2.1844 mm);
  **hole centre spacing not published**.
- Kawasaki FR691V service literature: closing the stop switch grounds the ignition primary, so
  engine OFF may be **ground-to-kill**, not +12 V interruption.
- Two HiLetGo single-channel modules and one automotive harness are owned; the harness contact
  count is unverified.
- `config/tractor.yaml` (unchanged by this work): PCA9685 0x41/50 Hz, left CH0, throttle CH1,
  right CH2, starter GPIO5, blade_pto GPIO26, `enabled: false`.

### HYPOTHESIS
- 3.3 V PCA9685 outputs are accepted by the mux at the chosen VM, and the servos accept the mux's
  output-high amplitude. Manufacturer publishes no input threshold; **bench gate**.
- A dedicated protected low-current `V_MUX_LOGIC` branch can be derived from an owned converter.
- The Alta's PCA9685 can be addressed on its own bus without colliding with the Pi's 0x41.

### TO MEASURE
- Mower serial and engine FR691V suffix; the matching OEM wiring revision.
- Rear of the PTO rocker: connector, pin markings, wire colours; switch contact map.
- Automotive harness: relay body count, coil voltage, contact form, pin count.
- HiLetGo: coil voltage, trigger polarity, NO/COM/NC labels, DC inductive suitability for the clutch.
- Clutch steady/transient current, polarity, suppression, OEM fuse.
- Converter output rails and spare current; total relay-coil demand.
- Pololu hole centres and installed height; mux switching at 3.3 V inputs; FAILMODE state.
- Throttle lever cross-section, rightward lean, engagement depth, travel, and the true datums behind
  the 40/44-inch figures.
- MG995 shaft XY, lug thickness, hole diameter, X-horn radius and hole size.
- Steering neutral repeatability, independent return-to-idle, and PARK reachability.
- Blade coast-down after clutch release; engine stop time after kill grounding.

### BLOCKED
- Any Toro pin assignment (serial-specific diagram not obtained).
- Engine-kill contact form and engine-family compatibility.
- Second PTO permission contact (inventory).
- Both-host failure, shared power loss, welded contacts and jammed linkage **by construction**.

## 9. Bench procedures (not executed here)

1. **Isolated dummy load**, engine off, blades disconnected, machine immobilized: put a low-voltage
   lamp across both PTO contacts in series with a dummy supply; remove Pi power → Alta must open its
   contact; repeat with Alta removed. Record both contact voltages, not just "it went out".
2. **Scope SEL**: capture 1000 µs and 2000 µs at the mux, confirm the output source changes, and
   confirm the mux returns to master ≈130 ms after SEL stops. Confirm FAILMODE jumper OFF.
3. **Stuck-contact simulation**: hold one contact closed mechanically; verify the other still opens
   and that the software reports the modelled feed rather than claiming a shutdown.
4. **Manual neutral and PARK**: with servo power off, hand-check lever self-centring; separately
   verify that fore/aft neutral is not PARK and that the OEM crank interlock still blocks.
5. **Throttle linkage**: measure force, retention slip, backlash and return-to-idle; run wear cycles;
   record thermal creep; then and only then consider live engine operation.
6. **Coast-down**: measure blade RPM decay after commanded clutch release; it is not instantaneous.
7. **Engine stop**: qualified technician only; measure time from kill grounding to stopped engine.

No live engine, clutch or relay experiment is authorized by this implementation.
