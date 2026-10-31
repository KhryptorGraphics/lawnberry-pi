# JSN-SR04T — Four-Sensor Ultrasonic Wiring Research (supersedes outdated three-HC-SR04 guide)

**Status:** Updated 2026-10-30. **Target is four waterproof JSN-SR04T modules**. Hardware revision numbers and actual GPIO assignments are not yet verified. Previous instructions for **three HC-SR04** are superseded. **Important correction:** Raspberry Pi **physical header pin 17 is 3.3 V, NOT 5 V**; using the prior table to power a 5 V sensor from pin 17 is wrong. Manufacturer reference: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html .

## Required electrical behavior

- Each sensor (subject to actual purchased board revision) uses nominal **regulated 5 V**, common reference ground, an input `TRIG`, and output `ECHO`. The 5 V ECHO must **never connect directly to a 3.3 V Raspberry Pi or Alta GPIO**.
- Add a voltage divider / logic-level translator per sensor ECHO; a representative resistor pair is **1 kΩ between ECHO and MCU input, 2 kΩ between input and ground**, yielding ~3.33 V when ECHO is 5 V. For extra margin choose a verified level-shift circuit or e.g. 1 kΩ plus 1.8 kΩ (~3.21 V). Verify supply range, divider behavior, the board's power-on pins and ground integrity.
- Example JSN manuals give approximate **25 cm near blind-zone** and up to **4.5 m** range on some versions; neither is guaranteed for every variant. Their broad beam and weather/terrain effects make them **supplementary sensing**, not primary human/pet safety proof.
- **No fixed GPIO assignments yet.** Reserve **eight distinct digital signal paths** for four independent TRIG/ECHO pairs (or a properly engineered alternative trigger/capture topology). Reconcile actual allocation with HaLow SPI (if used), Hailo, I²C PCA9685, relays, sonar, UART/GNSS and camera board interface. GPIO **BCM number ≠ physical pin number**.
- Avoid old guide GPIO 10/9/11 assignments until SPI overlaps are reconciled; the older guide was written for **three HC-SR04s** and includes power/pin errors.
- Use physically robust waterproof transducer mounting, protected electronics, decoupling, fused/limited 5 V sensor rail, and cable strain relief. Verify cross-talk; trigger sensors sequentially and reject stale/invalid echoes.

## Acceptance tasks

1. Read the exact JSN-SR04T PCB version/mode; identify signal protocol and actual voltage requirements.
2. Create a *single authoritative* per-board GPIO allocation matrix including each peripheral. Run repository pin conflict audit before connecting any new sensor.
3. Test one sensor with a verified level translator and 5 V supply on a dummy setup; measure output and dead zone.
4. Expand to four sensors with staggered ping times. Measure range, crosstalk, false positives/negatives, vibration and water ingress on a fixed test rig.
5. Do not treat ultrasonic inputs as an independent blade safety/interlock or proof of obstacle-free path. If sensor coverage is unknown or faulty, software should revoke autonomous movement permission.

Sources: [JSN module manual](https://www.openbikesensor.org/docs/other/collective-order/jsn-sr04t-en.pdf), [Raspberry Pi GPIO official documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html), [JSN manufacturer/vendor examples](https://shillehtek.com/blogs/shillehtek-product-manuals/jsn-sr04t-waterproof-ultrasonic-distance-sensor-arduino-esp32-manual).

See [continued manufacturer-source audit](2026-10-30-followup-hardware-research-findings.md).
