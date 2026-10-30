# LawnBerry-Pi RTK base station hardware selection

Status: **conditional comparison only**. Existing owner-stated Quectel LC29H-BS/DA architecture must be checked before purchasing a replacement; receiver ownership and performance are not confirmed.

## Preferred first check: existing/planned Quectel LC29H-BS base + LC29H-DA rover

**Manufacturer confirmation:** The **LC29H-BS** explicitly supports **survey-in/fixed-coordinate RTCM base-station corrections**; **LC29H-DA** is an RTK-capable rover specified for 1 Hz RTK. Verify the owner's actual board inventory, correction message support, update-rate requirement and firmware before placing an order. A new ZED-F9P is **not automatically required**. [Quectel series hardware specification](https://www.quectel.com/content/uploads/2022/06/Quectel_LC29H_Series_Hardware_Design_V1.2.pdf); [LC29H BS RTCM protocol](https://quectel.com/content/uploads/2024/02/Quectel_LC29HBS_GNSS_Protocol_Specification_V1.0.pdf); [variant overview](https://www.quectel.com/product/gnss-lc29h/). See [continued manufacturer-source research](2026-10-30-followup-hardware-research-findings.md).

## Optional alternate receiver, only if a capability gap is established

**ArduSimple simpleRTK2B Budget, u-blox ZED-F9P, SKU AS-RTK2B-F9P-L1L2-NH-03**. Manufacturer: https://www.ardusimple.com/product/simplertk2b/ . This is a USB-connected receiver board, not a Raspberry Pi-specific GPIO HAT. It supports stationary base and rover modes, RTCM correction generation, multiband L1/L2 GNSS, and an SMA antenna connection. Manufacturer lists 172 EUR for the board as checked 2026-10-25; price and stock may change.

## Conditional items only if not already owned

- Compatible **survey-grade multiband L1/L2 GNSS antenna**, with appropriate SMA cable, suitable ground plane and weatherproof permanent outdoor mounting. Antenna is **not included** with this board.
- USB data/power cable appropriate to revision -03 (USB-C), weather-resistant enclosure, power supply, and surge/lightning protection as appropriate to installation.
- Communication link for RTCM corrections: local network/NTRIP or compatible telemetry radio; Wi-Fi HaLow is a separate transport decision, not an integrated feature of this receiver.
- **Separate RTK-capable receiver on mower (rover)**; a single base receiver does not give the mower RTK positioning.

## Proposed ZED-F9P alternative integration (NOT selected)

Stationary antenna -> ZED-F9P base -> USB -> Linux host (Orange Pi 4 Pro, Raspberry Pi 5, or other always-on computer) -> RTCM3 correction service/NTRIP caster -> authenticated network/HaLow transport -> mower rover receiver -> navigation controller. Prefer USB to avoid assuming interchangeable GPIO headers. Record stable serial device IDs, RTCM message set and correction latency; verify rover supports base message configuration.

## Configuration and acceptance criteria

1. Install base antenna at a fixed, surveyed location with clear sky visibility; record antenna reference point, height, coordinate datum and mounting details. Survey-in can establish a provisional position but does not guarantee absolute centimeter-level accuracy.
2. Configure ZED-F9P in stationary base mode, accurate fixed coordinates, and supported RTCM3 output (e.g., 1005/1006 and relevant MSM messages); confirm exact messages and rates against firmware and rover compatibility.
3. Test reception, correction age, RTK FIX stability, repeatability, multipath sensitivity and network interruptions across the property; record logs.
4. Do not use GNSS/RTK alone for personnel safety, emergency stopping or blade interlocks. Loss of valid corrections/position must invoke an independently validated safe behavior.

Sources: https://www.ardusimple.com/product/simplertk2b/ ; https://www.ardusimple.com/user-guide-simplertk2b-budget/ ; https://docs.ardupilot.org/copter/docs/common-ardusimple-rtk-gps-simplertk2b-budget.html
