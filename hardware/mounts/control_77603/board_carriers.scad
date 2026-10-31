// Board carriers: edge-capture sleds for the owned electronics.
//
// Where a board outline is unverified the carrier is an adjustable edge clamp, not a plate
// with invented holes. The Pi 5 carrier is the only one with a modelled hole pattern,
// because that pattern is documented.
include <parameters.scad>

module rounded_plate_2d(l, w, r) {
    hull() for (x = [-1, 1], y = [-1, 1])
        translate([x * (l / 2 - r), y * (w / 2 - r)]) circle(r = r);
}

module sled_plate(l, w, t = shelf_t_mm, r = corner_r_mm) {
    linear_extrude(height = t) rounded_plate_2d(l, w, r);
}

// Adjustable edge clamp: two rails the board slides between, with a printed wedge that
// takes up slack. No hole positions are assumed.
module edge_clamp(l, w, t = 12, rail = 5) {
    // BLIND captive recess, not a through-slot: cutting the clamp channel through the full
    // plate thickness severs the two rails into separate solids.
    difference() {
        sled_plate(l, w, t, 4);
        translate([-l / 2 + rail, -w / 2 + rail, t * 0.45])
            cube([l - 2 * rail, w - 2 * rail, t]);
        // Printed wedge slot along one edge (blind, so it cannot split the rail).
        translate([-l / 2 + rail, w / 2 - rail - 2, t * 0.6])
            cube([l - 2 * rail, 4, t * 0.4 + 0.5]);
    }
}

module carrier_pi5(part = "body") {
    if (part == "body") {
        difference() {
            sled_plate(pi5_pcb_mm[0] + 24, pi5_pcb_mm[1] + 24, shelf_t_mm, 6);
            // Documented hole pattern only.
            for (u = [-1, 1], v = [-1, 1])
                translate([u * pi5_holes_mm[0] / 2, v * pi5_holes_mm[1] / 2, -1])
                    cylinder(d = pi5_hole_d_mm, h = shelf_t_mm + 2);
            // Connector side access.
            translate([-pi5_pcb_mm[0] / 2 - 13, -8, -1])
                cube([6, 16, shelf_t_mm + 2]);
        }
    } else if (part == "post") {
        // Printed standoff for one corner, kept separate so it prints cleanly.
        difference() {
            cylinder(d = 7, h = 8);
            translate([0, 0, -1]) cylinder(d = pi5_hole_d_mm, h = 10);
        }
    } else {
        assert(false, str("pi5: unknown part ", part));
    }
}

module carrier_alta(part = "body") {
    edge_clamp(alta_capacity_mm[0] + 20, alta_capacity_mm[1] + 20, 12, 6);
}

module carrier_pca9685(part = "body") {
    edge_clamp(pca_capacity_mm[0] + 16, pca_capacity_mm[1] + 16, 10, 5);
}

module carrier_pololu_2806(part = "body") {
    // Edge capture only: the 0.97 in board outline and 0.086 in holes are published, but
    // their centre coordinates are not, so no hole is cut.
    if (part == "body") {
        difference() {
            sled_plate(pololu_pcb_mm[0] + 16, pololu_pcb_mm[1] + 16, 6, 3);
            translate([-pololu_pcb_mm[0] / 2, -pololu_pcb_mm[1] / 2, -1])
                cube([pololu_pcb_mm[0], pololu_pcb_mm[1], 8]);
            // Header/service relief above the board.
            translate([-pololu_pcb_mm[0] / 2 - 1, -3, 5])
                cube([pololu_pcb_mm[0] + 2, 6, 4]);
        }
    } else if (part == "riser") {
        // Spacer block that lifts the mux clear of the shelf for header access.
        difference() {
            translate([-12, -12, 0]) cube([24, 24, pololu_overhead_mm * 0.4]);
            translate([-6, -6, -1]) cube([12, 12, pololu_overhead_mm]);
        }
    } else {
        assert(false, str("pololu: unknown part ", part));
    }
}

module carrier_hiletgo(part = "body") {
    edge_clamp(hiletgo_capacity_mm[0] + 16, hiletgo_capacity_mm[1] + 16, 12, 6);
}

module carrier_converter(part = "body") {
    // One family for both the steering converters and the USB-C converters; the bay is
    // sized to the larger of the two capacities by the enclosure.
    edge_clamp(steering_converter_capacity_mm[0] + 16,
               steering_converter_capacity_mm[1] + 16, 12, 6);
}

module carrier_harness_clamp(part = "body") {
    if (part == "body") {
        difference() {
            sled_plate(harness_clamp_mm[0], harness_clamp_mm[1], 10, 5);
            for (k = [-1, 0, 1])
                translate([k * 26, -harness_clamp_mm[1] / 2 - 1, 3])
                    cube([14, harness_clamp_mm[1] + 2, 8]);
            // Strap land for a printed ratchet tie.
            translate([-harness_clamp_mm[0] / 2 - 1, -2, 5])
                cube([harness_clamp_mm[0] + 2, 4, 3]);
        }
    } else if (part == "strap") {
        // Printed ratchet tie: teeth engage the clamp land, no bought cable tie.
        union() {
            translate([-45, -2.4, 0]) cube([90, 4.8, 2.4]);
            for (k = [-8:8])
                translate([k * 5, -2.4, 2.4]) cube([2, 4.8, 1.6]);
        }
    } else {
        assert(false, str("harness: unknown part ", part));
    }
}

carrier_kind = "pi5";
carrier_part = "body";

module control_carrier(kind = carrier_kind, part = carrier_part) {
    if (kind == "pi5") carrier_pi5(part);
    else if (kind == "alta") carrier_alta(part);
    else if (kind == "pca9685") carrier_pca9685(part);
    else if (kind == "pololu_2806") carrier_pololu_2806(part);
    else if (kind == "hiletgo") carrier_hiletgo(part);
    else if (kind == "converter") carrier_converter(part);
    else if (kind == "harness_clamp") carrier_harness_clamp(part);
    else assert(false, str("unknown carrier kind: ", kind));
}

control_carrier(carrier_kind, carrier_part);
