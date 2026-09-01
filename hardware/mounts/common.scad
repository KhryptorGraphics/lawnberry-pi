// Shared parameters and helper modules for the LawnBerry Pi zero-turn
// actuation mounts (Toro TimeCutter 50" conversion).
//
// EVERY dimension marked "MEASURE" is a placeholder. The Toro operator's
// manual (form 3465-589) carries no dimensioned drawings, so frame and
// lever sizes must come off the actual machine with calipers before you
// print anything structural. See README.md.

$fn = 64;

// ---------------------------------------------------------------- machine
// MEASURE: outside diameter of the frame tube the servo bracket clamps to.
frame_tube_od     = 38.1;   // 1.5" round tube, placeholder
// MEASURE: outside diameter of the lap-bar tube the pushrod clamp grips.
lapbar_tube_od    = 25.4;   // 1.0" round tube, placeholder
// MEASURE: wall thickness available for the throttle-servo mounting face.
panel_thickness   = 2.0;

// ----------------------------------------------------------------- servo
// DSSERVO RDS51150SG, from the manufacturer spec: 65 x 30 x 48 mm.
// The servo ships with U-shaped aluminium holders; these mounts bolt to
// those holders rather than gripping the servo body, which is both
// stronger and avoids trapping heat.
servo_body_l      = 65;
servo_body_w      = 30;
servo_body_h      = 48;
// MEASURE: hole spacing on the supplied U-bracket. Slots absorb error.
servo_bracket_hole_spacing_x = 50;
servo_bracket_hole_spacing_y = 20;

// --------------------------------------------------------------- fixings
m3_clear          = 3.4;
m4_clear          = 4.5;
m5_clear          = 5.5;
m6_clear          = 6.4;
m6_nut_af         = 10.0;   // across flats
m6_nut_h          = 5.2;
m5_nut_af         = 8.0;
m5_nut_h          = 4.7;

// ----------------------------------------------------------------- print
wall              = 5.0;    // structural wall; do not thin below 4 mm
clamp_wall        = 7.0;    // wall around clamped tubes
clearance         = 0.35;   // printer fit clearance, tune per machine

// ---------------------------------------------------------------- helpers

// Hex pocket for a captive nut, opening along +Z.
module nut_pocket(af, h) {
    cylinder(h = h, d = af / cos(30), $fn = 6);
}

// Slotted hole: lets you shim out mounting error without reprinting.
module slot(d, len, h) {
    hull() {
        translate([-len / 2, 0, 0]) cylinder(h = h, d = d);
        translate([ len / 2, 0, 0]) cylinder(h = h, d = d);
    }
}

// One half of a split tube clamp.
//
// Geometry: tube axis along Z. The split plane is X = 0, so this half
// occupies X >= 0 and the mating half is its mirror. Bolt ears stand off
// in +/-Y and the clamp bolts run along X through both halves. Any
// mounting boss extends in +X, clear of the ears.
//
// half = 0 -> the half carrying the mounting boss (bolt heads this side)
// half = 1 -> the plain closing half (captive nuts)
module tube_clamp_half(tube_od, width, ear_bolt = m6_clear,
                       ear_len = 16, half = 0) {
    r_in    = tube_od / 2 + clearance;
    r_out   = r_in + clamp_wall;
    ear_thk = ear_bolt + 6;              // ear thickness along X
    ear_yi  = r_in * 0.5;                // inner edge, buried in the barrel
    ear_yo  = r_out + ear_len;           // outer edge
    bolt_y  = r_out + ear_len * 0.5;
    nut_h   = (ear_bolt > 5) ? m6_nut_h : m5_nut_h;
    nut_af  = (ear_bolt > 5) ? m6_nut_af : m5_nut_af;
    difference() {
        union() {
            // Half barrel, X >= 0
            intersection() {
                difference() {
                    cylinder(h = width, r = r_out, center = true);
                    cylinder(h = width + 2, r = r_in, center = true);
                }
                translate([0, -r_out - 1, -width / 2 - 1])
                    cube([r_out + 1, 2 * r_out + 2, width + 2]);
            }
            // Bolt ears, both sides of the barrel
            for (s = [-1, 1])
                translate([ear_thk / 2,
                           s * (ear_yi + ear_yo) / 2,
                           0])
                    cube([ear_thk, ear_yo - ear_yi, width], center = true);
        }
        // Bore (re-cut so the ears cannot intrude)
        cylinder(h = width + 2, r = r_in, center = true);
        // Clamp bolt holes, axis along X
        for (s = [-1, 1])
            translate([0, s * bolt_y, 0])
                rotate([0, 90, 0])
                    cylinder(h = ear_thk * 4, d = ear_bolt, center = true);
        // Captive nuts in the closing half only
        if (half == 1)
            for (s = [-1, 1])
                translate([ear_thk + 0.01, s * bolt_y, 0])
                    rotate([0, -90, 0])
                        nut_pocket(nut_af, nut_h + 0.5);
    }
}
